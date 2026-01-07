"""
VLM OCR App

A CLAMS app that uses vision-language models for OCR on video frames.
Supports multiple backends: HuggingFace, MLX, Ollama, vLLM. # note llm utils will be moved to clams_python
"""

import argparse
import logging
import time
import yaml
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

import tqdm

from clams import ClamsApp, Restifier
from mmif import Mmif, View, Document, AnnotationTypes, DocumentTypes
from mmif.utils import video_document_helper as vdh

from llm_utils import LocalVLMClient, GenerationParams


class VlmOcr(ClamsApp):

    def __init__(self):
        super().__init__()
        self.llm = LocalVLMClient()

    def _appmetadata(self):
        # Using metadata.py
        pass

    def load_config(self, config_file):
        """Load configuration from YAML file."""
        with open(config_file, 'r') as f:
            return yaml.safe_load(f)

    def get_prompt(self, label: str, parameters: dict) -> str:
        """Get user prompt for a given label."""
        if 'promptMap' in parameters and parameters['promptMap']:
            for mapping in parameters['promptMap']:
                if ':' in mapping:
                    map_label, map_prompt = mapping.split(':', 1)
                    if map_label == label:
                        return map_prompt
        if 'defaultPrompt' in parameters:
            return parameters['defaultPrompt']
        return ""

    def get_system_prompt(self, label: str, parameters: dict) -> str:
        """Get system prompt for a given label."""
        if 'systemPromptMap' in parameters and parameters['systemPromptMap']:
            for mapping in parameters['systemPromptMap']:
                if ':' in mapping:
                    map_label, map_prompt = mapping.split(':', 1)
                    if map_label == label:
                        return map_prompt
        if 'defaultSystemPrompt' in parameters:
            return parameters['defaultSystemPrompt']
        return ""

    def get_prompts(self, label: str, parameters: dict) -> Tuple[str, str]:
        """Get system and user prompts separately for a given label."""
        system_prompt = self.get_system_prompt(label, parameters) or ""
        user_prompt = self.get_prompt(label, parameters) or ""
        return system_prompt, user_prompt

    def _annotate(self, mmif: Mmif, **parameters) -> Mmif:
        self.logger.debug(f"Annotating with parameters: {parameters}")

        # Extract parameters (handle both CLI and web app formats)
        def get_param(key, default=None):
            val = parameters.get(key, default)
            if isinstance(val, list):
                return val[0] if val else default
            return val

        # Load config file if specified
        config_file = get_param('config')
        if config_file:
            config_dir = Path(__file__).parent
            config_file_path = config_dir / config_file
            if config_file_path.exists():
                config = self.load_config(config_file_path)
                if 'default_prompt' in config:
                    parameters['defaultPrompt'] = config['default_prompt']
                if 'custom_prompts' in config:
                    prompt_map = []
                    for label, prompt in config['custom_prompts'].items():
                        prompt_map.append(f"{label}:{prompt}")
                    parameters['promptMap'] = prompt_map
                if 'default_system_prompt' in config:
                    parameters['defaultSystemPrompt'] = config['default_system_prompt']
                if 'custom_system_prompts' in config:
                    system_prompt_map = []
                    for label, prompt in config['custom_system_prompts'].items():
                        system_prompt_map.append(f"{label}:{prompt}")
                    parameters['systemPromptMap'] = system_prompt_map

        # Get model from parameters (default comes from metadata.py)
        model_id = get_param("hfModel")
        
        # allTargets parameter - process all targets in TimeFrames instead of just representative
        all_targets = get_param("allTargets", False)
        if isinstance(all_targets, str):
            all_targets = all_targets.lower() in ('true', '1', 'yes')

        # tfLabel can be multivalued - filter TimeFrames by label
        tf_labels = parameters.get("tfLabel", [])
        if not isinstance(tf_labels, list):
            tf_labels = [tf_labels] if tf_labels else []
        # Case-insensitive label filtering
        tf_labels = [l.lower() for l in tf_labels]

        # appUri filtering (from metadata.py)
        app_uri = get_param('appUri', '')

        # Create new view
        new_view: View = mmif.new_view()
        self.sign_view(new_view, parameters)
        new_view.new_contain(DocumentTypes.TextDocument)
        new_view.new_contain(AnnotationTypes.Alignment)

        # Get video document
        video_docs = mmif.get_documents_by_type(DocumentTypes.VideoDocument)
        if not video_docs:
            self.logger.error("No VideoDocument found in MMIF")
            return mmif
        video_doc: Document = video_docs[0]

        # Find ALL TimeFrame annotations
        # Using a more robust search that handles version mismatches better
        all_views = mmif.get_all_views_contain(AnnotationTypes.TimeFrame)
        if not all_views:
            # Fallback: check all views manually if get_all_views_contain failed
            self.logger.debug("get_all_views_contain returned nothing, trying manual view search")
            all_views = [view for view in mmif.views if any(str(ann.at_type).endswith('TimeFrame') for ann in view.annotations)]
        
        timeframes = []
        
        found_labels = set()
        total_tf_found = 0
        
        for view in all_views:
            # Filter by appUri if specified
            if app_uri and app_uri not in view.metadata.app:
                continue
                
            for tf in view.get_annotations(AnnotationTypes.TimeFrame):
                total_tf_found += 1
                # Filter by label if specified
                label = tf.get_property('label')
                if label:
                    found_labels.add(label)
                    
                if tf_labels:
                    if not label or label.lower() not in tf_labels:
                        continue
                
                # Ensure timeUnit is set
                tf.add_property('timeUnit', 'milliseconds')
                timeframes.append(tf)

        if not timeframes:
            self.logger.warning(f"No TimeFrame annotations found matching filters (appUri='{app_uri}', tfLabels={tf_labels})")
            self.logger.info(f"Total timeframes in MMIF: {total_tf_found}")
            if found_labels:
                self.logger.info(f"Labels found in MMIF: {found_labels}")
            return mmif

        self.logger.info(f"Found {len(timeframes)} timeframes to process")

        # Build task list
        tasks: List[Dict[str, Any]] = []
        frames_to_extract: List[int] = []
        
        # Cache for view lookups
        view_cache = {}
        
        def get_target_framenum(target_id: str) -> Optional[int]:
            """Resolve target ID to frame number."""
            vid = target_id.split(':')[0]
            if vid not in view_cache:
                try:
                    view_cache[vid] = mmif.get_view_by_id(vid)
                except:
                    self.logger.warning(f"Could not find view for target {target_id}")
                    return None
            
            view = view_cache[vid]
            if not view:
                return None
            
            try:
                ann = view.annotations.get(target_id)
                if not ann:
                    return None
                if ann.at_type == AnnotationTypes.TimePoint:
                    ms = vdh.convert_timepoint(mmif, ann, 'milliseconds')
                    return vdh.millisecond_to_framenum(video_doc, ms)
                return None
            except Exception as e:
                self.logger.error(f"Error resolving target {target_id}: {e}")
                return None

        for timeframe in timeframes:
            label = timeframe.get_property('label') or 'default'
            prompt = self.get_prompts(label, parameters)
            
            if all_targets:
                # Process all targets within the TimeFrame
                targets = timeframe.get_property('targets')
                if targets:
                    for target_id in targets:
                        framenum = get_target_framenum(target_id)
                        if framenum is not None:
                            tasks.append({
                                'prompt': prompt,
                                'source': target_id,
                                'origin': timeframe.long_id,
                                'label': label,
                            })
                            frames_to_extract.append(framenum)
            else:
                # Just use representative frame
                framenum = vdh.get_representative_framenum(mmif, timeframe)
                tasks.append({
                    'prompt': prompt,
                    'source': timeframe.long_id,
                    'origin': timeframe.long_id,
                    'label': label,
                })
                frames_to_extract.append(framenum)

        if not tasks:
            self.logger.info("No tasks generated from timeframes.")
            return mmif

        self.logger.info(f"Generated {len(tasks)} tasks, extracting {len(frames_to_extract)} frames")

        # Extract all frames at once
        self.logger.info("Extracting frames from video...")
        start_extract = time.time()
        all_images = vdh.extract_frames_as_images(video_doc, frames_to_extract, as_PIL=True)
        extract_time = time.time() - start_extract
        self.logger.info(f"Extracted {len(all_images)} frames in {extract_time:.2f}s")

        # Process each task
        self.logger.info(f"Processing {len(tasks)} frames with model {model_id}...")
        start_process = time.time()
        processed = 0
        
        for i, (task, image) in enumerate(tqdm.tqdm(zip(tasks, all_images), total=len(tasks))):
            try:
                system_prompt, user_prompt = task['prompt']
                
                # Skip if prompt is "-"
                if (user_prompt or "").strip() == "-":
                    continue

                # Run OCR
                text = self.llm.generate(
                    model=str(model_id),
                    image=image,
                    user_prompt=user_prompt,
                    system_prompt=system_prompt,
                    params=GenerationParams(max_new_tokens=500),
                )

                if not text:
                    continue

                # Create text document
                text_document = new_view.new_textdocument(
                    text=text,
                    document=video_doc.id,
                    origin=task['origin'],
                    provenance='derived',
                )

                # Create alignment
                alignment = new_view.new_annotation(AnnotationTypes.Alignment)
                alignment.add_property("source", task['source'])
                alignment.add_property("target", text_document.long_id)
                
                processed += 1

            except Exception as e:
                self.logger.error(f"Error processing task {i}: {e}")
                continue

        process_time = time.time() - start_process
        total_time = extract_time + process_time
        fps = len(tasks) / process_time if process_time > 0 else 0
        
        self.logger.info(f"Processed {processed}/{len(tasks)} frames")
        self.logger.info(f"Frame extraction: {extract_time:.2f}s")
        self.logger.info(f"OCR processing: {process_time:.2f}s")
        self.logger.info(f"Total time: {total_time:.2f}s")
        self.logger.info(f"Processing rate: {fps:.2f} frames/sec")

        return mmif


def get_app():
    """Factory function to create app instance."""
    return VlmOcr()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5000, help="Port to listen on")
    parser.add_argument("--production", action="store_true", help="Run in production mode")
    args = parser.parse_args()

    app = get_app()
    http_app = Restifier(app, port=args.port)

    if args.production:
        http_app.serve_production()
    else:
        app.logger.setLevel(logging.DEBUG)
        http_app.run()
