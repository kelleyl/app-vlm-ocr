"""
VLM OCR App

A CLAMS app that uses vision-language models for OCR on video frames.
Supports multiple VLM backends including Qwen2-VL and others.
Optional DSPy integration for optimized prompts.
"""

import argparse
import logging
import yaml
from pathlib import Path
from typing import Optional

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

from clams import ClamsApp, Restifier
from mmif import Mmif, View, Document, AnnotationTypes, DocumentTypes
from mmif.utils import video_document_helper as vdh

# Import DSPy integration (optional)
try:
    import dspy as dspy_lib
    from dspy_clams import DSPyHFVLM, OCRModule, ArtifactLoader
    DSPY_AVAILABLE = True
except ImportError:
    DSPY_AVAILABLE = False
    dspy_lib = None


# Tested models list
TESTED_MODELS = [
    "Qwen/Qwen2-VL-2B-Instruct",
    "Qwen/Qwen2-VL-7B-Instruct",
    # Add dots.ocr when tested
]


class VlmOcr(ClamsApp):

    def __init__(self):
        super().__init__()
        self.model = None
        self.processor = None
        self.current_model_id = None
        self._dspy_module_cache = {}
        self._dspy_lm_cache = {}

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

    def get_combined_prompt(self, label: str, parameters: dict) -> str:
        """Get combined system and user prompt for a given label."""
        system_prompt = self.get_system_prompt(label, parameters)
        user_prompt = self.get_prompt(label, parameters)

        if system_prompt and user_prompt:
            return f"{system_prompt}\n\n{user_prompt}"
        elif system_prompt:
            return system_prompt
        elif user_prompt:
            return user_prompt
        else:
            return ""

    def _load_model(self, model_id: str):
        """Load a VLM model and processor."""
        if self.current_model_id == model_id:
            self.logger.info(f"Model {model_id} already loaded")
            return

        self.logger.info(f"Loading model: {model_id}")
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

        # Load the appropriate model class based on model type
        if "qwen" in model_id.lower() and "vl" in model_id.lower():
            # Qwen2-VL and Qwen3-VL use the same model class
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_id,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16
            )
        else:
            # Fallback to AutoModel for other VLMs
            from transformers import AutoModelForVision2Seq
            self.model = AutoModelForVision2Seq.from_pretrained(
                model_id,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16
            )

        self.model.eval()
        self.current_model_id = model_id
        self.logger.info(f"Model {model_id} loaded successfully")

    def _run_ocr_basic(self, image: Image.Image, prompt: str, system_prompt: str = "") -> str:
        """Run OCR using direct model inference.

        Args:
            image: PIL Image to process
            prompt: User prompt (combined system+user if system_prompt not provided separately)
            system_prompt: Optional system prompt to prepend as separate message
        """
        # Check if this is a Qwen VL model (requires chat message format)
        is_qwen_vl = "qwen" in self.current_model_id.lower() and "vl" in self.current_model_id.lower()

        if is_qwen_vl:
            # Qwen VL models require chat message format with vision tokens
            # Save image to temp file for processing
            import tempfile
            import os

            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tf:
                image.save(tf.name, format='JPEG')
                temp_image_path = tf.name

            try:
                # Build messages array
                messages = []

                # Add system message if provided (content is just a string for system messages)
                if system_prompt:
                    messages.append({
                        "role": "system",
                        "content": system_prompt
                    })

                # Add user message with image and text
                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "image", "image": f"file://{temp_image_path}"},
                        {"type": "text", "text": prompt}
                    ]
                })

                # Apply chat template
                text_input = self.processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )

                # Process with image
                inputs = self.processor(
                    text=[text_input],
                    images=[image],
                    return_tensors="pt"
                )

                # Generate text
                with torch.no_grad():
                    generated_ids = self.model.generate(**inputs, max_new_tokens=500)

                # Trim input tokens from output
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]

                # Decode the output
                generated_text = self.processor.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False
                )[0].strip()

            finally:
                # Clean up temp file
                try:
                    os.unlink(temp_image_path)
                except:
                    pass
        else:
            # Standard VLM processing (non-Qwen models)
            inputs = self.processor(images=image, text=prompt, return_tensors="pt")

            # Generate text
            with torch.no_grad():
                generated_ids = self.model.generate(**inputs, max_new_tokens=500)

            # Decode the output
            generated_text = self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True
            )[0].strip()

        return generated_text

    def _run_ocr_with_dspy(self, image: Image.Image, dspy_module: 'OCRModule') -> str:
        """Run OCR using a DSPy-optimized module."""
        import tempfile
        import os

        # Save image to temp file for DSPy
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tf:
            image.save(tf.name, format='JPEG')
            dspy_image = dspy_lib.Image(path=tf.name)

            try:
                prediction = dspy_module(image=dspy_image)

                # Extract text based on prediction structure
                if hasattr(prediction, 'transcription'):
                    text = prediction.transcription
                elif hasattr(prediction, 'result'):
                    text = prediction.result
                else:
                    text = str(prediction)

                return text.strip()
            finally:
                try:
                    os.unlink(tf.name)
                except:
                    pass

    def _load_dspy_module(self, model_id: str, artifact_path: Optional[str] = None) -> Optional['OCRModule']:
        """Load a DSPy artifact for optimized prompts."""
        if not DSPY_AVAILABLE:
            self.logger.warning("DSPy not available")
            return None

        cache_key = f"{model_id}_{artifact_path}"
        if cache_key in self._dspy_module_cache:
            return self._dspy_module_cache[cache_key]

        artifacts_dir = Path(__file__).parent / "artifacts"
        loader = ArtifactLoader(artifacts_dir=artifacts_dir)

        try:
            if artifact_path:
                artifact_file = Path(artifact_path)
                if not artifact_file.is_absolute():
                    artifact_file = artifacts_dir / artifact_file
            else:
                # Auto-discover artifact
                artifact_file = loader.find_artifact(model_id)

            if not artifact_file or not artifact_file.exists():
                self.logger.info(f"No DSPy artifact found for {model_id}")
                return None

            self.logger.info(f"Loading DSPy artifact: {artifact_file}")
            module = loader.load_artifact(artifact_file)

            # Configure DSPy with this model
            dspy_lm = DSPyHFVLM(model_id=model_id)
            dspy_lib.settings.configure(lm=dspy_lm)

            self._dspy_module_cache[cache_key] = module
            return module

        except Exception as e:
            self.logger.error(f"Failed to load DSPy module: {e}")
            return None

    def _annotate(self, mmif: Mmif, **parameters) -> Mmif:
        self.logger.debug(f"Annotating with parameters: {parameters}")

        # Load config file if specified
        config_file = parameters.get('config')
        self.logger.debug(f"config_file: {config_file}")
        if config_file:
            config_dir = Path(__file__).parent
            config_file_path = config_dir / config_file
            config = self.load_config(config_file_path)

            # Config overrides parameters
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
        else:
            config = {}

        # Extract parameters (handle both CLI and web app formats)
        def get_param(key, default):
            val = parameters.get(key, default)
            # CLI passes direct values, web app passes lists
            if isinstance(val, list):
                return val[0] if val else default
            return val

        model_id = get_param("hfModel", "Qwen/Qwen2-VL-2B-Instruct")
        use_dspy = get_param("useDSPy", False)
        dspy_artifact = get_param("dspyArtifact", "")

        # tfLabel can be multivalued
        tf_labels = parameters.get("tfLabel", [])
        if not isinstance(tf_labels, list):
            tf_labels = [tf_labels]

        # Load model
        self._load_model(model_id)

        # Load DSPy module if requested
        dspy_module = None
        if use_dspy:
            dspy_module = self._load_dspy_module(model_id, artifact_path=dspy_artifact)
            if dspy_module:
                self.logger.info("Using DSPy-optimized prompts")
            else:
                self.logger.warning("DSPy module not loaded, using basic prompts")

        # Create new view
        new_view: View = mmif.new_view()
        self.sign_view(new_view, parameters)
        new_view.new_contain(DocumentTypes.TextDocument)
        new_view.new_contain(AnnotationTypes.Alignment)

        # Get video document
        video_doc: Document = mmif.get_documents_by_type(DocumentTypes.VideoDocument)[0]

        # Find TimeFrame annotations from upstream app (e.g., swt-detection)
        timeframes = []
        for view in mmif.get_all_views_contain(AnnotationTypes.TimeFrame):
            for tf in view.get_annotations(AnnotationTypes.TimeFrame):
                # Filter by label if specified
                if tf_labels and tf.get("label") not in tf_labels:
                    continue

                # Ensure timeUnit is set (default to milliseconds if missing)
                if not tf.get("timeUnit"):
                    tf.add_property("timeUnit", "milliseconds")
                    self.logger.debug(f"Added default timeUnit=milliseconds to {tf.id}")

                timeframes.append(tf)

        if not timeframes:
            self.logger.warning("No TimeFrame annotations found")
            return mmif

        self.logger.info(f"Processing {len(timeframes)} timeframes")

        # Process each timeframe
        for i, timeframe in enumerate(timeframes, 1):
            try:
                self.logger.info(f"Processing timeframe {i}/{len(timeframes)}: {timeframe.id}")

                # Extract representative frame
                frame_image = vdh.extract_representative_frame(
                    mmif, timeframe, as_PIL=True, first_only=True
                )

                # Get prompt based on timeframe label
                label = timeframe.get("label", "default")
                system_prompt = self.get_system_prompt(label, parameters)
                user_prompt = self.get_prompt(label, parameters)

                self.logger.info(f"Using system prompt: {system_prompt[:100] if system_prompt else 'None'}...")
                self.logger.info(f"Using user prompt: {user_prompt[:100]}...")

                # Run OCR
                if dspy_module:
                    text = self._run_ocr_with_dspy(frame_image, dspy_module)
                else:
                    text = self._run_ocr_basic(frame_image, user_prompt, system_prompt)

                if not text:
                    self.logger.warning(f"No text extracted from {timeframe.id}")
                    continue

                self.logger.info(f"Extracted text: {text[:100]}{'...' if len(text) > 100 else ''}")

                # Create text document
                text_document = new_view.new_textdocument(text=text)

                # Create alignment
                source_id = timeframe.get("representatives", [timeframe.id])[0]
                alignment = new_view.new_annotation(AnnotationTypes.Alignment)
                alignment.add_property("source", source_id)
                alignment.add_property("target", text_document.long_id)

                self.logger.info(f"Created text document {text_document.id} aligned to {source_id}")

                # DEBUG: Break after first timeframe for testing
                self.logger.info("DEBUG: Breaking after first timeframe for testing")
                break

            except Exception as e:
                self.logger.error(f"Error processing {timeframe.id}: {e}")
                continue

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
