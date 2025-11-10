"""
VLM OCR Wrapper

This CLAMS app uses a user-specified Hugging Face image-to-text model to transcribe text
from representative frames of TimeFrame annotations.

Enhanced with DSPy integration for optimized prompts and few-shot learning.
"""

import argparse
import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image
from transformers import pipeline
from transformers import AutoModel, AutoTokenizer  # for prompt-capable models like DeepSeek-OCR

from clams import ClamsApp, Restifier
from mmif import Mmif, View, Annotation, Document, AnnotationTypes, DocumentTypes
from mmif.utils import video_document_helper as vdh

# Import DSPy integration (optional - only used if useDSPy=True)
try:
    import dspy
    from dspy_integration import DSPyHFVLM, OCRModule, ArtifactLoader
    DSPY_AVAILABLE = True
except ImportError:
    DSPY_AVAILABLE = False
    dspy = None


class VlmOcr(ClamsApp):

    def __init__(self):
        super().__init__()
        self._pipeline_cache = {}
        self._prompt_model_cache = {}
        self._dspy_module_cache = {}  # Cache for loaded DSPy modules
        self._dspy_lm_cache = {}  # Cache for DSPy LM instances

    def _appmetadata(self):
        # using metadata.py
        pass

    def _get_pipeline(self, model_id: str):
        if model_id in self._pipeline_cache:
            return self._pipeline_cache[model_id]
        try:
            # image-to-text pipeline works for OCR models like TrOCR
            pl = pipeline("image-to-text", model=model_id)
        except Exception as e:
            self.logger.error(f"Failed to initialize image-to-text pipeline for '{model_id}': {e}")
            raise
        self._pipeline_cache[model_id] = pl
        return pl

    def _get_prompt_model(self, model_id: str, local_model_path: Optional[str] = None):
        if model_id in self._prompt_model_cache:
            return self._prompt_model_cache[model_id]
        try:
            if local_model_path:
                print(f"[vlm-ocr] Using local model at {local_model_path}")
                tok = AutoTokenizer.from_pretrained(local_model_path, trust_remote_code=True, local_files_only=True)
                mdl = AutoModel.from_pretrained(local_model_path, trust_remote_code=True, local_files_only=True, use_safetensors=True)
            else:
                tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
                mdl = AutoModel.from_pretrained(model_id, trust_remote_code=True, use_safetensors=True)

            # don't force .cuda(); many environments are CPU-only
            mdl = mdl.eval()
        except Exception as e:
            self.logger.error(f"Failed to load prompt-capable model '{model_id}': {e}")
            raise
        self._prompt_model_cache[model_id] = (tok, mdl)
        return tok, mdl

    def _load_dspy_module(self, model_id: str, artifact_path: Optional[str] = None, dataset_name: Optional[str] = None) -> Optional['OCRModule']:
        """
        Load a DSPy OCR module from an artifact.

        Returns None if DSPy is not available or artifact not found.
        """
        if not DSPY_AVAILABLE:
            self.logger.warning("DSPy not available. Install with: pip install dspy-ai")
            return None

        cache_key = f"{model_id}_{artifact_path}_{dataset_name}"
        if cache_key in self._dspy_module_cache:
            self.logger.info(f"Using cached DSPy module for {model_id}")
            return self._dspy_module_cache[cache_key]

        # Initialize artifact loader
        artifacts_dir = Path(__file__).parent / "artifacts"
        loader = ArtifactLoader(artifacts_dir=artifacts_dir)

        # Find or load artifact
        try:
            if artifact_path:
                # Use specific artifact file
                artifact_file = Path(artifact_path)
                if not artifact_file.is_absolute():
                    artifact_file = artifacts_dir / artifact_file
            else:
                # Auto-discover artifact
                artifact_file = loader.find_artifact(model_id, dataset_name)

            if artifact_file is None or not artifact_file.exists():
                self.logger.info(f"No DSPy artifact found for model={model_id}, dataset={dataset_name}")
                return None

            self.logger.info(f"Loading DSPy artifact from {artifact_file}")

            # Load the module
            module = loader.load_artifact(artifact_file)

            # Cache it
            self._dspy_module_cache[cache_key] = module

            return module

        except Exception as e:
            self.logger.error(f"Failed to load DSPy module: {e}", exc_info=True)
            return None

    def _get_or_create_dspy_lm(self, model_id: str) -> Optional['DSPyHFVLM']:
        """Get or create a DSPy LM interface for the model."""
        if not DSPY_AVAILABLE:
            return None

        if model_id in self._dspy_lm_cache:
            return self._dspy_lm_cache[model_id]

        try:
            lm = DSPyHFVLM(model_id=model_id)
            self._dspy_lm_cache[model_id] = lm
            return lm
        except Exception as e:
            self.logger.error(f"Failed to create DSPy LM: {e}")
            return None

    def _run_ocr_with_dspy(self, img: np.ndarray, model_id: str, dspy_module: 'OCRModule') -> str:
        """
        Run OCR using a DSPy module with optimized prompts.
        """
        try:
            # Convert numpy array to PIL Image
            if isinstance(img, np.ndarray):
                if img.ndim == 3 and img.shape[2] == 3:
                    pil_image = Image.fromarray(img[:, :, ::-1].astype(np.uint8))
                elif img.ndim == 2:
                    pil_image = Image.fromarray(img.astype(np.uint8)).convert('RGB')
                else:
                    pil_image = Image.fromarray(img.astype(np.uint8))
            else:
                pil_image = img

            # Save to temp file and create dspy.Image
            from tempfile import NamedTemporaryFile
            with NamedTemporaryFile(suffix='.jpg', delete=False) as tf:
                pil_image.save(tf.name, format='JPEG')
                dspy_image = dspy.Image(path=tf.name)

                # Run through DSPy module
                prediction = dspy_module(image=dspy_image)

                # Extract text based on module mode
                if hasattr(prediction, 'transcription'):
                    text = prediction.transcription
                elif hasattr(prediction, 'result'):
                    text = prediction.result
                elif hasattr(prediction, 'parsed_fields'):
                    # For structured mode, serialize fields as JSON
                    import json
                    text = json.dumps(prediction.parsed_fields, indent=2)
                else:
                    text = str(prediction)

            # Cleanup temp file
            import os
            try:
                os.unlink(tf.name)
            except:
                pass

            self.logger.debug(f"DSPy OCR result length: {len(text)}")
            return text

        except Exception as e:
            self.logger.error(f"DSPy OCR failed: {e}", exc_info=True)
            return ""

    def _run_ocr(self, img: np.ndarray, model_id: str, prompt_text: Optional[str], use_dspy: bool = False,
                 dspy_module: Optional['OCRModule'] = None, local_model_path: Optional[str] = None) -> str:
        print(f"[vlm-ocr] Running OCR with model '{model_id}' on image shape: {getattr(img, 'shape', None)}")

        # DSPy path - use optimized prompts and few-shot learning
        if use_dspy and dspy_module is not None:
            self.logger.info(f"Using DSPy-optimized module for OCR")
            return self._run_ocr_with_dspy(img, model_id, dspy_module)

        # Traditional path - direct model inference
        model_id_str = str(model_id).lower()
        is_prompt_model = any(x in model_id_str for x in [
            'deepseek-ocr', 'deepseek_ai', 'deepseek-ai/deepseek-ocr',
            'llava', 'qwen-vl', 'idefics', 'smol-vlm', 'minigpt', 'blip-2'
        ])

        # Prompt-capable flow
        if is_prompt_model:
            if prompt_text is None or str(prompt_text).strip() == '':
                prompt_text = 'Describe the image'
            try:
                tok, mdl = self._get_prompt_model(model_id, local_model_path=local_model_path)
                # DeepSeek-OCR expects a path; save PIL image to a temp file
                pil_image = img
                if isinstance(img, np.ndarray):
                    if img.ndim == 3 and img.shape[2] == 3:
                        pil_image = Image.fromarray(img[:, :, ::-1].astype(np.uint8))
                    elif img.ndim == 2:
                        pil_image = Image.fromarray(img.astype(np.uint8)).convert('RGB')
                    else:
                        pil_image = Image.fromarray(img.astype(np.uint8))
                from tempfile import NamedTemporaryFile
                with NamedTemporaryFile(suffix='.jpg', delete=True) as tf:
                    pil_image.save(tf.name, format='JPEG')
                    # Many prompt VLMs with trust_remote_code expose .infer(tokenizer, prompt=..., image_file=...)
                    if hasattr(mdl, 'infer'):
                        res = mdl.infer(tok, prompt=prompt_text, image_file=tf.name, output_path=' ',
                                        base_size=1024, image_size=640, crop_mode=True,
                                        save_results=False, test_compress=False)
                        # DeepSeek-OCR infer may return a dict or structured object
                        if isinstance(res, dict):
                            # Try common keys for OCR output
                            text = res.get('text', res.get('result', res.get('output', str(res))))
                        elif isinstance(res, (list, tuple)) and len(res) > 0:
                            text = str(res[0]) if isinstance(res[0], str) else str(res)
                        else:
                            text = str(res) if not isinstance(res, str) else res
                        # Ensure we have a string
                        text = text.strip() if text else ''
                    else:
                        # Fallback: try standard pipeline if available
                        pl = self._get_pipeline(model_id)
                        outputs = pl(pil_image)
                        text = outputs[0].get('generated_text', '') if outputs else ''
            except Exception as e:
                self.logger.error(f"Prompt-model inference failed for '{model_id}': {e}")
                text = ''
            print(f"[vlm-ocr] OCR text length (prompted): {len(text) if isinstance(text, str) else 'N/A'}")
            return text

        # Image-only flow (e.g., TrOCR)
        pl = self._get_pipeline(model_id)
        pil_image = img
        if isinstance(img, np.ndarray):
            try:
                if img.ndim == 3 and img.shape[2] == 3:
                    pil_image = Image.fromarray(img[:, :, ::-1].astype(np.uint8))
                elif img.ndim == 2:
                    pil_image = Image.fromarray(img.astype(np.uint8)).convert('RGB')
                else:
                    pil_image = Image.fromarray(img.astype(np.uint8))
            except Exception as e:
                print(f"[vlm-ocr] Failed to convert numpy array to PIL: {e}")
        outputs = pl(pil_image)
        if not outputs:
            print("[vlm-ocr] OCR pipeline returned no outputs")
            return ""
        top = outputs[0]
        text = top.get('generated_text', str(top))
        print(f"[vlm-ocr] OCR text length: {len(text) if isinstance(text, str) else 'N/A'}")
        return text

    def _process_time_annotation(self, mmif: Mmif, representative: Annotation, new_view: View,
                                 video_doc: Document, model_id: str, prompt_text: Optional[str],
                                 use_dspy: bool = False, dspy_module: Optional['OCRModule'] = None,
                                 local_model_path: Optional[str] = None) -> Tuple[int, Optional[str]]:
        print(f"[vlm-ocr] Processing representative type: {representative.at_type}")
        if representative.at_type == AnnotationTypes.TimePoint:
            rep_frame_index = vdh.convert(representative.get("timePoint"),
                                          representative.get("timeUnit", "ms"), "frame",
                                          video_doc.get("fps"))
            image: np.ndarray = vdh.extract_frames_as_images(video_doc, [rep_frame_index], as_PIL=False)[0]
            timestamp = vdh.convert(representative.get("timePoint"),
                                    representative.get("timeUnit", "ms"), "ms", video_doc.get("fps"))
        elif representative.at_type == AnnotationTypes.TimeFrame:
            image: np.ndarray = vdh.extract_representative_frame(mmif, representative, as_PIL=False, first_only=True)
            timestamp = vdh.convert(vdh.get_representative_framenum(mmif, representative),
                                    'f', 'ms', video_doc.get("fps"))
        else:
            self.logger.error(f"Representative annotation type {representative.at_type} is not supported.")
            return -1, None

        text_content = self._run_ocr(image, model_id=model_id, prompt_text=prompt_text,
                                      use_dspy=use_dspy, dspy_module=dspy_module, local_model_path=local_model_path).strip()
        print(f"[vlm-ocr] OCR result at timestamp {timestamp}ms: '{text_content}'")
        if not text_content:
            print(f"[vlm-ocr] Empty OCR result at timestamp {timestamp}")
            return timestamp, None

        text_document: Document = new_view.new_textdocument(text=text_content)
        td_id = text_document.id
        source_id = representative.id
        new_view.new_annotation(AnnotationTypes.Alignment, source=source_id, target=td_id)

        return timestamp, text_content

    def _annotate(self, mmif: Mmif, **parameters) -> Mmif:
        video_doc: Document = mmif.get_documents_by_type(DocumentTypes.VideoDocument)[0]

        # parameters handling (values are lists)
        tf_labels = parameters.get("tfLabel", [])
        model_param = parameters.get("hfModel", ["microsoft/trocr-base-printed"])  # default
        model_id = model_param[0] if isinstance(model_param, list) and model_param else model_param
        # Default prompt depends on model type - will be set per-model in _run_ocr
        prompt_param = parameters.get("prompt", [""])  # default empty, will use model-specific default
        prompt_text = prompt_param[0] if isinstance(prompt_param, list) and prompt_param else prompt_param

        # DSPy parameters
        use_dspy_param = parameters.get("useDSPy", [False])
        use_dspy = use_dspy_param[0] if isinstance(use_dspy_param, list) else use_dspy_param
        use_dspy = bool(use_dspy)  # Ensure boolean

        dspy_artifact_param = parameters.get("dspyArtifact", [""])
        dspy_artifact = dspy_artifact_param[0] if isinstance(dspy_artifact_param, list) and dspy_artifact_param else dspy_artifact_param

        dspy_dataset_param = parameters.get("dspyDataset", [""])
        dspy_dataset = dspy_dataset_param[0] if isinstance(dspy_dataset_param, list) and dspy_dataset_param else dspy_dataset_param

        local_model_path_param = parameters.get("localModelPath", [""])
        local_model_path = local_model_path_param[0] if isinstance(local_model_path_param, list) and local_model_path_param else local_model_path_param

        # Load DSPy module if requested
        dspy_module = None
        if use_dspy:
            self.logger.info("DSPy mode enabled - attempting to load optimized module")
            
            # First, try to load the module to get the model_id from the artifact
            loader = ArtifactLoader(artifacts_dir=Path(__file__).parent / "artifacts")
            artifact_file = None
            if dspy_artifact:
                artifact_file = Path(dspy_artifact)
                if not artifact_file.is_absolute():
                    artifact_file = loader.artifacts_dir / artifact_file
            else:
                artifact_file = loader.find_artifact(model_id, dspy_dataset)

            if artifact_file and artifact_file.exists():
                artifact_data = loader.get_artifact_metadata(artifact_file)
                model_id = artifact_data.get("model_id", model_id)
                self.logger.info(f"Using model_id from artifact: {model_id}")

            # Configure DSPy LM
            dspy_lm = self._get_or_create_dspy_lm(model_id)
            if dspy_lm and DSPY_AVAILABLE:
                dspy.settings.configure(lm=dspy_lm)

            # Load module
            dspy_module = self._load_dspy_module(
                model_id,
                artifact_path=dspy_artifact if dspy_artifact else None,
                dataset_name=dspy_dataset if dspy_dataset else None
            )

            if dspy_module is None:
                self.logger.warning("DSPy module not loaded - falling back to traditional OCR")
                use_dspy = False
            else:
                self.logger.info("DSPy module loaded successfully")
                # Override prompt with DSPy's optimized prompt
                prompt_text = None  # DSPy module handles prompts internally

        # Determine if model is prompt-capable
        model_id_str = str(model_id).lower()
        is_prompt_model = any(x in model_id_str for x in [
            'deepseek-ocr', 'deepseek_ai', 'deepseek-ai/deepseek-ocr',
            'llava', 'qwen-vl', 'idefics', 'smol-vlm', 'minigpt', 'blip-2'
        ])

        # Enforce: if a non-empty prompt is provided and model is not prompt-capable -> error
        if (prompt_text is not None and str(prompt_text).strip() != '') and not is_prompt_model:
            raise ValueError(f"Provided prompt for a non-prompt model '{model_id}'. Choose a prompt-capable model or clear the 'prompt' parameter.")
        frame_interval_param = parameters.get("frameInterval", ["290"])  # default every 290 frames
        try:
            frame_interval = int(frame_interval_param[0] if isinstance(frame_interval_param, list) else frame_interval_param)
        except Exception:
            frame_interval = 290

        new_view: View = mmif.new_view()
        # Register what this view will contain BEFORE signing so 'contains' is populated
        new_view.new_contain(DocumentTypes.TextDocument)
        new_view.new_contain(AnnotationTypes.Alignment)
        new_view.new_contain(AnnotationTypes.TimePoint)
        self.sign_view(new_view, parameters)

        views_for_doc = mmif.get_views_for_document(video_doc.id)
        print(f"[vlm-ocr] Video document location: {video_doc.get('location')}, fps: {video_doc.get('fps')}")
        print(f"[vlm-ocr] frameInterval: {frame_interval}, using existing views: {len(views_for_doc)}")

        # Helper to process provided time annotations from an existing view
        def process_timeframes_from_view(view: View) -> bool:
            nonlocal mmif, new_view, video_doc, model_id, tf_labels
            found_any = False
            for timeframe in view.get_annotations(AnnotationTypes.TimeFrame):
                if 'label' not in timeframe:
                    self.logger.debug(f'Found a time frame "{timeframe.id}" without label, skipping.')
                    continue
                if tf_labels and timeframe.get("label") not in tf_labels:
                    continue
                found_any = True
                self.logger.debug(f'Processing time frame "{timeframe.id}" with label: "{timeframe.get("label")}"')

                text_content = None
                # process representatives if present
                if 'representatives' in timeframe.properties:
                    for rep_id in timeframe.get("representatives"):
                        if Mmif.id_delimiter not in rep_id:
                            rep_id = f'{view.id}{Mmif.id_delimiter}{rep_id}'
                        representative = mmif[rep_id]
                        timestamp, text_content = self._process_time_annotation(
                            mmif, representative, new_view, video_doc, model_id, prompt_text, use_dspy, dspy_module, local_model_path)
                # fallback to middle frame
                if text_content is None:
                    timestamp, text_content = self._process_time_annotation(
                        mmif, timeframe, new_view, video_doc, model_id, prompt_text, use_dspy, dspy_module, local_model_path)
                self.logger.debug(f'Processed timepoint: {timestamp} ms, recognized text: "{text_content}"')
            # Also check for explicit TimePoint annotations
            for timepoint in view.get_annotations(AnnotationTypes.TimePoint):
                found_any = True
                timestamp, text_content = self._process_time_annotation(
                    mmif, timepoint, new_view, video_doc, model_id, prompt_text, use_dspy, dspy_module, local_model_path)
                self.logger.debug(f'Processed timepoint: {timestamp} ms, recognized text: "{text_content}"')
            return found_any

        used_existing_annotations = False
        if views_for_doc:
            # prefer the latest view but accept any that contain time annotations
            for candidate_view in reversed(views_for_doc):
                if process_timeframes_from_view(candidate_view):
                    used_existing_annotations = True
                    break

        if not used_existing_annotations:
            # No TimePoint/TimeFrame annotations found → sample every N frames across the video
            self.logger.info(f'No TimePoint/TimeFrame found. Sampling every {frame_interval} frames as fallback.')
            print(f"[vlm-ocr] Fallback sampling every {frame_interval} frames (first minute only)")
            fps = video_doc.get("fps")
            if fps:
                max_frames_for_1_minute = int(fps * 60)  # 60 seconds worth of frames
                print(f"[vlm-ocr] Processing first minute only: max {max_frames_for_1_minute} frames at {fps} fps")
            else:
                max_frames_for_1_minute = 1800  # fallback: assume 30fps * 60 seconds
                print(f"[vlm-ocr] FPS not available, using fallback limit of {max_frames_for_1_minute} frames")
            frame_index = 0
            while True:
                # Check if we've exceeded the first minute
                if frame_index > max_frames_for_1_minute:
                    print(f"[vlm-ocr] Reached first minute limit at frame {frame_index}, stopping")
                    break

                try:
                    print(f"[vlm-ocr] Attempting to extract frame {frame_index}")
                    images = vdh.extract_frames_as_images(video_doc, [frame_index], as_PIL=False)
                    if not images:
                        print(f"[vlm-ocr] No image returned for frame {frame_index}, stopping")
                        break
                    image = images[0]
                except Exception as e:
                    # assume we've reached beyond the last frame
                    self.logger.debug(f'Stopping sampling at frame {frame_index}: {e}')
                    print(f"[vlm-ocr] Exception extracting frame {frame_index}: {e}")
                    break

                # create a TimePoint annotation in this view for traceability
                try:
                    timestamp_ms = vdh.convert(frame_index, 'f', 'ms', fps)
                except Exception:
                    # if conversion fails due to missing fps, keep frame index in frames
                    timestamp_ms = frame_index
                    time_unit = 'f'
                else:
                    time_unit = 'ms'

                tp_ann = new_view.new_annotation(AnnotationTypes.TimePoint, timePoint=timestamp_ms, timeUnit=time_unit)
                # Reuse existing processing to OCR and align
                _ = self._process_time_annotation(mmif, tp_ann, new_view, video_doc, model_id, prompt_text, use_dspy, dspy_module, local_model_path)

                # advance
                frame_index += frame_interval

        return mmif

def get_app():
    """
    Create an instance of the app class.
    """
    return VlmOcr()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", action="store", default="5000", help="set port to listen")
    parser.add_argument("--production", action="store_true", help="run gunicorn server")
    # add more arguments as needed
    # parser.add_argument(more_arg...)

    parsed_args = parser.parse_args()

    # create the app instance
    # if get_app() call requires any "configurations", they should be set now as global variables
    # and referenced in the get_app() function. NOTE THAT you should not change the signature of get_app()
    app = get_app()

    http_app = Restifier(app, port=int(parsed_args.port))
    # for running the application in production mode
    if parsed_args.production:
        http_app.serve_production()
    # development mode
    else:
        app.logger.setLevel(logging.DEBUG)
        http_app.run()
