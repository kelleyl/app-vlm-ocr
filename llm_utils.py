"""
Local LLM/VLM utilities (LiteLLM-style wrapper)

This module provides a unified interface for calling vision-language models
from multiple backends: HuggingFace Transformers, Ollama, vLLM, and MLX.

Usage:
    client = LocalVLMClient()
    text = client.generate(
        model="hf:Qwen/Qwen2-VL-2B-Instruct",  # or "ollama:llava" or "mlx:..."
        image=pil_image,
        user_prompt="Transcribe text in this image.",
    )

Backends:
    - hf: HuggingFace Transformers (default, runs locally)
    - ollama: Ollama server (http://localhost:11434)
    - vllm: vLLM server (http://localhost:8000)
    - mlx: mlx_vlm server for Apple Silicon (http://localhost:8080)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, Any, List
import base64
import io

import torch
from PIL import Image


class Backend(Enum):
    """Supported LLM backends."""
    HUGGINGFACE = "hf"
    OLLAMA = "ollama"
    VLLM = "vllm"
    MLX = "mlx"


@dataclass
class TestedModel:
    """A tested model configuration."""
    name: str
    backend: Backend
    model_id: str
    requires_gpu: bool = False
    notes: str = ""


# Tested models - these have been validated to work
TESTED_MODELS: List[TestedModel] = [
    # HuggingFace models (fully tested)
    TestedModel(
        name="qwen2-vl-2b",
        backend=Backend.HUGGINGFACE,
        model_id="Qwen/Qwen2-VL-2B-Instruct",
        requires_gpu=False,
        notes="Default model, works on CPU and GPU",
    ),
    TestedModel(
        name="qwen2-vl-7b",
        backend=Backend.HUGGINGFACE,
        model_id="Qwen/Qwen2-VL-7B-Instruct",
        requires_gpu=False,
        notes="Higher quality, more memory required",
    ),
    # OCR-specialized HuggingFace models (requires GPU)
    TestedModel(
        name="dots-ocr",
        backend=Backend.HUGGINGFACE,
        model_id="rednote-hilab/dots.ocr",
        requires_gpu=True,
        notes="OCR-specialized model, requires CUDA GPU",
    ),
    TestedModel(
        name="monkey-ocr",
        backend=Backend.HUGGINGFACE,
        model_id="echo840/Monkey-Chat",
        requires_gpu=True,
        notes="Monkey OCR model, requires CUDA GPU",
    ),
    TestedModel(
        name="deepseek-ocr",
        backend=Backend.HUGGINGFACE,
        model_id="deepseek-ai/deepseek-vl2-small",
        requires_gpu=True,
        notes="DeepSeek VL2 model, requires CUDA GPU",
    ),
    # Ollama models
    TestedModel(
        name="llama3.2-vision",
        backend=Backend.OLLAMA,
        model_id="llama3.2-vision",
        requires_gpu=False,
        notes="Ollama vision model, requires Ollama server running",
    ),
    # MLX models (Apple Silicon)
    TestedModel(
        name="qwen2-vl-2b-mlx",
        backend=Backend.MLX,
        model_id="mlx-community/Qwen2-VL-2B-Instruct-4bit",
        requires_gpu=False,
        notes="MLX backend for Apple Silicon, requires mlx_vlm server on port 8080",
    ),
    # vLLM models (experimental - image processing may not work correctly)
    # Note: vLLM vision support requires specific server configuration.
    # Testing shows images may not be properly tokenized with default vLLM setup.
    # If you need vLLM, ensure your server is configured for multimodal inputs.
    TestedModel(
        name="qwen2-vl-2b-vllm",
        backend=Backend.VLLM,
        model_id="Qwen/Qwen2-VL-2B-Instruct",
        requires_gpu=True,
        notes="EXPERIMENTAL: Requires vLLM server with proper vision model config",
    ),
]


def get_tested_model(name: str) -> Optional[TestedModel]:
    """Get a tested model by name."""
    for model in TESTED_MODELS:
        if model.name == name:
            return model
    return None


def list_tested_models() -> List[str]:
    """List all tested model names."""
    return [m.name for m in TESTED_MODELS]


@dataclass(frozen=True)
class GenerationParams:
    """Parameters for text generation."""
    max_new_tokens: int = 500
    temperature: float = 0.0


class LocalVLMClient:
    """
    A unified client for local VLMs across multiple backends.
    
    Model format: "backend:model_id" or just "model_id" (defaults to HuggingFace)
    
    Examples:
        - "Qwen/Qwen2-VL-2B-Instruct" -> HuggingFace
        - "hf:Qwen/Qwen2-VL-2B-Instruct" -> HuggingFace (explicit)
        - "ollama:llava" -> Ollama
        - "vllm:Qwen/Qwen2-VL-2B-Instruct" -> vLLM
        - "mlx:mlx-community/Qwen2-VL-2B-Instruct-4bit" -> MLX (Apple Silicon)
    """

    def __init__(
        self,
        device: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
        ollama_host: str = "http://localhost:11434",
        vllm_host: str = "http://localhost:8000",
        mlx_host: str = "http://localhost:8080",
    ):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        if torch_dtype is None:
            self.torch_dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        else:
            self.torch_dtype = torch_dtype

        self.ollama_host = ollama_host
        self.vllm_host = vllm_host
        self.mlx_host = mlx_host

        # Cache for loaded HuggingFace models
        self._hf_cache: Dict[str, Dict[str, Any]] = {}

    def _parse_model_string(self, model: str) -> tuple[Backend, str]:
        """Parse model string into backend and model_id."""
        if ":" in model and model.split(":")[0] in [b.value for b in Backend]:
            parts = model.split(":", 1)
            backend = Backend(parts[0])
            model_id = parts[1]
        else:
            # Default to HuggingFace
            backend = Backend.HUGGINGFACE
            model_id = model
        return backend, model_id

    def generate(
        self,
        *,
        model: str,
        image: Image.Image,
        user_prompt: str,
        system_prompt: str = "",
        params: Optional[GenerationParams] = None,
    ) -> str:
        """
        Generate text for a single image.
        
        Args:
            model: Model identifier (e.g., "Qwen/Qwen2-VL-2B-Instruct" or "ollama:llava")
            image: PIL Image to process
            user_prompt: The prompt/question for the model
            system_prompt: Optional system prompt
            params: Generation parameters
            
        Returns:
            Generated text response
        """
        if params is None:
            params = GenerationParams()

        backend, model_id = self._parse_model_string(model)

        if backend == Backend.HUGGINGFACE:
            return self._generate_huggingface(model_id, image, user_prompt, system_prompt, params)
        elif backend == Backend.OLLAMA:
            return self._generate_ollama(model_id, image, user_prompt, system_prompt, params)
        elif backend == Backend.VLLM:
            return self._generate_vllm(model_id, image, user_prompt, system_prompt, params)
        elif backend == Backend.MLX:
            return self._generate_mlx(model_id, image, user_prompt, system_prompt, params)
        else:
            raise ValueError(f"Unsupported backend: {backend}")

    # -------------------------------------------------------------------------
    # HuggingFace Transformers Backend
    # -------------------------------------------------------------------------

    def _load_huggingface(self, model_id: str) -> Dict[str, Any]:
        """Load a HuggingFace model and processor."""
        if model_id in self._hf_cache:
            return self._hf_cache[model_id]

        from transformers import AutoProcessor, AutoModelForImageTextToText, AutoModel, AutoModelForCausalLM
        import warnings

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*preprocessor.json.*deprecated.*")
            
            # For dots.ocr and similar Qwen2-VL based models, AutoProcessor.from_pretrained 
            # can be finicky. We try to load it normally first.
            try:
                processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
            except Exception as e:
                # If we get "multiple values for chat_template", it's a known bug in 
                # transformers factory when loading some custom processors.
                if "chat_template" in str(e):
                    self.logger.warning(f"Caught chat_template error, attempting workaround for {model_id}")
                    # Workaround: Load without passing chat_template if it's already in the config
                    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True, chat_template=None)
                else:
                    raise e
            
            # Workaround for 'Unrecognized video processor' error in some transformers versions
            # when video_preprocessor_config.json is missing (common in Qwen2-VL/dots-ocr)
            # This is only needed if the model hasn't been patched locally.
            if not hasattr(processor, "video_processor") or processor.video_processor is None:
                try:
                    # Accessing video_processor property might trigger the error
                    _ = getattr(processor, "video_processor", None)
                except Exception:
                    try:
                        processor.video_processor = None
                    except Exception:
                        pass

            # Choose the right model class. dots.ocr model card recommends AutoModelForCausalLM
            model_class = AutoModelForImageTextToText
            if "dots.ocr" in model_id.lower():
                model_class = AutoModelForCausalLM

            try:
                model = model_class.from_pretrained(
                    model_id,
                    trust_remote_code=True,
                    torch_dtype=self.torch_dtype,
                    device_map="auto" if self.device == "cuda" else None
                )
            except Exception as e:
                self.logger.warning(f"Failed to load with {model_class.__name__}, trying AutoModel: {e}")
                model = AutoModel.from_pretrained(
                    model_id,
                    trust_remote_code=True,
                    torch_dtype=self.torch_dtype,
                    device_map="auto" if self.device == "cuda" else None
                )

        if hasattr(model, "to"):
            model = model.to(self.device)
        if hasattr(model, "eval"):
            model.eval()

        cache_entry = {"processor": processor, "model": model}
        self._hf_cache[model_id] = cache_entry
        return cache_entry

    def _generate_huggingface(
        self,
        model_id: str,
        image: Image.Image,
        user_prompt: str,
        system_prompt: str,
        params: GenerationParams,
    ) -> str:
        """Generate using HuggingFace Transformers."""
        loaded = self._load_huggingface(model_id)
        processor = loaded["processor"]
        model = loaded["model"]

        # Build messages for chat template
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        # Check if the model is Qwen2-VL based (common for DOTS)
        is_qwen2_vl = "qwen2_vl" in str(getattr(model.config, "model_type", "")).lower()
        
        if is_qwen2_vl:
            # Qwen2-VL specifically handles image tokens in a certain way
            messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "image", 
                        "image": image,
                        # DOTS often works better with original resolution or high-res
                        "resized_height": image.height,
                        "resized_width": image.width,
                    },
                    {"type": "text", "text": user_prompt},
                ],
            })
        else:
            messages.append({
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": user_prompt},
                ],
            })

        # Apply chat template and process
        text_input = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        # Prepare inputs with special handling for potential Qwen2-VL processor issues
        try:
            inputs = processor(
                text=[text_input], 
                images=[image], 
                return_tensors="pt", 
                padding=True
            )
        except Exception as e:
            if "Unrecognized video processor" in str(e):
                # Fallback: manually prepare inputs if the processor is getting confused about video
                print("Detected 'Unrecognized video processor' error. Attempting fallback input preparation...")
                # Some models need images=None if they handle vision via different keys
                inputs = processor(
                    text=[text_input],
                    return_tensors="pt",
                    padding=True
                )
                # Manually add pixel values if they were missed
                if "pixel_values" not in inputs:
                    vision_inputs = processor(images=[image], return_tensors="pt")
                    inputs.update(vision_inputs)
            else:
                raise e

        # Move to device
        for k, v in inputs.items():
            if hasattr(v, "to"):
                inputs[k] = v.to(self.device)

        # Generate
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=params.max_new_tokens)

        # Trim input tokens from output
        input_len = inputs["input_ids"].shape[1]
        generated_ids_trimmed = generated_ids[:, input_len:]

        result = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        return result.strip()

    # -------------------------------------------------------------------------
    # Ollama Backend
    # -------------------------------------------------------------------------

    def _generate_ollama(
        self,
        model_id: str,
        image: Image.Image,
        user_prompt: str,
        system_prompt: str,
        params: GenerationParams,
    ) -> str:
        """Generate using Ollama API."""
        import requests

        # Convert image to base64
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        image_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        # Build request
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({
            "role": "user",
            "content": user_prompt,
            "images": [image_b64],
        })

        response = requests.post(
            f"{self.ollama_host}/api/chat",
            json={
                "model": model_id,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": params.temperature,
                    "num_predict": params.max_new_tokens,
                },
            },
            timeout=300,
        )
        response.raise_for_status()

        result = response.json()
        return result.get("message", {}).get("content", "").strip()

    # -------------------------------------------------------------------------
    # vLLM Backend
    # -------------------------------------------------------------------------

    def _generate_vllm(
        self,
        model_id: str,
        image: Image.Image,
        user_prompt: str,
        system_prompt: str,
        params: GenerationParams,
    ) -> str:
        """Generate using vLLM OpenAI-compatible API."""
        import requests

        # Convert image to base64
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        image_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        # Build messages in OpenAI format
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": user_prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
            ],
        })

        response = requests.post(
            f"{self.vllm_host}/v1/chat/completions",
            json={
                "model": model_id,
                "messages": messages,
                "max_tokens": params.max_new_tokens,
                "temperature": params.temperature,
            },
            timeout=300,
        )
        response.raise_for_status()

        result = response.json()
        return result["choices"][0]["message"]["content"].strip()

    # -------------------------------------------------------------------------
    # MLX Backend (Apple Silicon)
    # -------------------------------------------------------------------------

    def _generate_mlx(
        self,
        model_id: str,
        image: Image.Image,
        user_prompt: str,
        system_prompt: str,
        params: GenerationParams,
    ) -> str:
        """Generate using mlx_vlm server API (Apple Silicon optimized)."""
        import requests
        import tempfile
        import os

        # mlx_vlm server prefers file paths over base64
        # Save image to a temp file
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            image.save(f, format="PNG")
            temp_path = f.name

        try:
            # Build messages in mlx_vlm format
            content = [
                {"type": "text", "text": user_prompt},
                {"type": "input_image", "image_url": temp_path},
            ]

            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": content})

            response = requests.post(
                f"{self.mlx_host}/chat/completions",
                json={
                    "model": model_id,
                    "messages": messages,
                    "max_tokens": params.max_new_tokens,
                    "temperature": params.temperature if params.temperature > 0 else 0.1,
                },
                timeout=300,
            )
            response.raise_for_status()

            result = response.json()
            return result["choices"][0]["message"]["content"].strip()
        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.unlink(temp_path)
