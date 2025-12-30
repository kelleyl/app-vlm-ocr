# VLM OCR

A CLAMS app that performs **OCR on video frames** using vision-language models (VLMs). It processes `TimeFrame` annotations (e.g., from `swt-detection`) and extracts text from representative frames or all target frames.

## Description

This app applies configurable vision-language models to extract text from video frames. It:

- Processes `TimeFrame` annotations from upstream apps (e.g., `swt-detection`)
- Extracts representative frames or all target frames from each TimeFrame
- Runs OCR using local VLMs via multiple backends
- Outputs `TextDocument` annotations aligned to the source TimeFrames/TimePoints

## Supported Backends

The app supports multiple VLM backends via a LiteLLM-style interface:

| Backend | Prefix | Example | Server Port | Status |
|---------|--------|---------|-------------|--------|
| **MLX** (Apple Silicon) | `mlx:` | `mlx:mlx-community/Qwen2-VL-2B-Instruct-4bit` | 8080 | ✅ Recommended for Mac |
| HuggingFace Transformers | `hf:` (default) | `Qwen/Qwen2-VL-2B-Instruct` | N/A (local) | ✅ Verified |
| Ollama | `ollama:` | `ollama:llama3.2-vision` | 11434 | ✅ Verified |
| vLLM | `vllm:` | `vllm:Qwen/Qwen2-VL-2B-Instruct` | 8000 | ⚠️ Experimental |

### Tested Models

| Model | Backend | GPU Required | Status | Notes |
|-------|---------|--------------|--------|-------|
| `mlx-community/Qwen2-VL-2B-Instruct-4bit` | MLX | No | ✅ Verified | **Recommended for Apple Silicon** |
| `Qwen/Qwen2-VL-2B-Instruct` | HuggingFace | No | ✅ Verified | Default model, works on CPU/GPU |
| `Qwen/Qwen2-VL-7B-Instruct` | HuggingFace | No | ✅ Verified | Higher quality, more memory |
| `llama3.2-vision` | Ollama | No | ✅ Verified | Requires Ollama server |
| `Qwen/Qwen2-VL-2B-Instruct` | vLLM | Yes | ⚠️ Experimental | May have image processing issueson |

## Performance Benchmarks

Tested on Apple Silicon (M-series Mac) with MLX backend:

| Scenario | Frames | Time | Rate |
|----------|--------|------|------|
| All targets (Bars, Slate, Chyron, Neg) | 127 | 4m 29s | **0.48 fps** |
| Slate + Chyron only | 31 | 1m 21s | **0.38 fps** |

- **~2 seconds per frame** average processing time
- OCR quality is excellent on text-containing frames (Slate, Chyron)
- Use `--tfLabel` to filter frame types and improve efficiency

## Installation

### Requirements

- Python 3.11+
- ~8GB disk space for model weights
- For MLX backend: Apple Silicon Mac with mlx-vlm installed

### Setup

```bash
# Create virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Starting Backend Servers

**MLX (Apple Silicon - Recommended for Mac)**:
```bash
# In a separate terminal/environment with mlx-vlm installed
pip install mlx-vlm
python -m mlx_vlm.server  # Runs on port 8080
```

**Ollama**:
```bash
ollama serve  # Runs on port 11434
ollama pull llama3.2-vision
```

## Usage

### CLI Usage

```bash
# Use MLX backend (Apple Silicon - recommended)
python cli.py --hfModel "mlx:mlx-community/Qwen2-VL-2B-Instruct-4bit" input.mmif output.mmif

# Process ALL targets within TimeFrames (not just representative frames)
python cli.py --hfModel "mlx:mlx-community/Qwen2-VL-2B-Instruct-4bit" \
  --allTargets true input.mmif output.mmif

# Filter by TimeFrame labels (recommended for efficiency)
python cli.py --hfModel "mlx:mlx-community/Qwen2-VL-2B-Instruct-4bit" \
  --tfLabel Slate --tfLabel Chyron \
  --allTargets true input.mmif output.mmif

# Use HuggingFace backend (default, no server needed)
python cli.py --hfModel "Qwen/Qwen2-VL-2B-Instruct" input.mmif output.mmif

# Use Ollama backend
python cli.py --hfModel "ollama:llama3.2-vision" input.mmif output.mmif

# Use a custom config file for prompts
python cli.py --config "config/default.yaml" input.mmif output.mmif
```

### HTTP Server

```bash
# Development server
python app.py --port 5000

# Production server
python app.py --port 5000 --production
```

Then POST MMIF to `http://localhost:5000/`:

```bash
curl -X POST -H "Content-Type: application/json" \
  -d @input.mmif \
  "http://localhost:5000/?hfModel=mlx:mlx-community/Qwen2-VL-2B-Instruct-4bit&allTargets=true"
```

### Example Output

Given a video with a slate frame, the app produces:

```
STATELINE
Number: 703
Air Dates: 10/15/86 19:30
Repeat: 10/19/86 12:30
Producer: Hatch
Director: Sheehan
TH 2389
```

## Configuration

### Config Files

Config files (YAML) specify prompts for different TimeFrame labels:

```yaml
# config/default.yaml
default_prompt: |
  Transcribe all text visible in this image.

custom_prompts:
  Slate: |
    This is a slate frame. Please transcribe all visible text including:
    - Title of the program
    - Date of recording
    - Any identifiers or codes
    
  Chyron: |
    This is a chyron (lower third). Transcribe the text exactly as shown.
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `hfModel` | `Qwen/Qwen2-VL-2B-Instruct` | Model ID with optional backend prefix (`mlx:`, `ollama:`, `vllm:`) |
| `allTargets` | `false` | Process ALL TimePoint targets within each TimeFrame |
| `tfLabel` | `[]` (all) | TimeFrame labels to process (e.g., `Slate`, `Chyron`) |
| `config` | `config/default.yaml` | Path to YAML config file |
| `defaultPrompt` | (see metadata) | Default OCR prompt |
| `promptMap` | `[]` | Label-specific prompts (`LABEL:PROMPT`) |
| `defaultSystemPrompt` | `""` | System prompt for the model |
| `systemPromptMap` | `[]` | Label-specific system prompts |

## Input/Output

### Input Requirements

- `VideoDocument` with a valid video file
- `TimeFrame` annotations (e.g., from `swt-detection`) with:
  - `start` and `end` times
  - `label` property (e.g., "Slate", "Chyron", "Bars")
  - `targets` property pointing to TimePoint annotations

### Output

- `TextDocument` for each processed frame containing the OCR result
- `Alignment` linking each TextDocument to its source TimeFrame or TimePoint

## Docker/Podman

```bash
# Build container
podman build -t app-vlm-ocr -f Containerfile .

# Run container
podman run -p 5000:5000 app-vlm-ocr

# With GPU support
podman run --gpus all -p 5000:5000 app-vlm-ocr
```

## Development

### Project Structure

```
app-vlm-ocr/
├── app.py              # Main CLAMS app
├── metadata.py         # App metadata and parameters
├── cli.py              # CLI interface
├── llm_utils.py        # Multi-backend VLM client (MLX, HuggingFace, Ollama, vLLM)
├── config/
│   └── default.yaml    # Default prompts config
├── requirements.txt
└── Containerfile
```

### Adding New Models

To add support for a new model:

1. Add to `TESTED_MODELS` list in `llm_utils.py`
2. Specify the backend and model_id
3. Test with sample MMIF files

### Backend Architecture

The `llm_utils.py` module provides a unified `LocalVLMClient` that routes requests to different backends:

```python
from llm_utils import LocalVLMClient

client = LocalVLMClient()

# MLX backend (Apple Silicon)
result = client.generate(
    model="mlx:mlx-community/Qwen2-VL-2B-Instruct-4bit",
    image=pil_image,
    user_prompt="Transcribe text in this image.",
)

# HuggingFace backend
result = client.generate(
    model="Qwen/Qwen2-VL-2B-Instruct",  # hf: prefix optional
    image=pil_image,
    user_prompt="Transcribe text in this image.",
)

# Ollama backend
result = client.generate(
    model="ollama:llama3.2-vision",
    image=pil_image,
    user_prompt="Transcribe text in this image.",
)
```

## License

Apache 2.0
