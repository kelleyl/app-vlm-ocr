# VLM OCR

A CLAMS app that performs **OCR on video frames** using vision-language models (VLMs). It processes `TimeFrame` annotations (e.g., from `swt-detection`) and extracts text from representative frames or all target frames.

## Description

This app applies configurable vision-language models to extract text from video frames. It:

- Processes `TimeFrame` annotations from upstream apps
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
| `mlx-community/Qwen2-VL-2B-Instruct-4bit` | MLX | No | ✅ Verified | **Default** - Recommended for Apple Silicon |
| `Qwen/Qwen2-VL-2B-Instruct` | HuggingFace | No | ✅ Verified | Works on CPU/GPU |
| `Qwen/Qwen2-VL-7B-Instruct` | HuggingFace | No | ✅ Verified | Higher quality, more memory |
| `llama3.2-vision` | Ollama | No | ✅ Verified | Requires Ollama server |
| `NielsRogge/dots.llm.7b` | HuggingFace | Yes | 🔬 Untested | OCR-specialized |
| `echo840/Monkey-Chat` | HuggingFace | Yes | 🔬 Untested | Monkey OCR |
| `deepseek-ai/deepseek-vl2-small` | HuggingFace | Yes | 🔬 Untested | DeepSeek VL2 |

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
# Default: MLX backend (Apple Silicon)
python cli.py input.mmif output.mmif

# Use HuggingFace backend
python cli.py --hfModel "hf:Qwen/Qwen2-VL-2B-Instruct" input.mmif output.mmif

# Filter by TimeFrame labels
python cli.py --tfLabel Slate --tfLabel Chyron input.mmif output.mmif

# Process ALL targets within TimeFrames (not just representative)
python cli.py --allTargets true input.mmif output.mmif

# Use custom prompts via config file
python cli.py --config config/default.yaml input.mmif output.mmif

# Use Ollama backend
python cli.py --hfModel "ollama:llama3.2-vision" input.mmif output.mmif
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
  "http://localhost:5000/?allTargets=true&tfLabel=Slate"
```

### Batch Processing

For processing multiple files with multiple models:

```bash
# Run default OCR models over all MMIF files
python batch_ocr.py ./mmif_input ./ocr_output

# Run specific models
python batch_ocr.py ./input ./output --models "hf:Qwen/Qwen2-VL-2B-Instruct"

# Filter by labels
python batch_ocr.py ./input ./output --tf-labels Slate Chyron
```

Output structure:
```
output/
├── dots-ocr/
│   └── *.json
├── monkey-ocr/
│   └── *.json
├── deepseek-ocr/
│   └── *.json
└── batch_summary.json  # Timing and results
```

---

## Prompt System

The app uses a flexible prompt system with three layers of configuration:

### 1. Parameter Defaults (metadata.py)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `defaultPrompt` | `"Transcribe all text visible in this image."` | User prompt sent to model |
| `defaultSystemPrompt` | `""` (empty) | System prompt (role="system") |
| `promptMap` | `[]` | Label-specific user prompts |
| `systemPromptMap` | `[]` | Label-specific system prompts |

### 2. Config File Override (YAML)

Config files can override defaults:

```yaml
# config/default.yaml
default_prompt: |
  Transcribe all text visible in this image.

default_system_prompt: |
  You are an expert OCR system.

custom_prompts:
  Slate: |
    This is a slate frame. Extract title, date, and identifiers.
  Chyron: |
    This is a chyron. Transcribe the text exactly.

custom_system_prompts:
  Slate: |
    Focus on structured information extraction.
```

### 3. CLI/API Parameter Override

Parameters passed via CLI or API override config file settings:

```bash
python cli.py \
  --defaultPrompt "Extract all visible text" \
  --promptMap "Slate:Extract slate info" \
  --promptMap "Chyron:Transcribe chyron text" \
  input.mmif output.mmif
```

### Prompt Flow

```
┌─────────────────┐
│  metadata.py    │  ← Default values
│  (defaults)     │
└────────┬────────┘
         ↓
┌─────────────────┐
│  config.yaml    │  ← Config file overrides (if --config specified)
│  (optional)     │
└────────┬────────┘
         ↓
┌─────────────────┐
│  CLI/API params │  ← Runtime parameters override all
│  (runtime)      │
└────────┬────────┘
         ↓
┌─────────────────┐
│  get_prompts()  │  ← Resolves (system_prompt, user_prompt) for label
└────────┬────────┘
         ↓
┌─────────────────┐
│  llm_utils.py   │  ← Sends to VLM backend
│  generate()     │
└─────────────────┘
```

### Skipping Labels

Set prompt to `-` to skip processing for a label:

```bash
# Skip "Bars" label, process others
python cli.py --promptMap "Bars:-" input.mmif output.mmif
```

---

## MMIF Output Structure

The app produces two annotation types:

### TextDocument

Contains the OCR result text:

```json
{
  "@type": "http://mmif.clams.ai/vocabulary/TextDocument/v1",
  "properties": {
    "id": "td1",
    "text": {
      "@value": "STATELINE\nNumber: 703\nAir Dates: 10/15/86"
    },
    "document": "m1",
    "origin": "v2:tf5",
    "provenance": "derived"
  }
}
```

| Property | Description |
|----------|-------------|
| `text.@value` | The extracted OCR text |
| `document` | Reference to source VideoDocument |
| `origin` | Reference to source TimeFrame |
| `provenance` | Always `"derived"` |

### Alignment

Links source frame to OCR result:

```json
{
  "@type": "http://mmif.clams.ai/vocabulary/Alignment/v1",
  "properties": {
    "id": "a1",
    "source": "v2:tf5",
    "target": "v3:td1"
  }
}
```

| Property | Description |
|----------|-------------|
| `source` | TimeFrame or TimePoint ID that was processed |
| `target` | TextDocument ID containing OCR result |

### allTargets Mode

When `allTargets=true`, each TimePoint within a TimeFrame gets its own TextDocument and Alignment:

```
TimeFrame (tf5)
├── TimePoint (tp10) → TextDocument (td1) + Alignment (a1)
├── TimePoint (tp11) → TextDocument (td2) + Alignment (a2)
└── TimePoint (tp12) → TextDocument (td3) + Alignment (a3)
```

When `allTargets=false` (default), only the representative frame is processed:

```
TimeFrame (tf5) → TextDocument (td1) + Alignment (a1)
```

---

## Parameters Reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `hfModel` | string | `mlx:mlx-community/Qwen2-VL-2B-Instruct-4bit` | Model ID with backend prefix |
| `tfLabel` | string[] | `[]` | TimeFrame labels to process (empty = all) |
| `allTargets` | boolean | `false` | Process all targets vs representative only |
| `config` | string | `""` | Path to YAML config file |
| `defaultPrompt` | string | `"Transcribe all text..."` | Default OCR prompt |
| `promptMap` | map | `[]` | Label → prompt mappings (`LABEL:PROMPT`) |
| `defaultSystemPrompt` | string | `""` | Default system prompt |
| `systemPromptMap` | map | `[]` | Label → system prompt mappings |

---

## Input/Output Requirements

### Input

- `VideoDocument` with a valid video file path
- `TimeFrame` annotations with:
  - `start` and `end` times
  - `label` property (e.g., "Slate", "Chyron")
  - Optional `targets` property pointing to TimePoint annotations

### Output

- `TextDocument` for each processed frame
- `Alignment` linking TextDocument to source

---

## Performance

Tested on Apple Silicon (M-series Mac) with MLX backend:

| Scenario | Frames | Time | Rate |
|----------|--------|------|------|
| All targets (127 frames) | 127 | 4m 29s | 0.48 fps |
| Slate + Chyron only (31 frames) | 31 | 1m 21s | 0.38 fps |

**Tips for efficiency:**
- Use `--tfLabel` to filter frame types
- Use MLX backend on Apple Silicon
- Consider `allTargets=false` for faster processing

---

## Docker/Podman

```bash
# Build container
podman build -t app-vlm-ocr -f Containerfile .

# Run container
podman run -p 5000:5000 app-vlm-ocr

# With GPU support
podman run --gpus all -p 5000:5000 app-vlm-ocr
```

---

## Project Structure

```
app-vlm-ocr/
├── app.py              # Main CLAMS app
├── metadata.py         # App metadata and parameter definitions
├── cli.py              # CLI interface
├── llm_utils.py        # Multi-backend VLM client
├── batch_ocr.py        # Batch processing script
├── config/
│   └── default.yaml    # Example prompts config
├── requirements.txt
├── Containerfile
├── LICENSE
└── README.md
```

---

## Development

### Adding New Models

1. Add to `TESTED_MODELS` list in `llm_utils.py`:

```python
TestedModel(
    name="my-model",
    backend=Backend.HUGGINGFACE,
    model_id="organization/model-name",
    requires_gpu=True,
    notes="Description",
),
```

2. Test with sample MMIF files

### Backend Architecture

The `llm_utils.py` module provides a unified `LocalVLMClient`:

```python
from llm_utils import LocalVLMClient, GenerationParams

client = LocalVLMClient()

# Generate with any backend
result = client.generate(
    model="mlx:mlx-community/Qwen2-VL-2B-Instruct-4bit",
    image=pil_image,
    user_prompt="Transcribe text in this image.",
    system_prompt="You are an OCR expert.",
    params=GenerationParams(max_new_tokens=500, temperature=0.0),
)
```

The client automatically routes to the correct backend based on the model prefix.

---

## License

Apache 2.0
