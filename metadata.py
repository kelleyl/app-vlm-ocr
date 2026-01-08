"""
The purpose of this file is to define the metadata of the app with minimal imports.

DO NOT CHANGE the name of the file
"""
from mmif import DocumentTypes, AnnotationTypes

from clams.app import ClamsApp
from clams.appmetadata import AppMetadata


# Import tested models from llm_utils
from llm_utils import TESTED_MODELS as LLM_TESTED_MODELS, Backend

# Default model - MLX backend for Apple Silicon (local development)
# For GPU servers, use: hf:prithivMLmods/DeepSeek-OCR-Latest-BF16.I64 (deepseek-ocr)
#                    or: hf:deepseek-ai/DeepSeek-OCR (deepseek-ocr-official)
#                    or: hf:echo840/Monkey-Chat (monkey-ocr)
#                    or: hf:deepseek-ai/deepseek-vl2-small (deepseek-vl2-small)
#                    or: hf:rednote-hilab/dots.ocr (dots-ocr)
DEFAULT_MODEL = "mlx:mlx-community/Qwen2-VL-2B-Instruct-4bit"

# OCR-specialized models (for batch processing on GPU servers)
OCR_MODELS = [
    "hf:rednote-hilab/dots.ocr",       # dots-ocr
    "hf:echo840/Monkey-Chat",           # monkey-ocr  
    "hf:prithivMLmods/DeepSeek-OCR-Latest-BF16.I64",  # deepseek-ocr (latest)
    "hf:deepseek-ai/DeepSeek-OCR",      # deepseek-ocr-official
    "hf:deepseek-ai/deepseek-vl2-small",  # deepseek-vl2-small
]

# Tested model IDs for metadata display (with backend prefix)
TESTED_MODELS = [f"{m.backend.value}:{m.model_id}" for m in LLM_TESTED_MODELS]


# DO NOT CHANGE the function name
def appmetadata() -> AppMetadata:
    """
    Function to set app-metadata values and return it as an ``AppMetadata`` obj.
    Read these documentations before changing the code below
    - https://sdk.clams.ai/appmetadata.html metadata specification.
    - https://sdk.clams.ai/autodoc/clams.appmetadata.html python API

    :return: AppMetadata object holding all necessary information.
    """
    # first set up some basic information
    metadata = AppMetadata(
        name="VLM OCR",
        description=(
            "Applies vision-language models to extract text from video frames. "
            "Processes TimeFrame annotations to perform OCR on frames containing text. "
            "Supports multiple backends: MLX (Apple Silicon), HuggingFace, Ollama, and vLLM."
        ),
        app_license="Apache 2.0",
        identifier="vlm-ocr",
        url="https://github.com/clamsproject/app-vlm-ocr",
        analyzer_version="various",  # depends on the selected model
        analyzer_license="various",
    )

    # Inputs: a video and its TimeFrame(s) with optional representatives
    metadata.add_input(DocumentTypes.VideoDocument)
    in_tf = metadata.add_input(AnnotationTypes.TimeFrame)
    in_tf.add_description(
        'The labeled TimeFrame annotation that represents the video segment to be processed. When '
        '`representatives` property is present, the app will process still frames referred to by the '
        '`representatives` property. Otherwise, the app will process the middle frame of the video segment.'
    )

    # Outputs: only TextDocument and Alignment
    out_td = metadata.add_output(DocumentTypes.TextDocument, **{'@lang': 'en'})
    out_td.add_description(
        'Fully serialized text content recognized from the input images.'
    )
    out_ali = metadata.add_output(AnnotationTypes.Alignment)
    out_ali.add_description(
        'Alignments between TimePoint/TimeFrame annotations and the produced TextDocument annotations.'
    )

    # Parameters
    metadata.add_parameter(
        name='hfModel',
        default=DEFAULT_MODEL,
        type='string',
        multivalued=False,
        description=(
            f'Model ID with optional backend prefix. Format: "backend:model_id" or just "model_id". '
            f'Backends: mlx (Apple Silicon), hf (HuggingFace), ollama, vllm. '
            f'Tested models: {", ".join(TESTED_MODELS)}. '
            f'Default: {DEFAULT_MODEL}'
        )
    )
    metadata.add_parameter(
        name='appUri',
        default='',
        type='string',
        multivalued=False,
        description=(
            'URI of the app that produced TimeFrame annotations to use. '
            'If empty, uses TimeFrames from any app. '
            'Example: "http://apps.clams.ai/swt-detection/"'
        )
    )
    metadata.add_parameter(
        name='tfLabel',
        default=[],
        type='string',
        multivalued=True,
        description=(
            'Labels of TimeFrame annotations to process (case-insensitive). '
            'Default ([]): process all labeled TimeFrames.'
        )
    )
    metadata.add_parameter(
        name='allTargets',
        type='boolean',
        default=False,
        description=(
            'If true, process ALL TimePoint targets within each TimeFrame instead of just the '
            'representative frame. This can significantly increase processing time but provides '
            'OCR for every detected frame.'
        )
    )
    metadata.add_parameter(
        name='config',
        default='',
        type='string',
        multivalued=False,
        description=(
            'Path to YAML config file (relative to app directory). '
            'Config files can specify prompts and system prompts. '
            'See config/default.yaml for example. Leave empty to use parameter defaults.'
        )
    )

    # Prompt parameters
    metadata.add_parameter(
        name='defaultPrompt',
        type='string',
        default='Transcribe all text visible in this image.',
        description=(
            'Default user prompt for OCR. Config file can override this via default_prompt. '
            'If set to `-`, any label not mapped in promptMap will be skipped.'
        )
    )
    metadata.add_parameter(
        name='promptMap',
        type='map',
        default=[],
        description=(
            'Mapping of TimeFrame labels to user prompts. Format: "LABEL:PROMPT". '
            'To skip a label, set PROMPT to `-`. Config file can override via custom_prompts.'
        )
    )
    metadata.add_parameter(
        name='defaultSystemPrompt',
        type='string',
        default='',
        description=(
            'Default system prompt passed as a system message (role="system"). '
            'Config file can override via default_system_prompt.'
        )
    )
    metadata.add_parameter(
        name='systemPromptMap',
        type='map',
        default=[],
        description=(
            'Mapping of TimeFrame labels to system prompts. Format: "LABEL:SYSTEM_PROMPT". '
            'Config file can override via custom_system_prompts.'
        )
    )

    return metadata


# DO NOT CHANGE the main block
if __name__ == '__main__':
    import sys
    metadata = appmetadata()
    for param in ClamsApp.universal_parameters:
        metadata.add_parameter(**param)
    sys.stdout.write(metadata.jsonify(pretty=True))
