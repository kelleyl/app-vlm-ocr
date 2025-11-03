"""
The purpose of this file is to define the metadata of the app with minimal imports.

DO NOT CHANGE the name of the file
"""
from mmif import DocumentTypes, AnnotationTypes

from clams.app import ClamsApp
from clams.appmetadata import AppMetadata


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
        name="CLAMS VLM OCR",
        description=(
            "CLAMS app that uses a user-specified Hugging Face vision-language or OCR model "
            "to transcribe text from representative frames in a video segment. Outputs a single "
            "TextDocument per processed frame and Alignment annotations linking the source TimePoint/TimeFrame."
        ),
        app_license="Apache 2.0",
        identifier="vlm-ocr",
        url="https://github.com/clamsproject/app-vlm-ocr",
        analyzer_version="various",  # depends on the selected HF model
        analyzer_license="various",
    )

    # Inputs: a video and its TimeFrame(s) with optional representatives
    metadata.add_input(DocumentTypes.VideoDocument)
    in_tf = metadata.add_input(AnnotationTypes.TimeFrame, representatives='?', label='*')
    in_tf.add_description(
        'The labeled TimeFrame annotation that represents the video segment to be processed. When '
        '`representatives` property is present, the app will process still frames referred to by the '
        '`representatives` property. Otherwise, the app will process the middle frame of the video segment. '
        'Generic TimeFrames with no `label` property will not be processed.'
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
        name='tfLabel',
        default=[],
        type='string',
        multivalued=True,
        description='Labels of TimeFrame annotations to process. Default ([]): process all labeled TimeFrames.'
    )
    metadata.add_parameter(
        name='hfModel',
        default='microsoft/trocr-base-printed',
        type='string',
        multivalued=False,
        description=(
            'Hugging Face model id to use with an image-to-text pipeline. '
            'Examples: microsoft/trocr-base-printed, microsoft/trocr-large-printed, nougat, etc.'
        )
    )

    metadata.add_parameter(
        name='frameInterval',
        default=290,
        type='integer',
        multivalued=False,
        description=(
            'When no TimePoint/TimeFrame annotations are present in the MMIF, sample every N frames '
            'from the input video. Default: 290.'
        )
    )

    metadata.add_parameter(
        name='prompt',
        default='',
        type='string',
        multivalued=False,
        description='Prompt text to guide OCR for models that support prompts (e.g., DeepSeek-OCR). Defaults to "Describe the image" for prompt-capable models if not provided. If a non-empty prompt is provided with a non-prompt model, the app will error.'
    )

    return metadata


# DO NOT CHANGE the main block
if __name__ == '__main__':
    import sys
    metadata = appmetadata()
    for param in ClamsApp.universal_parameters:
        metadata.add_parameter(**param)
    sys.stdout.write(metadata.jsonify(pretty=True))
