#!/usr/bin/env python3
"""
Batch OCR Processing Script

Runs multiple VLM OCR models over all MMIF files in a directory.
Creates subdirectories for each model's output.

Usage:
    python batch_ocr.py <input_dir> <output_dir> [--models MODEL1 MODEL2 ...]
    
Example:
    python batch_ocr.py ./mmif_input ./ocr_output
    python batch_ocr.py ./mmif_input ./ocr_output --models dots-ocr monkey-ocr deepseek-ocr
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Models to run by default (GPU-required OCR models)
DEFAULT_MODELS = [
    "hf:rednote-hilab/dots.ocr",      # dots.ocr
    "hf:echo840/Monkey-Chat",          # monkey OCR
    "hf:prithivMLmods/DeepSeek-OCR-Latest-BF16.I64",  # DeepSeek-OCR latest
]

# Model name mapping for directory names
MODEL_DIR_NAMES = {
    "hf:rednote-hilab/dots.ocr": "dots-ocr",
    "hf:echo840/Monkey-Chat": "monkey-ocr",
    "hf:prithivMLmods/DeepSeek-OCR-Latest-BF16.I64": "deepseek-ocr",
    "hf:deepseek-ai/DeepSeek-OCR": "deepseek-ocr-official",
    "hf:deepseek-ai/deepseek-vl2-small": "deepseek-vl2-small",
    "hf:Qwen/Qwen2-VL-2B-Instruct": "qwen2-vl-2b",
    "hf:Qwen/Qwen2-VL-7B-Instruct": "qwen2-vl-7b",
    "mlx:mlx-community/Qwen2-VL-2B-Instruct-4bit": "qwen2-vl-mlx",
    "ollama:llama3.2-vision": "llama32-vision",
}


def get_model_dir_name(model: str) -> str:
    """Get a safe directory name for a model."""
    if model in MODEL_DIR_NAMES:
        return MODEL_DIR_NAMES[model]
    # Fall back to sanitized model name
    return model.replace(":", "_").replace("/", "_")


def find_mmif_files(input_dir: Path) -> List[Path]:
    """Find all MMIF JSON files in the input directory."""
    mmif_files = []
    for ext in ["*.json", "*.mmif"]:
        mmif_files.extend(input_dir.glob(ext))
    return sorted(mmif_files)


def process_file(
    input_file: Path,
    output_file: Path,
    model: str,
    tf_labels: Optional[List[str]] = None,
    all_targets: bool = False,
) -> dict:
    """
    Process a single MMIF file with the VLM OCR app.
    
    Returns a dict with timing and status info.
    """
    from app import VlmOcr
    from mmif import Mmif
    
    result = {
        "input": str(input_file),
        "output": str(output_file),
        "model": model,
        "status": "success",
        "error": None,
        "time_seconds": 0,
        "frames_processed": 0,
    }
    
    start_time = time.time()
    
    try:
        # Load input MMIF
        with open(input_file, 'r') as f:
            mmif_str = f.read()
        
        # Create app instance
        app = VlmOcr()
        
        # Build parameters
        params = {
            "hfModel": model,
            "allTargets": all_targets,
        }
        if tf_labels:
            params["tfLabel"] = tf_labels
        
        # Run annotation
        output_mmif = app.annotate(mmif_str, **params)
        
        # Count processed frames (TextDocuments created)
        mmif_obj = Mmif(output_mmif)
        frames_processed = 0
        for view in mmif_obj.views:
            for ann in view.annotations:
                if "TextDocument" in str(ann.at_type):
                    frames_processed += 1
        
        result["frames_processed"] = frames_processed
        
        # Write output
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            f.write(output_mmif)
            
    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
        logger.error(f"Error processing {input_file} with {model}: {e}")
    
    result["time_seconds"] = time.time() - start_time
    return result


def run_batch(
    input_dir: Path,
    output_dir: Path,
    models: List[str],
    tf_labels: Optional[List[str]] = None,
    all_targets: bool = False,
) -> dict:
    """
    Run batch processing for all models on all MMIF files.
    
    Returns a summary dict with all results.
    """
    mmif_files = find_mmif_files(input_dir)
    
    if not mmif_files:
        logger.error(f"No MMIF files found in {input_dir}")
        return {"error": "No MMIF files found"}
    
    logger.info(f"Found {len(mmif_files)} MMIF files to process")
    logger.info(f"Models to run: {models}")
    
    summary = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "models": models,
        "total_files": len(mmif_files),
        "results": {},
    }
    
    for model in models:
        model_dir_name = get_model_dir_name(model)
        model_output_dir = output_dir / model_dir_name
        model_output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing with model: {model}")
        logger.info(f"Output directory: {model_output_dir}")
        logger.info(f"{'='*60}")
        
        model_results = []
        model_start = time.time()
        
        for i, mmif_file in enumerate(mmif_files, 1):
            output_file = model_output_dir / mmif_file.name
            
            logger.info(f"[{i}/{len(mmif_files)}] {mmif_file.name}")
            
            result = process_file(
                input_file=mmif_file,
                output_file=output_file,
                model=model,
                tf_labels=tf_labels,
                all_targets=all_targets,
            )
            
            model_results.append(result)
            
            if result["status"] == "success":
                logger.info(f"  ✓ {result['frames_processed']} frames in {result['time_seconds']:.1f}s")
            else:
                logger.error(f"  ✗ Error: {result['error']}")
        
        model_total_time = time.time() - model_start
        successful = sum(1 for r in model_results if r["status"] == "success")
        total_frames = sum(r["frames_processed"] for r in model_results)
        
        summary["results"][model] = {
            "output_dir": str(model_output_dir),
            "files_processed": len(model_results),
            "files_successful": successful,
            "total_frames": total_frames,
            "total_time_seconds": model_total_time,
            "individual_results": model_results,
        }
        
        logger.info(f"\nModel {model} complete:")
        logger.info(f"  Files: {successful}/{len(model_results)} successful")
        logger.info(f"  Total frames: {total_frames}")
        logger.info(f"  Total time: {model_total_time:.1f}s")
    
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Run multiple VLM OCR models over MMIF files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run default OCR models (dots, monkey, deepseek)
    python batch_ocr.py ./mmif_input ./ocr_output
    
    # Run specific models
    python batch_ocr.py ./mmif_input ./ocr_output --models hf:Qwen/Qwen2-VL-2B-Instruct
    
    # Filter by TimeFrame label
    python batch_ocr.py ./mmif_input ./ocr_output --tf-labels Slate Chyron
    
    # Process all targets in TimeFrames
    python batch_ocr.py ./mmif_input ./ocr_output --all-targets
        """
    )
    
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing input MMIF files"
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Output directory (subdirs created per model)"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help=f"Models to run. Default: {DEFAULT_MODELS}"
    )
    parser.add_argument(
        "--tf-labels",
        nargs="+",
        default=None,
        help="TimeFrame labels to process (default: all)"
    )
    parser.add_argument(
        "--all-targets",
        action="store_true",
        help="Process all targets in TimeFrames instead of just representative"
    )
    parser.add_argument(
        "--summary-file",
        type=Path,
        default=None,
        help="Write JSON summary to this file"
    )
    
    args = parser.parse_args()
    
    # Validate input directory
    if not args.input_dir.exists():
        logger.error(f"Input directory does not exist: {args.input_dir}")
        sys.exit(1)
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run batch processing
    summary = run_batch(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        models=args.models,
        tf_labels=args.tf_labels,
        all_targets=args.all_targets,
    )
    
    # Write summary
    summary_file = args.summary_file or (args.output_dir / "batch_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"\nSummary written to: {summary_file}")
    
    # Print final summary
    print("\n" + "="*60)
    print("BATCH PROCESSING COMPLETE")
    print("="*60)
    for model, results in summary.get("results", {}).items():
        print(f"\n{model}:")
        print(f"  Output: {results['output_dir']}")
        print(f"  Success: {results['files_successful']}/{results['files_processed']} files")
        print(f"  Frames: {results['total_frames']}")
        print(f"  Time: {results['total_time_seconds']:.1f}s")


if __name__ == "__main__":
    main()

