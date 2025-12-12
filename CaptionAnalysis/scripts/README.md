# Caption Analysis Scripts

## correlated_comments.py

This script correlates YouTube viewer comments with video frame captions to identify comments that are strongly related to specific visual content.

### Prerequisites

*   Python 3.x
*   Use llava conda environment
*   Dependencies: `python-docx` (Install via `pip install python-docx`)
*   Configuration: Ensure `config.py` and `utils/` are available in the parent directory.

### Usage

```bash
python correlate_comments.py  --frames_path <frames_input> \
    --comments_docx <comments_docx> \
    --results_root <output_directory> \
    --integrated_captions_root <captions_directory> \
    [--model <model_name>] \
    [--min_score <score>] \
    [--top_k <count>]
```

### Arguments

*   `frames_input`: 
    *   Path to a root directory containing video subfolders (e.g., `../data/raw_frames_root`).
    *   OR path to a single `raw_frames` folder.
    *   The script uses this to discover which videos to process.
*   `comments_docx`: Path to the DOCX file containing video URLs and comments.
*   `--results_root`: Directory where output files will be saved. The script mirrors the input directory structure here.
*   `--integrated_captions_root`: Root directory containing the integrated caption JSON files (e.g., `../data/integrated_caption`).
    *   The script searches for files named `<video_id>_captions_integrated.json`.

### Optional Arguments

*   `--model`: Name of the LLM model to use (default: configured OLLAMA model).
*   `--min_score`: Minimum correlation score (0-100) to include a comment (default: 60).
*   `--top_k`: Number of top correlated comments to show per frame (default: 5).
*   `--rebuild`: Regenerate the DOCX report from existing `.jsonl` results without re-running the LLM.

### Example

```bash
python correlate_comments.py \
    -- frames_path../data/integrated_caption/frames_conventional_captions_integrated \
    --comments_docx comments.docx \
    --results_root ../results \
    --integrated_captions_root ../data/integrated_caption \
    --min_score 70
```
