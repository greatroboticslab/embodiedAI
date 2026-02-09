# Fusion Analysis Scripts

This directory contains the pipeline for **Fusion Analysis**, which combines Transcript (Audio) and Caption (Visual) data to analyze viewer engagement.

The pipeline consists of two main steps:
1.  **Segmentation (`generate_sections.py`)**: Aligns multimodal data and uses an LLM to segment the video into "Engagement Sections".
2.  **Correlation (`correlate_sections.py`)**: Matches user comments to these sections to identify engagement drivers.

## Prerequisites

*   Python 3.8+
*   Dependencies: `sentence-transformers` (optional for embeddings), `python-docx` (required for DOCX reports), `numpy`.
*   Local `config.py` and `utils/` folder must be present in this directory (they should have been copied from `TranscriptAnalysis/scripts`).

## Workflow

### Step 1: Generate Sections

Run `generate_sections.py` to create the engagement sections. This script fuses the transcript and caption data, handles frame rate synchronization, and produces both machine-readable JSONs and human-readable DOCX reports.

**Usage:**

```bash
python generate_sections.py \
    --transcripts_root ../../TranscriptAnalysis/data/transcripts_conventional \
    --captions_root ../../CaptionAnalysis/data/integrated_caption/frames_conventional_captions_integrated \
    --frames_root ../../VideoAnalysis/data/frames/frames_conventional \
    --metrics_csv ../../CaptionAnalysis/output/Conventional_metrics.csv \
    --output_dir ../data/sections_conventional \
    --model llama3
```

**Arguments:**
*   `--transcripts_root`: Directory containing `_timestamped.txt` or `.json` transcript files.
*   `--captions_root`: Root directory for integrated captions (expects `<VideoID>/<VideoID>_captions_integrated.json` structure).
*   `--frames_root`: Root directory for video frames (used to insert images into the DOCX report).
*   `--metrics_csv`: One or more CSV paths (e.g., `Embodied_metrics.csv`) containing `frames_count` and `duration` for FPS calculation.
*   `--output_dir`: Base output folder. Sections will be saved in `output_dir/sections`.
*   `--model`: Name of the LLM model to use (default: `llama3`).

**Outputs:**
*   `data/sections/<VideoID>_sections.json`: JSON data of sections (Inputs for Step 2).
*   `data/sections/<VideoID>_sections.docx`: Report with timestamps, summaries, transcripts, and 5 representative frames per section.

---

### Step 2: Correlate Comments

Run `correlate_sections.py` to match user comments to the generated sections.

**Usage:**

```bash
python correlate_sections.py \
    --sections_root ../results/fusion_analysis/sections \
    --comments_root "path/to/comments" \
    --output_dir ../results/fusion_analysis/correlation \
    --model llama3
```

**Arguments:**
*   `--sections_root`: Directory containing the `_sections.json` files generated in Step 1.
*   `--comments_root`: 
    *   **Option A**: Path to a **DOCX file** (Student data format).
    *   **Option B**: Path to a **Directory** of `.json` files (YouTube data format).
*   `--output_dir`: Target directory for correlation results.

**Outputs:**
*   `.../correlation/<VideoID>_correlation.json`: Full data with matched comments.
*   `.../correlation/<VideoID>_correlation.docx`:
    *   **Student (DOCX input)**: "Coverage Analysis" report showing matched/unmatched comments and top matches per section.
    *   **YouTube (Dir input)**: "Statistical Summary" report with top 10 matches per section.

---

### Step 3: Collect Metrics

Run `collect_fusion_metrics.py` to aggregate comprehensive metrics from the correlation results into Excel files.

**Usage:**

```bash
# Generate both embodied and conventional metrics in one run
python collect_fusion_metrics.py --label both --output_dir ../results

# Or generate them separately
python collect_fusion_metrics.py --label embodied --output ../results/fusion_metrics_embodied.xlsx
python collect_fusion_metrics.py --label conventional --output ../results/fusion_metrics_conventional.xlsx
```

**Arguments:**
*   `--results_dir`: Path to results directory containing correlation subdirectories (default: `../results`)
*   `--label`: Which label to process: `embodied`, `conventional`, or `both` (default: `both`)
*   `--output`: Output Excel file path (used when label is not "both")
*   `--output_dir`: Output directory for Excel files (used when label is "both")
*   `--score_threshold`: Score threshold for "high score" classification (default: 60)

**Outputs:**
*   `fusion_metrics_embodied.xlsx`: Excel file with comprehensive metrics for all embodied videos
*   `fusion_metrics_conventional.xlsx`: Excel file with comprehensive metrics for all conventional videos

**Metrics Included:**
*   **Video Info**: video_id, label, duration, number of segments, comments used
*   **Comment Stats**: average/median comment word count
*   **Visual Correlation**: total pairs, correlated count/percentage, high score count/percentage, average/median scores
*   **Transcript Correlation**: total pairs, correlated count/percentage, high score count/percentage, average/median scores
*   **Union Metrics**: pairs where visual OR transcript is correlated, high score statistics
*   **Segment Statistics**: average correlations per segment, segments with correlations
