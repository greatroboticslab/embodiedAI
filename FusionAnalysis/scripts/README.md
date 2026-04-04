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

---

### Step 4: Classify Video Segments (Gemini Video Understanding)

Run `gemini_video_classify.py` to classify each 10-second segment of every video using Gemini's direct video understanding. The script uploads videos to the Gemini File API and asks Gemini to watch the video and classify what is happening in each segment.

**Prerequisites:**
*   Gemini API key (passed via `--api_key` or `GOOGLE_API_KEY` environment variable)
*   `google-genai` or `google-generativeai` Python SDK installed
*   Video files in `VideoAnalysis/rawvideos/conventional_videos/` and `embodied_videos/`
*   Video durations file at `../results/video_durations.json` (pre-computed from fusion metrics)

**Usage:**

```bash
# Process all videos
python gemini_video_classify.py --api_key YOUR_KEY_HERE

# Process only one label
python gemini_video_classify.py --api_key YOUR_KEY_HERE --label conventional

# Dry run (no API calls)
python gemini_video_classify.py --api_key YOUR_KEY_HERE --dry_run
```

**SLURM submission (recommended for full run):**

```bash
export GOOGLE_API_KEY='your_key_here'
sbatch --export=GOOGLE_API_KEY classify_videos.sbatch
```

**Arguments:**
*   `--api_key`: Gemini API key (required)
*   `--model`: Gemini model name (default: `gemini-2.0-flash`)
*   `--label`: Which video set to process: `embodied`, `conventional`, or `both` (default: `both`)
*   `--output_dir`: Output directory for classification JSONs (default: `../results/video_classification`)
*   `--sleep`: Seconds between API calls (default: 4)
*   `--dry_run`: Print what would be done without calling API

**How it works:**
1.  Each video is uploaded once to the Gemini File API
2.  The video is processed in 5-minute chunks (~30 segments per call) to keep output reliable
3.  For each chunk, Gemini watches that portion and classifies every 10-second interval
4.  Results are saved after each chunk (resume-safe — re-run to continue after interruption)
5.  The uploaded file is deleted after processing to free Gemini storage

**Per-segment metrics:**
*   **content_type**: one of 9 categories — `hands_on_demonstration`, `equipment_closeup`, `presenter_talking`, `slide_or_powerpoint`, `diagram_or_whiteboard`, `screen_or_software`, `animation_or_graphic`, `device_in_operation`, `other`
*   **alignment_score**: 0-100 — how well audio narration matches what is shown visually
*   **narration**: true/false — is someone speaking?
*   **hands_visible**: true/false — are hands actively doing something on screen?
*   **instructional_density**: 1-5 — how much teaching content is in this interval

**Outputs:**
*   `results/video_classification/conventional/<video_id>.json`
*   `results/video_classification/embodied/<video_id>.json`

**Output JSON format:**
```json
{
  "video_id": "0Mu2L9z1MH8",
  "duration": 434.0,
  "total_expected_segments": 44,
  "segments_classified": 44,
  "complete": true,
  "segments": [
    {
      "segment_index": 1,
      "start": 0,
      "end": 10,
      "content_type": "hands_on_demonstration",
      "alignment_score": 85,
      "narration": true,
      "hands_visible": true,
      "instructional_density": 4
    }
  ]
}
```
