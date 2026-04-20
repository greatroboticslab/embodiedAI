# embodiedAI Project — Context for Claude

## Project Overview
Comparative study of **embodied vs. conventional learning** in robotics education using 130 YouTube videos (56 conventional, 74 embodied). Two papers published from this dataset.

**Embodied videos:** Hands-on tutorials — hardware assembly, wiring, calibration, physical robot operation.
**Conventional videos:** Lecture-style — slides, math, diagrams, verbal explanation.

**Three-way subgroup split** (60 WPM threshold, clean gap):
- 56 conventional (lecture-style, ~167 wpm)
- 48 verbally embodied (≥60 wpm, narrated hands-on)
- 26 visually embodied (<60 wpm, silent demonstrations)

---

## Paper 1 — Transcript-Only Analysis (IEEE, DONE)
- **Title:** "A Comparative Study of Embodied and Non-Embodied Learning Approaches in Learning Robotics with LLM"
- **Files:** `Journal_paper.tex`, `finalpaper2026.tex`
- **Core thesis:** Audio/transcript alone CAN teach embodied content if the teacher is skilled
- **Key finding:** Transcript correlation gap (conv 61.2 vs emb 37.7) is driven by 26 silent demos dragging down the embodied average. After removing them, gap narrows to 61.2 vs 51.3.
- **Analyses:** Speech ratio, Gemini keypoint extraction, embodied phrase identification

## Paper 2 — Multimodal Fusion Analysis (Scientific Reports, IN REVISION)
- **Title:** "Multimodal Fusion Analysis of Visual and Verbal Engagement in Embodied Robotics Instruction"
- **File:** `Journal_of_Scientific_Reports_Ben_Second_paper/main.tex` (~1140 lines)
- **Core thesis:** Single-channel metrics (transcript OR visual) systematically mis-rank instructional modalities. Only dual-channel union measurement produces stable rankings.
- **Key finding:** Visual and transcript channels produce inverted rankings — hands-on ranks 1st visually (64.1%) but last verbally (12.5%). Union metric resolves this.

### Paper 2 Current State (Session 10, 2026-04-19)
- All Results subsections have full dual-channel treatment (visual + transcript + union)
- Three parallel OLS regressions, two-way ANOVAs for both channels
- Every Results subsection now has statistical tests (chi-square, ANOVA, t-tests, Pearson r)
- Engagement metrics formally defined in Methods ("Engagement metrics" subsection)
- Pipeline stages in paragraph format (professor preference)
- Representative video screenshots at 6 figure locations (extracted via ffmpeg)
- 9 figures, 13 tables, ~35 statistical tests in summary table

### Key Data Points (Paper 2)
- **Content distribution chi-square:** χ²=3,462.3, df=16, p<10⁻¹⁰⁰, Cramér's V=0.416
- **Visual two-way ANOVA:** Content type F=39.5 (sig), Subgroup F=1.7 (n.s. → full mediation)
- **Transcript two-way ANOVA:** Subgroup F=130.0 (sig), Content F=11.8 (sig)
- **Video-level:** visual r=+0.333, transcript r=−0.353 (opposite sign, same magnitude)
- **Union ranking:** VisEmb 79.1% > VerbEmb 66.7% > Conv 64.6%

### Metric Definitions
- **Per-pair rate** (Vis%, Tr%): fraction of comments correlated per segment, averaged across segments
- **Segment-level binary** (Union Seg%, Both Seg%): fraction of segments with any/both channels correlated
- **Union score**: max(visual_score, transcript_score) per segment — used in regression/video-level analyses
- Per-pair rates and segment-level binary are NOT interchangeable — always label which is used

---

## Key File Locations

### Paper 1
| What | Path |
|---|---|
| Paper | `Journal_paper.tex` / `finalpaper2026.tex` |
| Transcript files | `TranscriptAnalysis/data/transcripts_{conventional,embodied}/*.{txt,srt}` |
| Speech ratio | `TranscriptAnalysis/results/speech_ratio/speech_ratio_combined.csv` |
| Gemini analysis | `TranscriptAnalysis/scripts/gemini_analysis/gemini_analyze_transcripts.py` |
| Master CSV | `TranscriptAnalysis/results/gemini_analysis/master_analysis.csv` (includes `subgroup` column) |

### Paper 2
| What | Path |
|---|---|
| Paper | `Journal_of_Scientific_Reports_Ben_Second_paper/main.tex` |
| Screenshots | `Journal_of_Scientific_Reports_Ben_Second_paper/figures/*.jpg` |
| Video classification | `FusionAnalysis/results/video_classification/{conventional,embodied}/<video_id>.json` |
| Full segment data | `FusionAnalysis/results/classification_analysis/full_segment_dataset.csv` (9,997 segments) |
| Video-level metrics | `FusionAnalysis/results/classification_analysis/video_level_metrics_dual.csv` |
| Regression results | `FusionAnalysis/results/classification_analysis/regression_dual_channel_summary.txt` |
| Classification script | `FusionAnalysis/scripts/gemini_video_classify.py` |
| Analysis scripts | `FusionAnalysis/scripts/additional_analysis.py`, `dual_channel_analysis.py`, `dual_channel_regression_video.py`, `dual_channel_supplemental.py`, `compute_missing_stats.py` |
| Screenshot extraction | `FusionAnalysis/scripts/extract_screenshots.py` |
| Raw videos | `VideoAnalysis/rawvideos/{conventional,embodied}_videos/*.mp4` |

---

## User Preferences
- Keep all existing results — do not remove or overwrite them
- New scripts go in new files/folders, do not modify old scripts
- Gemini API key provided per session, not stored anywhere
- LLaVA should only do visual observations — no calculations (for FFT pipeline)
- Professor prefers paragraph-style method descriptions (not numbered subsections)

## Technical Notes
- **Conda envs:** `llava` (training), `llava_infer` (inference with Qwen), base env for embodiedAI scripts
- **Gemini API:** `gemini-2.0-flash` via `google-generativeai` SDK. Free tier: ~15 req/min, use 4s sleep.
- **HPC (Bridges-2):** V100 does NOT support bf16/tf32 — use `--fp16 True --tf32 False`. `RM-shared` needs `--qos=low`. ffmpeg available via `module load ffmpeg/4.3.1`.
- **Correlation scoring** (Stage 5 of fusion pipeline) uses **LLaMA 3 via Ollama**, NOT Gemini. Gemini is only for the separate video classification pipeline.
- **`.gitignore` excludes `*.png`** — plots are local only.
