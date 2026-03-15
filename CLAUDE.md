# embodiedAI Project — Context for Claude

## Project Overview
Comparative study of **embodied vs. conventional learning** in robotics education using YouTube videos.
- Paper title: "A Comparative Study of Embodied and Non-Embodied Learning Approaches in Learning Robotics with LLM"
- Format: IEEE journal paper (`Journal_paper.tex`, `finalpaper2026.tex`)
- Dataset: 74 embodied videos + 56 conventional videos (~130 total)

**Embodied videos:** Hands-on tutorials — hardware assembly, wiring, calibration, physical robot operation.
**Conventional videos:** Lecture-style — slides, math, diagrams, verbal explanation.

---

## Current TranscriptAnalysis Results (favor conventional — this is the confound problem)

| Metric | Conventional | Embodied |
|---|---|---|
| Avg transcript words | 2,637 | 1,531 |
| Avg topics | 4.4 | 3.7 |
| Avg comments | 1.91 | 1.69 |
| Avg corr_avg | **61.2** | **37.7** |
| Engagement rate | 0.17 | 0.20 (embodied higher) |

**Why results favor conventional (the confound):** Conventional videos are more verbal (more words, more topics covered), which inflates correlation scores. Many embodied videos show without narrating — the teacher assembles/wires things silently — so their transcripts are sparse. This is not because embodied content is worse, but because the "embodied" part is visual, not verbal.

---

## Paper Goal / Core Thesis
The paper argues that **audio/transcript alone CAN teach embodied content** — if the teacher is skilled. Like pre-1970s teachers who taught hands-on subjects purely through verbal description without computers or video. Some teachers are able to do this well, others are not. This is the unique finding of the paper.

We want to show:
1. What drives the correlation results (explain the confound via 3 new variables)
2. How well pure audio conveys embodied components
3. How correlation changes with keypoint number/strength
4. How embodied keywords in audio change correlation

**This paper is audio/transcript ONLY.** Visual analysis is a separate future paper.

---

## Three New Analyses to Add

### Analysis 1 — Speech Ratio (speech time / video duration)
- **What:** Proportion of the video that has actual speech vs silence
- **Why:** Embodied videos often have silent segments (teacher shows without talking). Low speech ratio = low transcript = low correlation — explains the confound
- **How:** Parse existing `.srt` files (Whisper output, already have them for all videos) to sum speech segment durations, divide by `duration_s`
- **File location:** `TranscriptAnalysis/data/transcripts_conventional/*.srt` and `transcripts_embodied/*.srt`
- **Fallback:** If SRT parsing has issues, use `transcript_words / duration_s` (words per minute)

### Analysis 2 — Keypoint Analysis (via Gemini on raw transcripts)
- **What:** Extract keypoints from each video's transcript, with strength/clarity
- **Two dimensions:** (1) Number of keypoints, (2) Strength of each keypoint
- **Keypoint strength definition:** How often it's mentioned, whether it's revisited throughout the video, and whether it's reinforced at the end — both repetition and structural return add strength
- **Output from Gemini (maximum info, process later):**
  - Keypoint name + brief description
  - Number of times mentioned
  - Position distribution (early / mid / late / throughout)
  - Whether reinforced at the end
  - Strength rating 1–5 with reasoning
- **Goal in paper:** Show scatter plots of keypoint strength vs. corr_avg (colored by embodied/conventional), reveal whether strong/clear keypoints predict higher correlation

### Analysis 3 — Embodied Keywords (via Gemini on raw transcripts)
- **What:** Identify and count embodied action phrases in transcript audio
- **Examples:** "look at this", "connect the wire to", "you can feel the tension", "hold it steady while", "press the button"
- **Categories to detect:**
  - Visual reference: "look at this", "as you can see", "watch how"
  - Action narration: "connect the wire to", "I'm now attaching", "screw this in"
  - Sensory description: "you can feel", "notice how it clicks"
  - Procedural instruction: "hold it steady", "make sure this is aligned"
- **Output from Gemini:** Every phrase verbatim + category + position in transcript (%) + total count per category
- **Goal in paper:** Show that some conventional videos AND embodied videos use rich verbal embodied language, and those have higher correlation — supporting the professor's thesis

---

## Implementation Plan

### Step 1 — SRT Speech Ratio Script
- Parse `.srt` files for all ~130 videos
- Sum speech segment durations
- Compute `speech_time / duration_s` per video
- Output: add `speech_ratio` column to existing metrics CSVs
- No new API calls needed

### Step 2 — Gemini Script (Analyses 2 + 3 combined)
- One Gemini API call per video (combine keypoints + embodied keywords in single prompt)
- Input: raw transcript `.txt` files from `TranscriptAnalysis/data/transcripts_*/`
- ~130 calls total — feasible at free tier (15 req/min, 4s sleep, ~10 min total)
- Output: JSON per video with full keypoint + embodied keyword data
- Gemini API key provided by user per session (not stored)
- Prefer newer/capable Gemini model (user to confirm which model they have access to)

### Step 3 — Statistical Analysis
- Scatter plots: speech_ratio vs corr_avg, keypoint_strength vs corr_avg, embodied_keyword_count vs corr_avg
- Check for bimodal split in embodied videos (verbally embodied vs visually embodied)
  - If clear bimodal: classify and compare explicitly
  - If not: use speech_ratio as continuous variable
- Keep ALL existing TranscriptAnalysis results in paper (do not remove)
- Add new results as additional sections to explain/contextualize the confound

---

## Key File Locations

| What | Path |
|---|---|
| Main paper | `embodiedAI/Journal_paper.tex` |
| Transcript text files | `TranscriptAnalysis/data/transcripts_conventional/*.txt` |
| Transcript SRT files | `TranscriptAnalysis/data/transcripts_conventional/*.srt` |
| Embodied transcripts | `TranscriptAnalysis/data/transcripts_embodied/*.txt` and `*.srt` |
| Existing topic JSONs | `TranscriptAnalysis/data/topics_conventional/` and `topics_embodied/` |
| Per-video metrics CSV | `TranscriptAnalysis/results/conventional/Conventional_metrics.csv` |
| Per-video metrics CSV | `TranscriptAnalysis/results/embodied/Embodied_metrics.csv` |
| Aggregate results | `TranscriptAnalysis/results/conventional/Conventional_aggregate.csv` |
| Aggregate results | `TranscriptAnalysis/results/embodied/Embodied_aggregate.csv` |
| Analysis scripts | `TranscriptAnalysis/scripts/` |
| Raw videos | `VideoAnalysis/rawvideos/` |

---

## User Preferences
- Keep all existing results — do not remove or overwrite them
- New scripts go in new files/folders, do not modify old scripts
- Gemini analyzes raw transcript `.txt` files (not existing topic JSONs — those used Ollama/LLaMA locally and are lower quality)
- This paper: audio/transcript ONLY — no visual/frame analysis (that is a separate future paper)
- Gemini API key provided per session, not stored anywhere
- Output maximum info from Gemini prompts — can always filter/process later

---

## Session 2 Progress (2026-03-14) — All Three Analyses Completed

### Analysis 1 — Speech Ratio: DONE ✓
- Script: `TranscriptAnalysis/scripts/speech_ratio/compute_speech_ratio.py`
- Output: `TranscriptAnalysis/results/speech_ratio/speech_ratio_combined.csv`
- Key finding: **30% of embodied videos (22/74) are "silent demos"** — Whisper hallucination flags near-zero real speech (5–30 wpm). Only 2% of conventional (1/56) are silent. These are flagged with `hallucination_flag=1`.
- Once silent demos excluded, verbal rate is nearly identical: conventional 170 wpm vs embodied 166 wpm.
- `words_per_min` is the primary audio-richness metric (cleaner than raw `speech_ratio` which breaks on hallucination).

### Analysis 2+3 — Gemini Keypoints + Embodied Keywords: DONE ✓
- Script: `TranscriptAnalysis/scripts/gemini_analysis/gemini_analyze_transcripts.py`
- Analysis + plots: `TranscriptAnalysis/scripts/gemini_analysis/analyze_results.py`
- Model used: `gemini-2.0-flash` via `google-generativeai` SDK (google-genai incompatible with Python 3.12)
- Output per video: `TranscriptAnalysis/results/gemini_analysis/conventional/<video_id>.json` and `embodied/<video_id>.json`
- Summary CSV: `TranscriptAnalysis/results/gemini_analysis/master_analysis.csv` (130 rows, all metrics joined)
- Plots: `TranscriptAnalysis/results/gemini_analysis/plots/` (5 plots)

### Key Findings from Gemini Analysis

| Metric | Conventional | Embodied |
|---|---|---|
| Corr. avg | 61.2 | 37.7 |
| Keypoint count | **3.36** | 2.03 |
| Avg keypoint strength (1–5) | **3.58** | 2.52 |
| Embodied phrase count | 12.6 | 8.5 |
| Embodied phrase rate (per 1000w) | 5.98 | 5.06 |
| visual_reference phrases | **6.68** | 2.80 |
| action_narration phrases | 3.41 | **3.81** |
| Words per min (all) | 167 | 118 |

**Finding 1 — Keypoint strength explains the correlation gap:**
Conventional videos have more keypoints (3.4 vs 2.0) and stronger keypoints (3.6 vs 2.5). Stronger, more focused keypoints mean student comments are more likely to match the transcript topics → higher corr_avg.

**Finding 2 — visual_reference is the biggest category difference:**
Conventional teachers say "look at this", "as you can see" 2.4× more often than embodied teachers (6.68 vs 2.80). These phrases anchor student comments to specific transcript moments → inflates conventional correlation.

**Finding 3 — action_narration is nearly equal (supports professor's thesis):**
Embodied teachers who DO talk describe physical actions at nearly the same rate as conventional teachers (3.81 vs 3.41). The gap is in total verbal volume, not in how actions are described. This supports the claim that skilled verbal embodied teaching exists.

**Finding 4 — Bimodal split in embodied videos:**
Clear two groups: 22 "visually embodied" (silent demos, <30 wpm) and 51 "verbally embodied" (~165 wpm). Not a continuous distribution — classification is justified.

### Next Step
Write new results sections in the paper integrating these three analyses. Proposed structure:
- Subsection: Speech ratio / verbal density (Analysis 1)
- Subsection: Keypoint analysis (Analysis 2)
- Subsection: Embodied keyword analysis (Analysis 3)
- Discussion: How these three variables explain the confound and support the core thesis

---

## New File Locations (Session 2)

| What | Path |
|---|---|
| Speech ratio script | `TranscriptAnalysis/scripts/speech_ratio/compute_speech_ratio.py` |
| Speech ratio results | `TranscriptAnalysis/results/speech_ratio/speech_ratio_combined.csv` |
| Gemini analysis script | `TranscriptAnalysis/scripts/gemini_analysis/gemini_analyze_transcripts.py` |
| Gemini analysis + plots | `TranscriptAnalysis/scripts/gemini_analysis/analyze_results.py` |
| Per-video Gemini JSON | `TranscriptAnalysis/results/gemini_analysis/conventional/` and `embodied/` |
| Master analysis CSV | `TranscriptAnalysis/results/gemini_analysis/master_analysis.csv` |
| Plots | `TranscriptAnalysis/results/gemini_analysis/plots/` |

---

## Conda Environments
- `llava` — training
- `llava_infer` — inference with Qwen
- Base conda env used for embodiedAI scripts (has `google-generativeai`, `matplotlib`)
- Install: `pip install google-generativeai matplotlib` if missing
