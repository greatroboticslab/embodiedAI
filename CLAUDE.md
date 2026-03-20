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

## Session 3 Progress (2026-03-18) — Three-Way Subgroup Split

### Bimodal Split Formalized
Embodied videos have a clear bimodal distribution in words-per-minute. Some have very few transcript words with hallucinated speech (silent demos — teaching is purely visual), while others have real narration describing the physical actions. These are fundamentally different teaching modalities and should be analyzed separately.

**Threshold:** 60 wpm (clean gap between 58.3 and 96.3 wpm — no ambiguous cases)

**Subgroup sizes:**
- 56 conventional
- 48 verbally embodied (≥60 wpm, real narration)
- 26 visually embodied (<60 wpm, silent demos with hallucinated/minimal transcripts)

The `subgroup` column was added to `master_analysis.csv`. The `analyze_results.py` script now produces both three-way and filtered (conv vs verbally embodied) comparisons.

### Conventional Videos — No Split Needed
Checked conventional videos for a similar bimodal pattern: **none found**. 55 of 56 conventional videos are 120–231 wpm (stdev=33.7, tight continuous distribution). Only 1 outlier (`zUyj7Xn9-sk`, 25 wpm, hallucination-flagged) but it still has corr_avg=88.4. Conventional videos are lecture-style by definition, so they are inherently verbal — no silent-demo modality exists.

### Key Findings — Three-Way Comparison

| Metric | Conventional | Verbally Embodied | Visually Embodied |
|---|---|---|---|
| Corr avg | 61.2 | 51.3 | 11.2 |
| Keypoint count | 3.36 | 2.94 | 0.35 |
| Avg keypoint strength | 3.58 | **3.67** | 0.40 |
| Embodied phrase count | 12.6 | 12.3 | 1.3 |
| Phrase rate (per 1000w) | 5.98 | **7.18** | 1.14 |
| visual_reference | **6.68** | 4.08 | 0.42 |
| action_narration | 3.41 | **5.56** | 0.58 |
| sensory_description | 0.71 | 0.79 | 0.04 |
| procedural_instruction | 1.59 | 1.92 | 0.31 |
| Transcript words | 2,637 | 2,267 | 96 |
| Words per min | 167 | 174 | 13.6 |

### Key Findings — Filtered (Conventional vs Verbally Embodied)

**Finding 5 — Correlation gap shrinks dramatically:**
When silent demos are removed, corr_avg gap shrinks from 23.5 points (61.2 vs 37.7) to just 10 points (61.2 vs 51.3). Most of the original gap was caused by visually embodied videos dragging down the embodied average.

**Finding 6 — Verbally embodied teachers have HIGHER keypoint strength:**
Avg keypoint strength is 3.67 for verbally embodied vs 3.58 for conventional. When embodied teachers do narrate, they reinforce their key teaching points more effectively than conventional lecturers.

**Finding 7 — Verbally embodied teachers use MORE embodied language per word:**
Phrase rate is 7.18 per 1000 words (verbally embodied) vs 5.98 (conventional). They pack more embodied action descriptions into their speech.

**Finding 8 — action_narration is 1.6x higher in verbally embodied:**
5.56 vs 3.41 action narration phrases per video. Verbally embodied teachers actively describe physical manipulations ("I'm connecting this wire", "now I'm screwing this in") far more than conventional teachers.

**Finding 9 — visual_reference explains the remaining corr gap:**
Conventional teachers use 6.68 visual reference phrases vs 4.08 for verbally embodied. These "look at this" / "as you can see" anchors point to slides/diagrams on screen, making student comments more likely to match transcript language. This inflates conventional correlation but doesn't indicate better teaching.

**Finding 10 — WPM is virtually identical when silent demos excluded:**
167 wpm (conventional) vs 174 wpm (verbally embodied). There is no speech rate difference — the original gap was entirely caused by silent demos.

### Plots Generated (Session 3)
All in `TranscriptAnalysis/results/gemini_analysis/plots/`:

**Three-way plots (3 colors: blue/green/orange):**
- `scatter_phrase_rate_vs_corr_3way.png`
- `scatter_keypoint_strength_vs_corr_3way.png`
- `scatter_wpm_vs_corr_3way.png`
- `position_distribution_3way.png`
- `category_breakdown_3way.png`

**Filtered plots (conv vs verbally embodied only):**
- `scatter_phrase_rate_vs_corr_filtered.png`
- `scatter_keypoint_strength_vs_corr_filtered.png`
- `scatter_wpm_vs_corr_filtered.png`
- `category_breakdown_filtered.png`

**Bimodal distribution visualization:**
- `wpm_bimodal_histogram.png` — histogram of embodied video WPM showing clean two-cluster split with 60 wpm threshold line

Note: `.gitignore` excludes `*.png`, so plots are local only (not pushed to GitHub).

---

## New File Locations (Sessions 2–3)

| What | Path |
|---|---|
| Speech ratio script | `TranscriptAnalysis/scripts/speech_ratio/compute_speech_ratio.py` |
| Speech ratio results | `TranscriptAnalysis/results/speech_ratio/speech_ratio_combined.csv` |
| Gemini analysis script | `TranscriptAnalysis/scripts/gemini_analysis/gemini_analyze_transcripts.py` |
| Gemini analysis + plots | `TranscriptAnalysis/scripts/gemini_analysis/analyze_results.py` |
| Per-video Gemini JSON | `TranscriptAnalysis/results/gemini_analysis/conventional/` and `embodied/` |
| Master analysis CSV | `TranscriptAnalysis/results/gemini_analysis/master_analysis.csv` (includes `subgroup` column) |
| Plots | `TranscriptAnalysis/results/gemini_analysis/plots/` (15 plots, local only — .gitignore excludes *.png) |

---

## Conda Environments
- `llava` — training
- `llava_infer` — inference with Qwen
- Base conda env used for embodiedAI scripts (has `google-generativeai`, `matplotlib`)
- Install: `pip install google-generativeai matplotlib` if missing
