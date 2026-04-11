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

## Session 4 Progress (2026-03-20) — Paper Updated

### Journal_paper.tex Updates
Paper (`Journal_paper.tex`) updated with all three-way subgroup analysis results:

**Abstract:** Lightly updated to mention bimodal finding and verbal embodied results as a key contribution.

**Methodology — 3 new subsections:**
- `Speech Rate Analysis` — WPM computation formula, hallucination detection methodology
- `Transcript Content Analysis` — two sub-subsections:
  - `Keypoint Extraction` — Gemini-based keypoint identification with strength rating 1–5
  - `Embodied Phrase Identification` — 4 categories (visual_reference, action_narration, sensory_description, procedural_instruction)
- `Subgroup Classification` — bimodal distribution discovery, 60 WPM threshold justification, 3 subgroups defined, conventional unimodal confirmed

**Results — 7 new subsections (after original 2-way results, kept intact):**
1. `Speech Rate Distribution and Subgroup Identification` — bimodal histogram (pgfplots `ybar interval` with all 73 raw WPM data points, red dashed threshold at 60 WPM)
2. `Three-Way Student Correlation Analysis` — table with $S_{corr}$, $W_{trans}$, $R_{wpm}$
3. `Three-Way YouTube Correlation Analysis` — table with $P_{corr}$, $E_{yt}$, $P_{top}$
4. `Keypoint Analysis by Subgroup` — two grouped bar charts (count + strength)
5. `Embodied Phrase Analysis by Subgroup` — phrase rate bar chart + 4-category breakdown
6. `Summary of Subgroup Findings` — full-width `table*` with 14 metrics × 3 subgroups

**Conclusion:** New section with 5 paragraphs — initial findings, confound discovery, filtered results, thesis support, future work directions.

**All new plots** use pgfplots/tikzpicture matching existing paper style (gray fills, error bars, dashed grids).

### Bibliography.bib
Gemini 2.0 reference added:
```bibtex
@article{geminiteam2024gemini2,
  title={Gemini 2.0: A Family of Highly Capable Multimodal Models},
  author={{Gemini Team, Google}},
  journal={arXiv preprint arXiv:2412.04948},
  year={2024}
}
```

### YouTube 3-Way Results
YouTube per-video data from `TranscriptAnalysis/results_youtube/metrics/` analyzed with same 3-way split:
- pct_correlated: 20% (conv) vs 21% (verb emb) vs 7% (vis emb) — gap eliminated for verbal embodied
- Engagement higher for verbally embodied: 38.2 vs 22.8 (conventional)
- top_correlated_pct similar: 56% vs 52% vs 27%

### Status
All changes pushed to `greatroboticslab/embodiedAI` on `main`. Paper has not been compiled — IEEEtran.cls not available on HPC. Structure verified programmatically (environment balance, label uniqueness).

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

## Session 5 Progress (2026-04-03) — Fusion Paper & Video Classification

### Second Paper: Fusion Analysis
- Target journal: Scientific Reports (Nature)
- Template: `Journal_of_Scientific_Reports_Ben_Second_paper/main.tex` (wlscirep class)
- Structure: Introduction, Results, Discussion, Methods (per journal template)
- Same 130 videos, same 3-way subgroup split (conventional / verbally embodied / visually embodied)

### Fusion Analysis Key Findings (from existing metrics)

| Metric | Conventional | Verbally Embodied | Visually Embodied |
|---|---|---|---|
| Visual corr % | 52.3 | 54.3 | **68.2** |
| Transcript corr % | 23.6 | 24.2 | 3.6 |
| Union corr % | 59.0 | 62.1 | **68.9** |
| Visual avg score | 48.2 | 50.2 | **62.2** |
| Transcript avg score | 20.5 | 21.1 | 3.4 |

**Key insight:** Transcript-only analysis (Paper 1) underestimates embodied learning because it ignores the visual channel where embodied content excels. Embodied videos outperform conventional on visual correlation — the opposite of transcript-only findings.

### New Analysis: Gemini Video Classification
To go deeper than group-level comparison, we classify each 10-second segment by content type and engagement features using Gemini's direct video understanding.

**Script:** `FusionAnalysis/scripts/gemini_video_classify.py`
**SLURM:** `FusionAnalysis/scripts/classify_videos.sbatch`

**Per-segment metrics collected from Gemini:**
1. **Content type** (9 mutually exclusive categories):
   - `hands_on_demonstration`, `equipment_closeup`, `presenter_talking`, `slide_or_powerpoint`, `diagram_or_whiteboard`, `screen_or_software`, `animation_or_graphic`, `device_in_operation`, `other`
2. **Audio-visual alignment score** (0-100): how well narration matches visuals
3. **Narration present** (true/false)
4. **Hands visible** (true/false)
5. **Instructional density** (1-5)

**Design decisions:**
- Videos uploaded to Gemini File API — Gemini watches actual video (not summaries or frames)
- Processed in 5-minute chunks (~30 segments per call) to keep output reliable
- Prompt tells Gemini exact number of segments expected per chunk
- ~427 API calls total for 130 videos, ~30-45 min runtime
- Resume support: saves after each chunk, skips completed videos on re-run
- Video durations pre-computed in `FusionAnalysis/results/video_durations.json` (ffprobe not available on HPC)

**Output:** `FusionAnalysis/results/video_classification/conventional/<video_id>.json` and `embodied/<video_id>.json`

**Planned analysis (next session):**
- Aggregate engagement by content type across all videos → "physical demonstration segments have X% higher visual correlation than slide segments"
- Content type distribution by subgroup → embodied videos have more hands-on segments
- Alignment score vs engagement correlation → does audio-visual synchrony predict engagement?
- Temporal engagement patterns by content type
- Statistical tests (ANOVA / t-test across content types)

**Goal:** Move from "embodied videos are more engaging" (group-level claim) to "physical demonstration content drives higher visual engagement, and embodied videos contain proportionally more of it" (mechanistic explanation).

### SLURM Notes
- `RM-shared` partition requires `--qos=low` for this account
- No GPU needed — script only makes Gemini API calls

---

## New File Locations (Session 5)

| What | Path |
|---|---|
| Fusion paper template | `Journal_of_Scientific_Reports_Ben_Second_paper/main.tex` |
| Video classification script | `FusionAnalysis/scripts/gemini_video_classify.py` |
| Video classification SLURM | `FusionAnalysis/scripts/classify_videos.sbatch` |
| Video durations JSON | `FusionAnalysis/results/video_durations.json` |
| Classification output (conv) | `FusionAnalysis/results/video_classification/conventional/` |
| Classification output (emb) | `FusionAnalysis/results/video_classification/embodied/` |

---

## Session 6 Progress (2026-04-04) — Fusion Paper Written

### Paper Content Written
Wrote the full fusion analysis paper in `Journal_of_Scientific_Reports_Ben_Second_paper/main.tex` using the Scientific Reports (wlscirep) template format.

**Sections written:**
- **Abstract**: Summarizes multimodal fusion approach, key findings (hands-on at 64.1% visual corr, content distribution differences, hand visibility effect), mechanistic conclusion
- **Introduction**: Motivates study from Paper 1's transcript-only limitation, introduces dual-channel framework, central hypothesis
- **Results** (7 subsections with pgfplots figures + tables):
  1. Engagement by content type — horizontal bar chart, ANOVA F=49.4, pairwise t-tests
  2. Content distribution by subgroup — bar chart (6.4% vs 27.0% vs 70.0% hands-on), full distribution table
  3. Subgroup visual engagement — bar chart of hands-on engagement, ANOVA F=99.4
  4. Hand visibility effect — bar chart (56.3% vs 44.3%), p<10^-40
  5. Narration effect — table showing dual-channel tradeoff
  6. Instructional density — bar chart across 5 levels
  7. Top vs bottom quartile — table (40.4% vs 12.9% hands-on in verbally embodied)
- **Discussion**: Three principal findings, practical implications, limitations
- **Methods**: Dataset, fusion framework, Gemini classification pipeline, statistical methods
- **Statistical tests summary table**: All 9 tests with F/t statistics and p-values

**Style:** All plots use pgfplots/tikzpicture with gray!60 fills, dashed grids, nodes near coords — matching Paper 1 conventions. All data embedded inline via filecontents (no external data files needed).

### Bibliography Updated
`sample.bib` updated with 7 references: wilson2002six, barsalou2008grounded, johnson2014embodied, zhang2025transcript, liu2023visual, radford2023robust, geminiteam2024gemini2.

### Compilation
No external images needed — all figures are pgfplots. Compile with: `pdflatex → bibtex → pdflatex → pdflatex`.

---

## Session 7 Progress (2026-04-10) — Paper Expanded with New Analyses

### Problem
Paper was only 8 pages. Professor asked to enrich Methods and add more Results.

### Methods Expansion (~1 page → ~3.5 pages)
- **Pipeline figure** (tikz flowchart) showing all 6 stages of the fusion pipeline
- **Detailed stage descriptions:**
  1. Frame extraction (OpenCV, every 100 frames, ~1 frame per 3.3s at 30fps)
  2. Audio transcription (Whisper turbo)
  3. Dual captioning (LLaVA + MiniCPM → LLaMA 3 integration via Ollama)
  4. Temporal alignment and 10-second segmentation
  5. Dual-channel correlation scoring (LLaMA 3 via Ollama — **corrected from "Gemini-based"**)
  6. Metric aggregation
- **Expanded classification methodology**: 5-min chunking, prompt design, SLURM processing, retry logic
- **Data integration** subsection explaining the join (9,997 segments, 128 videos)
- **Expanded statistical analysis**: two-way ANOVA, regression, temporal bins, channel independence

**Important correction:** The correlation scoring (Stage 5) uses LLaMA 3 via Ollama, NOT Gemini. Gemini is only used for the separate video content classification pipeline. This was incorrect in the Session 6 draft.

### New Results (5 new subsections + updated stats table)

1. **Content type mediates subgroup effect** (Two-way ANOVA)
   - Content type: F=39.5, p<10^-54, η²=3.03%
   - Subgroup main effect: F=1.7, **p=0.191** (non-significant after controlling for content type!)
   - Interaction: F=8.4, p<10^-19, η²=1.29%
   - **Key finding:** The "embodied" label carries NO independent explanatory power — engagement is entirely explained by content composition

2. **Multiple regression** (OLS, R²=0.050, F=40.2, p<10^-100)
   - Significant positive: hands_on_demonstration (β=+6.2), diagram (β=+3.9), hands_visible (β=+3.6)
   - Significant negative: screen_software (β=-13.2, strongest), device_in_operation (β=-10.1), other (β=-13.7)
   - Subgroup effects remain significant (β=+6.0 verb emb, β=+16.4 vis emb) — reflects quality advantage within content types
   - Narration (p=0.069) and instructional density (p=0.106) NOT significant

3. **Video-level scatter plot** (verbally embodied, n=48)
   - Pearson r=0.53, p=1.2×10^-4 between %hands-on and mean visual score
   - Conventional: r=-0.03 (too little hands-on content for variance)
   - Visually embodied: r=0.04 (ceiling effect, already ~70% hands-on)
   - Scatter plot uses all 48 verified data points with OLS regression line (y=33.8+0.547x)

4. **Temporal dynamics** (early/middle/late bins)
   - ANOVA: F=9.5, p<10^-4
   - Conventional declines monotonically: 46.6 → 41.2 → 38.6 (viewer fatigue)
   - Verbally embodied stable: 47.5 → 48.2 → 45.8
   - Visually embodied peaks mid-video: 58.3 → **65.4** → 59.3
   - Grouped bar chart by subgroup × temporal bin

5. **Channel independence** (visual-transcript correlation)
   - Overall: r=0.14 (weak positive, largely independent channels)
   - Conventional: r=0.20 (tightest coupling — diagrams described while pointing)
   - Verbally embodied: r=0.14
   - Visually embodied: r=0.06, p=0.052 (borderline non-significant — nearly zero coupling)

6. **Paper 1 comparison table** (ranking reversal)
   - Paper 1 transcript-only ranking: Conv (61.2%) > VerbEmb (51.3%) >> VisEmb (11.2%)
   - Paper 2 visual-channel ranking: VisEmb (68.2%) > VerbEmb (54.3%) > Conv (52.3%)
   - Complete ranking reversal verified from actual data files
   - Note: metrics are not directly comparable (whole-video topic corr vs segment-level visual corr) — framed as ranking comparison

### Discussion Updated
- Expanded from 3 to 4 principal findings (added content type mediation)
- Added temporal dynamics discussion
- Practical implications section expanded (screen content warning)

### Bibliography Updated
Added 2 new entries: yao2024minicpm (MiniCPM-V), touvron2023llama (LLaMA). Total: 9 references.

### New Files
| What | Path |
|---|---|
| Additional analysis script | `FusionAnalysis/scripts/additional_analysis.py` |
| Regression summary | `FusionAnalysis/results/classification_analysis/regression_summary.txt` |
| Video-level metrics | `FusionAnalysis/results/classification_analysis/video_level_metrics.csv` |
| Temporal analysis | `FusionAnalysis/results/classification_analysis/temporal_analysis.csv` |
| Channel correlations | `FusionAnalysis/results/classification_analysis/visual_transcript_correlation.csv` |

---

## Session 8 Progress (2026-04-10) — Dual-Channel Rebalance

### Problem
Session 7 draft focused too heavily on visual correlation. Transcript channel was underused, and no union/intersection analysis existed — the paper did not fully exploit the dual-channel framework it claimed to establish.

### Fix 1 — Figure 1 Caption Accuracy
Original caption said visual correlation values showed "fraction of segments classified as visually correlated" but the metric is actually mean of per-segment `visual_pct_correlated` (a per-pair rate averaged across segments, not a segment-level binary). Caption clarified: "mean per-segment visual correlation rate" to avoid conflating per-pair and segment-level binary metrics.

### Fix 2 — New Dual-Channel Analysis Script
Created `FusionAnalysis/scripts/dual_channel_analysis.py` which computes:
- `vis_corr` / `tr_corr` binary flags (segment has any correlated comment on that channel)
- `union_corr = vis_corr OR tr_corr` (segment-level binary)
- `both_corr = vis_corr AND tr_corr` (segment-level binary)
- `union_score = max(visual_avg_score, transcript_avg_score)` (0–100)

Run with `conda run -n llava python3 dual_channel_analysis.py`. Outputs three CSVs:
- `engagement_by_content_type_all_channels.csv`
- `engagement_by_subgroup_all_channels.csv`
- `temporal_analysis_all_channels.csv`

### Key Stats Added

**Transcript per-pair rates by content type** (new — mirror of Figure 1):
- Slide/PowerPoint: 22.8%  |  Diagram: 22.2%  |  Presenter: 21.8%  |  Animation: 21.2%
- Device in op: 20.4%  |  Equipment closeup: 15.7%  |  Screen/SW: 13.4%  |  Other: 13.1%
- Hands-on: **12.5%** (lowest — inverted ranking from visual channel)

**Transcript ANOVA:** F=23.2 by content type, **F=141.4 by subgroup** (vs. visual F=1.7 after controlling for content type — the two channels are driven by opposing factors)

**Two-way ANOVA transcript:** content F=11.8 (η²=0.91%), **subgroup F=130.0 (η²=2.50%)**, interaction F=4.6

**Subgroup per-pair rates:**
| Subgroup | Vis% | Tr% | Union% (seg-lvl) | Both% (seg-lvl) |
|---|---|---|---|---|
| Conventional | 45.8 | 20.4 | 64.6 | 22.3 |
| Verbally embodied | 51.0 | 19.9 | 66.7 | 21.3 |
| Visually embodied | 67.0 | 2.8 | 79.1 | 4.5 |

**Hand visibility transcript tradeoff:** 15.8% (hands visible) vs. 20.6% (not visible), t=-7.6, p<10^-13. Hands boost visual (+12.0 pp) but reduce transcript (-4.8 pp) — resource-competition model.

**Narration conservation:** Union engagement nearly identical with vs. without narration (67.0% vs. 67.9%). Narration redistributes attention between channels rather than adding to total engagement.

### Paper Updates (main.tex)
- **Abstract**: Rewritten to reflect dual-channel framework. Includes both visual and transcript rankings, ANOVA asymmetry, union/intersection findings.
- **Figure 1 caption**: Corrected to "mean per-segment visual correlation rate".
- **New Figure fig:tr_by_content**: Mirror of Figure 1 for transcript channel (horizontal bar chart, 9 content types).
- **Table 1 (tab:content_dual)**: Expanded from 5 to 8 columns — added `Union Seg.%` and `Both Seg.%` columns (segment-level binary). Caption clarifies metric definitions.
- **New section "Dual-channel engagement by subgroup"** with tab:subgroup_dual.
- **New transcript two-way ANOVA table (tab:twoway_tr)** showing subgroup F=130.0 contrast with visual F=1.7.
- **Narration table (tab:narration)**: Updated with dual-channel metrics including union conservation.
- **New dual-channel temporal table (tab:temporal_all)**.
- **Statistical tests summary table (tab:stats)**: Added transcript ANOVAs.
- **Discussion**: Rewritten from 4 findings to **5 principal findings**:
  1. Inverted rankings of content effectiveness between channels
  2. Content type drives visual / subgroup drives transcript (asymmetric ANOVAs)
  3. Union/intersection reveal hands-on and diagrams are effective for different reasons
  4. Engagement advantage scales from segments to videos
  5. Channel tradeoffs govern narration and hand visibility effects

### Critical Metric Consistency Fix
**Bug:** Two different metrics were mixed early in drafting:
- Per-pair rate: mean of `visual_pct_correlated` across segments (e.g., hands-on = 64.1)
- Segment-level binary: fraction of segments with `pct_correlated > 0` (e.g., hands-on visual = 74.9)

**Resolution:** Per-pair rates used for per-channel columns (Vis%, Tr%) throughout the paper, consistent with original Figure 1. Segment-level binary reserved for Union%/Both% where it is definitionally correct, with explicit `Seg.%` labeling. All Discussion and Abstract values updated for consistency:
- Hands-on transcript: 19.2 → **12.5** (per-pair)
- Slide transcript: 33.5 → **22.8** (per-pair)
- Conv vs. VerbEmb transcript: 30.4/28.9 → **20.4/19.9** (per-pair)
- Visually embodied transcript: 5.1 → **2.8** (per-pair)
- Hand visibility transcript delta: 7.1 → **4.8** (per-pair difference)

Union/intersection values (78.5%, 26.9%, 79.1%, 67.0%/67.9%) correctly kept as segment-level binary.

### New Files (Session 8)
| What | Path |
|---|---|
| Dual-channel analysis script | `FusionAnalysis/scripts/dual_channel_analysis.py` |
| Content type all channels | `FusionAnalysis/results/classification_analysis/engagement_by_content_type_all_channels.csv` |
| Subgroup all channels | `FusionAnalysis/results/classification_analysis/engagement_by_subgroup_all_channels.csv` |
| Temporal all channels | `FusionAnalysis/results/classification_analysis/temporal_analysis_all_channels.csv` |

---

## Conda Environments
- `llava` — training
- `llava_infer` — inference with Qwen
- Base conda env used for embodiedAI scripts (has `google-generativeai`, `matplotlib`)
- Install: `pip install google-generativeai matplotlib` if missing
