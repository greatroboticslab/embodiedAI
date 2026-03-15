#!/usr/bin/env python3
"""
analyze_results.py

Loads Gemini analysis results and speech_ratio data, produces:
  1. Summary statistics table (conventional vs embodied)
  2. Scatter plots: embodied_phrase_rate vs corr_avg (colored by label)
  3. Position distribution plot: embodied phrase density across transcript thirds
  4. Keypoint strength vs corr_avg scatter plot
  5. Category breakdown bar chart (visual_ref / action_narration / sensory / procedural)

All plots saved to results/gemini_analysis/plots/

Usage:
  cd TranscriptAnalysis/scripts/gemini_analysis/
  python analyze_results.py
"""

import os
import json
import csv
import math
from collections import defaultdict

# ── Paths ─────────────────────────────────────────────────────────────────────
HERE        = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.dirname(HERE)
BASE        = os.path.dirname(SCRIPTS_DIR)

GEMINI_DIR    = os.path.join(BASE, "results", "gemini_analysis")
SPEECH_CSV    = os.path.join(BASE, "results", "speech_ratio", "speech_ratio_combined.csv")
METRICS_CONV  = os.path.join(BASE, "results", "conventional", "Conventional_metrics.csv")
METRICS_EMB   = os.path.join(BASE, "results", "embodied",     "Embodied_metrics.csv")
PLOTS_DIR     = os.path.join(GEMINI_DIR, "plots")

CATEGORIES = ["visual_reference", "action_narration", "sensory_description", "procedural_instruction"]
N_BINS = 10   # divide transcript into 10% segments for position distribution


# ── Data loading ──────────────────────────────────────────────────────────────

def load_corr_map():
    """Load corr_avg per video from existing metrics CSVs. Returns {video_id: corr_avg}."""
    result = {}
    for path in [METRICS_CONV, METRICS_EMB]:
        if not os.path.isfile(path):
            continue
        with open(path) as f:
            for row in csv.DictReader(f):
                vid = row.get("video_id", "").strip()
                val = row.get("corr_avg", "").strip()
                if vid and val:
                    try:
                        result[vid] = float(val)
                    except ValueError:
                        pass
    return result


def load_speech_map():
    """Load words_per_min and speech_ratio per video. Returns {video_id: dict}."""
    result = {}
    if not os.path.isfile(SPEECH_CSV):
        return result
    with open(SPEECH_CSV) as f:
        for row in csv.DictReader(f):
            vid = row.get("video_id", "").strip()
            if vid:
                result[vid] = {
                    "speech_ratio":       row.get("speech_ratio", ""),
                    "words_per_min":      row.get("words_per_min", ""),
                    "transcript_words":   row.get("transcript_words", ""),
                    "hallucination_flag": row.get("hallucination_flag", "0"),
                }
    return result


def load_gemini_results():
    """
    Load all per-video Gemini JSON results.
    Returns list of dicts, one per video, with fields:
      video_id, label, keypoints, embodied_phrases, embodied_summary
    """
    records = []
    for label in ["conventional", "embodied"]:
        set_dir = os.path.join(GEMINI_DIR, label)
        if not os.path.isdir(set_dir):
            print(f"[WARN] No results dir for {label}: {set_dir}")
            continue
        for fn in sorted(os.listdir(set_dir)):
            if not fn.endswith(".json"):
                continue
            vid = fn[:-5]
            with open(os.path.join(set_dir, fn)) as f:
                data = json.load(f)
            parsed = data.get("parsed") or {}
            records.append({
                "video_id":        vid,
                "label":           label,
                "keypoints":       parsed.get("keypoints") or [],
                "embodied_phrases":parsed.get("embodied_phrases") or [],
                "embodied_summary":parsed.get("embodied_summary") or {},
                "parse_success":   1 if parsed else 0,
            })
    return records


def build_master_table(records, corr_map, speech_map):
    """
    Join Gemini results with corr_avg and speech metrics.
    Returns list of flat dicts suitable for analysis.
    """
    rows = []
    for r in records:
        vid   = r["video_id"]
        words = int(speech_map.get(vid, {}).get("transcript_words") or 0)
        es    = r["embodied_summary"]
        kps   = r["keypoints"]

        # Embodied phrase rate: phrases per 1000 transcript words
        total_phrases = es.get("total_count") or 0
        phrase_rate   = (total_phrases / words * 1000) if words > 0 else 0.0

        # Keypoint metrics
        scores = [k.get("strength_score") for k in kps
                  if isinstance(k.get("strength_score"), (int, float))]
        avg_strength = sum(scores) / len(scores) if scores else 0.0
        max_strength = max(scores) if scores else 0

        by_cat = es.get("by_category") or {}
        rows.append({
            "video_id":               vid,
            "label":                  r["label"],
            "corr_avg":               corr_map.get(vid),
            "transcript_words":       words,
            "words_per_min":          speech_map.get(vid, {}).get("words_per_min", ""),
            "speech_ratio":           speech_map.get(vid, {}).get("speech_ratio", ""),
            "hallucination_flag":     speech_map.get(vid, {}).get("hallucination_flag", "0"),
            "keypoint_count":         len(kps),
            "avg_strength_score":     round(avg_strength, 2),
            "max_strength_score":     max_strength,
            "embodied_total":         total_phrases,
            "embodied_phrase_rate":   round(phrase_rate, 2),
            "visual_reference":       by_cat.get("visual_reference", 0),
            "action_narration":       by_cat.get("action_narration", 0),
            "sensory_description":    by_cat.get("sensory_description", 0),
            "procedural_instruction": by_cat.get("procedural_instruction", 0),
            "verbal_richness":        es.get("verbal_embodied_richness", ""),
            "embodied_phrases_raw":   r["embodied_phrases"],
            "parse_success":          r["parse_success"],
        })
    return rows


# ── Statistics ────────────────────────────────────────────────────────────────

def group_stats(rows, key, label_filter=None):
    """Mean and SEM for a numeric key, optionally filtered by label."""
    vals = []
    for r in rows:
        if label_filter and r["label"] != label_filter:
            continue
        v = r.get(key)
        if v is not None and v != "":
            try:
                vals.append(float(v))
            except (ValueError, TypeError):
                pass
    if not vals:
        return None, None, 0
    n    = len(vals)
    mean = sum(vals) / n
    sem  = math.sqrt(sum((x - mean) ** 2 for x in vals) / (n * (n - 1))) if n > 1 else 0
    return round(mean, 3), round(sem, 3), n


def print_summary_table(rows):
    metrics = [
        ("corr_avg",             "Corr. avg"),
        ("keypoint_count",       "Keypoint count"),
        ("avg_strength_score",   "Avg keypoint strength"),
        ("embodied_total",       "Embodied phrase count"),
        ("embodied_phrase_rate", "Embodied phrase rate (per 1000w)"),
        ("visual_reference",     "  visual_reference"),
        ("action_narration",     "  action_narration"),
        ("sensory_description",  "  sensory_description"),
        ("procedural_instruction","  procedural_instruction"),
        ("words_per_min",        "Words per min"),
    ]
    print(f"\n{'Metric':<40} {'Conventional':>20} {'Embodied':>20}")
    print("-" * 82)
    for key, label in metrics:
        cm, cs, cn = group_stats(rows, key, "conventional")
        em, es, en = group_stats(rows, key, "embodied")
        conv_str = f"{cm} ± {cs} (n={cn})" if cm is not None else "—"
        emb_str  = f"{em} ± {es} (n={en})" if em is not None else "—"
        print(f"{label:<40} {conv_str:>20} {emb_str:>20}")


# ── Position distribution ─────────────────────────────────────────────────────

def compute_position_distribution(rows, label_filter=None):
    """
    For each video in the group, bin embodied phrase positions into N_BINS equal segments,
    normalize by total phrases in that video, then average across all videos.
    Returns list of N_BINS floats (average normalized density per bin).
    """
    bin_vectors = []
    for r in rows:
        if label_filter and r["label"] != label_filter:
            continue
        phrases = r["embodied_phrases_raw"]
        if not phrases:
            continue

        bins = [0] * N_BINS
        for p in phrases:
            pct = p.get("position_pct")
            if pct is None:
                continue
            try:
                idx = min(int(float(pct) / (100.0 / N_BINS)), N_BINS - 1)
                bins[idx] += 1
            except (ValueError, TypeError):
                pass

        total = sum(bins)
        if total > 0:
            bin_vectors.append([b / total for b in bins])

    if not bin_vectors:
        return [0.0] * N_BINS

    # Average across videos
    return [sum(v[i] for v in bin_vectors) / len(bin_vectors) for i in range(N_BINS)]


# ── Plotting ──────────────────────────────────────────────────────────────────

def make_plots(rows):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not available — skipping plots")
        return

    os.makedirs(PLOTS_DIR, exist_ok=True)

    conv = [r for r in rows if r["label"] == "conventional"]
    emb  = [r for r in rows if r["label"] == "embodied"]

    colors = {"conventional": "#2196F3", "embodied": "#FF5722"}

    # ── Plot 1: Embodied phrase rate vs corr_avg ──────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5))
    for label, group, color in [("Conventional", conv, colors["conventional"]),
                                  ("Embodied",     emb,  colors["embodied"])]:
        xs = [r["embodied_phrase_rate"] for r in group if r["corr_avg"] is not None]
        ys = [r["corr_avg"]             for r in group if r["corr_avg"] is not None]
        ax.scatter(xs, ys, c=color, label=label, alpha=0.7, edgecolors="white", s=60)

    ax.set_xlabel("Embodied Phrase Rate (per 1000 words)")
    ax.set_ylabel("Correlation Score (corr_avg)")
    ax.set_title("Embodied Phrase Rate vs. Correlation Score")
    ax.legend()
    ax.grid(True, alpha=0.3)
    path = os.path.join(PLOTS_DIR, "scatter_phrase_rate_vs_corr.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")

    # ── Plot 2: Keypoint avg strength vs corr_avg ─────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5))
    for label, group, color in [("Conventional", conv, colors["conventional"]),
                                  ("Embodied",     emb,  colors["embodied"])]:
        xs = [r["avg_strength_score"] for r in group if r["corr_avg"] is not None and r["avg_strength_score"] > 0]
        ys = [r["corr_avg"]           for r in group if r["corr_avg"] is not None and r["avg_strength_score"] > 0]
        ax.scatter(xs, ys, c=color, label=label, alpha=0.7, edgecolors="white", s=60)

    ax.set_xlabel("Avg Keypoint Strength Score (1–5)")
    ax.set_ylabel("Correlation Score (corr_avg)")
    ax.set_title("Keypoint Strength vs. Correlation Score")
    ax.legend()
    ax.grid(True, alpha=0.3)
    path = os.path.join(PLOTS_DIR, "scatter_keypoint_strength_vs_corr.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")

    # ── Plot 3: Position distribution (embodied phrase density across transcript)
    fig, ax = plt.subplots(figsize=(8, 4))
    bin_labels = [f"{i*10}–{i*10+10}%" for i in range(N_BINS)]
    x = list(range(N_BINS))
    width = 0.35

    dist_conv = compute_position_distribution(rows, "conventional")
    dist_emb  = compute_position_distribution(rows, "embodied")

    ax.bar([i - width/2 for i in x], dist_conv, width, label="Conventional",
           color=colors["conventional"], alpha=0.8)
    ax.bar([i + width/2 for i in x], dist_emb,  width, label="Embodied",
           color=colors["embodied"],     alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(bin_labels, rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("Position in Transcript")
    ax.set_ylabel("Avg Normalized Phrase Density")
    ax.set_title("Embodied Phrase Position Distribution\n(avg across all videos per group)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    path = os.path.join(PLOTS_DIR, "position_distribution.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")

    # ── Plot 4: Category breakdown (stacked bar) ──────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 5))
    cat_labels  = ["Visual\nReference", "Action\nNarration", "Sensory\nDesc.", "Procedural\nInstruct."]
    cat_keys    = ["visual_reference", "action_narration", "sensory_description", "procedural_instruction"]
    cat_colors  = ["#4CAF50", "#FF9800", "#9C27B0", "#F44336"]

    def mean_cat(group, key):
        vals = [r[key] for r in group if isinstance(r[key], (int, float))]
        return sum(vals) / len(vals) if vals else 0

    conv_means = [mean_cat(conv, k) for k in cat_keys]
    emb_means  = [mean_cat(emb,  k) for k in cat_keys]

    xpos   = list(range(len(cat_keys)))
    width  = 0.35
    bars_c = ax.bar([i - width/2 for i in xpos], conv_means, width,
                    label="Conventional", color=colors["conventional"], alpha=0.85)
    bars_e = ax.bar([i + width/2 for i in xpos], emb_means,  width,
                    label="Embodied",     color=colors["embodied"],     alpha=0.85)

    ax.set_xticks(xpos)
    ax.set_xticklabels(cat_labels, fontsize=9)
    ax.set_ylabel("Avg Count per Video")
    ax.set_title("Embodied Phrase Categories\n(avg per video)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    path = os.path.join(PLOTS_DIR, "category_breakdown.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")

    # ── Plot 5: Words per minute vs corr_avg ─────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5))
    for label, group, color in [("Conventional", conv, colors["conventional"]),
                                  ("Embodied",     emb,  colors["embodied"])]:
        xs, ys = [], []
        for r in group:
            if r["corr_avg"] is None or r["hallucination_flag"] == "1":
                continue
            try:
                wpm = float(r["words_per_min"])
                if wpm > 0:
                    xs.append(wpm)
                    ys.append(r["corr_avg"])
            except (ValueError, TypeError):
                pass
        ax.scatter(xs, ys, c=color, label=label, alpha=0.7, edgecolors="white", s=60)

    ax.set_xlabel("Words per Minute (speech rate)")
    ax.set_ylabel("Correlation Score (corr_avg)")
    ax.set_title("Speech Rate vs. Correlation Score\n(hallucinated videos excluded)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    path = os.path.join(PLOTS_DIR, "scatter_wpm_vs_corr.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")

    print(f"\nAll plots saved to: {PLOTS_DIR}")


# ── CSV export ────────────────────────────────────────────────────────────────

def export_master_csv(rows):
    path = os.path.join(GEMINI_DIR, "master_analysis.csv")
    export_keys = [
        "video_id", "label", "corr_avg", "transcript_words", "words_per_min",
        "speech_ratio", "hallucination_flag",
        "keypoint_count", "avg_strength_score", "max_strength_score",
        "embodied_total", "embodied_phrase_rate",
        "visual_reference", "action_narration", "sensory_description", "procedural_instruction",
        "verbal_richness", "parse_success",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=export_keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote master CSV → {path}  ({len(rows)} rows)")
    return path


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading data...")
    corr_map   = load_corr_map()
    speech_map = load_speech_map()
    records    = load_gemini_results()

    print(f"  Gemini results loaded: {len(records)} videos "
          f"({sum(1 for r in records if r['label']=='conventional')} conventional, "
          f"{sum(1 for r in records if r['label']=='embodied')} embodied)")
    print(f"  Parse failures: {sum(1 for r in records if not r['parse_success'])}")

    rows = build_master_table(records, corr_map, speech_map)

    # Print summary statistics
    print_summary_table(rows)

    # Export master CSV
    export_master_csv(rows)

    # Generate plots
    make_plots(rows)


if __name__ == "__main__":
    main()
