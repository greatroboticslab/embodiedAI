#!/usr/bin/env python3
"""
analyze_results.py

Loads Gemini analysis results and speech_ratio data, produces:
  1. Summary statistics tables:
     - Three-way: conventional vs verbally_embodied vs visually_embodied
     - Filtered:  conventional vs verbally_embodied only
  2. Scatter plots with three-way coloring (and filtered versions)
  3. Position distribution + category breakdown (three-way and filtered)

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

# ── Config ────────────────────────────────────────────────────────────────────
WPM_THRESHOLD = 60  # embodied videos below this are "visually_embodied"

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
            with open(os.path.join(set_dir, fn), encoding="utf-8") as f:
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


def assign_subgroup(label, wpm_str):
    """Assign subgroup based on label and words_per_min."""
    if label == "conventional":
        return "conventional"
    try:
        wpm = float(wpm_str)
    except (ValueError, TypeError):
        wpm = 0.0
    if wpm < WPM_THRESHOLD:
        return "visually_embodied"
    return "verbally_embodied"


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
        wpm_str = speech_map.get(vid, {}).get("words_per_min", "")
        rows.append({
            "video_id":               vid,
            "label":                  r["label"],
            "subgroup":               assign_subgroup(r["label"], wpm_str),
            "corr_avg":               corr_map.get(vid),
            "transcript_words":       words,
            "words_per_min":          wpm_str,
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

def group_stats(rows, key, filter_key="label", filter_val=None):
    """Mean and SEM for a numeric key, optionally filtered."""
    vals = []
    for r in rows:
        if filter_val and r.get(filter_key) != filter_val:
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


def print_summary_table(rows, title, groups, filter_key="subgroup"):
    """Print a summary table for the given groups."""
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
        ("transcript_words",     "Transcript words"),
        ("words_per_min",        "Words per min"),
    ]

    # Header
    col_width = 24
    header = f"\n{'':=<80}\n{title}\n{'':=<80}\n"
    header += f"{'Metric':<40}"
    for g_label, _ in groups:
        header += f" {g_label:>{col_width}}"
    print(header)
    print("-" * (40 + col_width * len(groups) + len(groups)))

    for key, label in metrics:
        line = f"{label:<40}"
        for _, g_val in groups:
            m, s, n = group_stats(rows, key, filter_key, g_val)
            cell = f"{m} ± {s} (n={n})" if m is not None else "—"
            line += f" {cell:>{col_width}}"
        print(line)
    print()


# ── Position distribution ─────────────────────────────────────────────────────

def compute_position_distribution(rows, filter_key="subgroup", filter_val=None):
    """
    For each video in the group, bin embodied phrase positions into N_BINS equal segments,
    normalize by total phrases in that video, then average across all videos.
    Returns list of N_BINS floats (average normalized density per bin).
    """
    bin_vectors = []
    for r in rows:
        if filter_val and r.get(filter_key) != filter_val:
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

    # Group definitions for three-way and filtered
    conv   = [r for r in rows if r["subgroup"] == "conventional"]
    v_emb  = [r for r in rows if r["subgroup"] == "verbally_embodied"]
    vi_emb = [r for r in rows if r["subgroup"] == "visually_embodied"]

    colors_3 = {
        "conventional":      "#2196F3",
        "verbally_embodied": "#4CAF50",
        "visually_embodied": "#FF5722",
    }

    three_groups = [
        ("Conventional",      conv,   colors_3["conventional"]),
        ("Verbally Embodied", v_emb,  colors_3["verbally_embodied"]),
        ("Visually Embodied", vi_emb, colors_3["visually_embodied"]),
    ]

    filtered_groups = [
        ("Conventional",      conv,  colors_3["conventional"]),
        ("Verbally Embodied", v_emb, colors_3["verbally_embodied"]),
    ]

    def _scatter(groups, x_key, y_key, xlabel, ylabel, title, filename,
                 skip_zero_x=False, skip_halluc=False):
        fig, ax = plt.subplots(figsize=(7, 5))
        for label, group, color in groups:
            xs, ys = [], []
            for r in group:
                if r["corr_avg"] is None:
                    continue
                if skip_halluc and r["hallucination_flag"] == "1":
                    continue
                try:
                    xv = float(r[x_key])
                except (ValueError, TypeError):
                    continue
                if skip_zero_x and xv <= 0:
                    continue
                xs.append(xv)
                ys.append(r["corr_avg"])
            ax.scatter(xs, ys, c=color, label=label, alpha=0.7, edgecolors="white", s=60)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        path = os.path.join(PLOTS_DIR, filename)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {path}")

    # ── THREE-WAY scatter plots ───────────────────────────────────────────────

    _scatter(three_groups,
             "embodied_phrase_rate", "corr_avg",
             "Embodied Phrase Rate (per 1000 words)", "Correlation Score (corr_avg)",
             "Embodied Phrase Rate vs. Correlation Score\n(three-way split)",
             "scatter_phrase_rate_vs_corr_3way.png")

    _scatter(three_groups,
             "avg_strength_score", "corr_avg",
             "Avg Keypoint Strength Score (1–5)", "Correlation Score (corr_avg)",
             "Keypoint Strength vs. Correlation Score\n(three-way split)",
             "scatter_keypoint_strength_vs_corr_3way.png",
             skip_zero_x=True)

    _scatter(three_groups,
             "words_per_min", "corr_avg",
             "Words per Minute (speech rate)", "Correlation Score (corr_avg)",
             "Speech Rate vs. Correlation Score\n(three-way split, all videos)",
             "scatter_wpm_vs_corr_3way.png",
             skip_zero_x=True)

    # ── FILTERED scatter plots (conventional vs verbally_embodied) ────────────

    _scatter(filtered_groups,
             "embodied_phrase_rate", "corr_avg",
             "Embodied Phrase Rate (per 1000 words)", "Correlation Score (corr_avg)",
             "Embodied Phrase Rate vs. Correlation Score\n(conventional vs. verbally embodied)",
             "scatter_phrase_rate_vs_corr_filtered.png")

    _scatter(filtered_groups,
             "avg_strength_score", "corr_avg",
             "Avg Keypoint Strength Score (1–5)", "Correlation Score (corr_avg)",
             "Keypoint Strength vs. Correlation Score\n(conventional vs. verbally embodied)",
             "scatter_keypoint_strength_vs_corr_filtered.png",
             skip_zero_x=True)

    _scatter(filtered_groups,
             "words_per_min", "corr_avg",
             "Words per Minute (speech rate)", "Correlation Score (corr_avg)",
             "Speech Rate vs. Correlation Score\n(conventional vs. verbally embodied)",
             "scatter_wpm_vs_corr_filtered.png",
             skip_zero_x=True)

    # ── Position distribution (three-way) ─────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 4))
    bin_labels = [f"{i*10}–{i*10+10}%" for i in range(N_BINS)]
    x = list(range(N_BINS))
    width = 0.25

    dist_conv  = compute_position_distribution(rows, "subgroup", "conventional")
    dist_vemb  = compute_position_distribution(rows, "subgroup", "verbally_embodied")
    dist_viemb = compute_position_distribution(rows, "subgroup", "visually_embodied")

    ax.bar([i - width for i in x], dist_conv,  width, label="Conventional",
           color=colors_3["conventional"], alpha=0.8)
    ax.bar(x,                       dist_vemb, width, label="Verbally Embodied",
           color=colors_3["verbally_embodied"], alpha=0.8)
    ax.bar([i + width for i in x], dist_viemb, width, label="Visually Embodied",
           color=colors_3["visually_embodied"], alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(bin_labels, rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("Position in Transcript")
    ax.set_ylabel("Avg Normalized Phrase Density")
    ax.set_title("Embodied Phrase Position Distribution\n(three-way split)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    path = os.path.join(PLOTS_DIR, "position_distribution_3way.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")

    # ── Category breakdown (three-way) ────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5))
    cat_labels = ["Visual\nReference", "Action\nNarration", "Sensory\nDesc.", "Procedural\nInstruct."]
    cat_keys   = CATEGORIES

    def mean_cat(group, key):
        vals = [r[key] for r in group if isinstance(r[key], (int, float))]
        return sum(vals) / len(vals) if vals else 0

    conv_means  = [mean_cat(conv,   k) for k in cat_keys]
    vemb_means  = [mean_cat(v_emb,  k) for k in cat_keys]
    viemb_means = [mean_cat(vi_emb, k) for k in cat_keys]

    xpos  = list(range(len(cat_keys)))
    width = 0.25
    ax.bar([i - width for i in xpos], conv_means,  width,
           label="Conventional",      color=colors_3["conventional"], alpha=0.85)
    ax.bar(xpos,                       vemb_means,  width,
           label="Verbally Embodied",  color=colors_3["verbally_embodied"], alpha=0.85)
    ax.bar([i + width for i in xpos], viemb_means, width,
           label="Visually Embodied",  color=colors_3["visually_embodied"], alpha=0.85)

    ax.set_xticks(xpos)
    ax.set_xticklabels(cat_labels, fontsize=9)
    ax.set_ylabel("Avg Count per Video")
    ax.set_title("Embodied Phrase Categories\n(three-way split)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    path = os.path.join(PLOTS_DIR, "category_breakdown_3way.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")

    # ── Category breakdown (filtered: conv vs verbally embodied) ──────────────
    fig, ax = plt.subplots(figsize=(6, 5))
    width = 0.35
    ax.bar([i - width/2 for i in xpos], conv_means, width,
           label="Conventional",      color=colors_3["conventional"], alpha=0.85)
    ax.bar([i + width/2 for i in xpos], vemb_means, width,
           label="Verbally Embodied",  color=colors_3["verbally_embodied"], alpha=0.85)

    ax.set_xticks(xpos)
    ax.set_xticklabels(cat_labels, fontsize=9)
    ax.set_ylabel("Avg Count per Video")
    ax.set_title("Embodied Phrase Categories\n(conventional vs. verbally embodied)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    path = os.path.join(PLOTS_DIR, "category_breakdown_filtered.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")

    # ── WPM histogram showing bimodal split ───────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    emb_wpm = []
    for r in rows:
        if r["label"] != "embodied":
            continue
        try:
            w = float(r["words_per_min"])
            if w > 0:
                emb_wpm.append(w)
        except (ValueError, TypeError):
            pass

    ax.hist(emb_wpm, bins=30, color="#FF9800", edgecolor="white", alpha=0.85)
    ax.axvline(WPM_THRESHOLD, color="red", linestyle="--", linewidth=2,
               label=f"Threshold = {WPM_THRESHOLD} wpm")
    ax.set_xlabel("Words per Minute")
    ax.set_ylabel("Number of Videos")
    ax.set_title("Bimodal Distribution of Speech Rate in Embodied Videos")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    path = os.path.join(PLOTS_DIR, "wpm_bimodal_histogram.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")

    print(f"\nAll plots saved to: {PLOTS_DIR}")


# ── CSV export ────────────────────────────────────────────────────────────────

def export_position_distribution_csv(rows):
    """Export binned position distribution data as CSV with three-way split."""
    path = os.path.join(GEMINI_DIR, "position_distribution.csv")

    def get_bin_vectors(filter_key, filter_val):
        vectors = []
        for r in rows:
            if r.get(filter_key) != filter_val:
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
                vectors.append([b / total for b in bins])
        return vectors

    conv_vecs  = get_bin_vectors("subgroup", "conventional")
    vemb_vecs  = get_bin_vectors("subgroup", "verbally_embodied")
    viemb_vecs = get_bin_vectors("subgroup", "visually_embodied")

    def avg_dist(vecs):
        if not vecs:
            return [0.0] * N_BINS
        return [sum(v[i] for v in vecs) / len(vecs) for i in range(N_BINS)]

    conv_dist  = avg_dist(conv_vecs)
    vemb_dist  = avg_dist(vemb_vecs)
    viemb_dist = avg_dist(viemb_vecs)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["bin_index", "bin_label",
                         "conventional_density", "verbally_embodied_density",
                         "visually_embodied_density",
                         "conventional_n", "verbally_embodied_n", "visually_embodied_n"])
        for i in range(N_BINS):
            writer.writerow([
                i,
                f"{i*10}-{i*10+10}%",
                round(conv_dist[i], 4),
                round(vemb_dist[i], 4),
                round(viemb_dist[i], 4),
                len(conv_vecs),
                len(vemb_vecs),
                len(viemb_vecs),
            ])
    print(f"Wrote position distribution → {path}")
    return path


def export_master_csv(rows):
    path = os.path.join(GEMINI_DIR, "master_analysis.csv")
    export_keys = [
        "video_id", "label", "subgroup", "corr_avg", "transcript_words", "words_per_min",
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

    # Subgroup counts
    sg_counts = defaultdict(int)
    for r in rows:
        sg_counts[r["subgroup"]] += 1
    print(f"\n  Subgroup split (wpm threshold = {WPM_THRESHOLD}):")
    for sg in ["conventional", "verbally_embodied", "visually_embodied"]:
        print(f"    {sg}: {sg_counts[sg]}")

    # === THREE-WAY summary table ===
    print_summary_table(rows, "THREE-WAY COMPARISON", [
        ("Conventional",      "conventional"),
        ("Verbally Embodied", "verbally_embodied"),
        ("Visually Embodied", "visually_embodied"),
    ])

    # === FILTERED summary table (conv vs verbally embodied) ===
    filtered = [r for r in rows if r["subgroup"] != "visually_embodied"]
    print_summary_table(filtered, "FILTERED: CONVENTIONAL vs. VERBALLY EMBODIED", [
        ("Conventional",      "conventional"),
        ("Verbally Embodied", "verbally_embodied"),
    ])

    # Export master CSV and position distribution
    export_master_csv(rows)
    export_position_distribution_csv(rows)

    # Generate plots
    make_plots(rows)


if __name__ == "__main__":
    main()
