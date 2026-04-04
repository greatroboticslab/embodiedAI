#!/usr/bin/env python3
"""
analyze_classification.py

Joins Gemini video classification data with fusion engagement scores to analyze
which content types drive viewer engagement across conventional, verbally embodied,
and visually embodied videos.

Analyses:
  1. Engagement by content type (across all videos and by subgroup)
  2. Alignment score vs engagement correlation
  3. Narration/hands-visible vs engagement
  4. Statistical tests (ANOVA, t-tests, correlation)
  5. Summary tables and CSV output for paper figures

Usage:
  cd FusionAnalysis/scripts/
  python analyze_classification.py
  python analyze_classification.py --output_dir ../results/classification_analysis
"""

import os
import json
import glob
import argparse
import csv
import math
from collections import Counter, defaultdict
from typing import Dict, List, Any, Tuple

import numpy as np

try:
    from scipy import stats as sp_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("[WARN] scipy not installed — statistical tests will be skipped. Install with: pip install scipy")

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    print("[WARN] pandas not installed — Excel output will be skipped.")

# ── Paths ─────────────────────────────────────────────────────────────────────
HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))

CLASSIFICATION_DIR = os.path.join(HERE, "..", "results", "video_classification")
CORRELATION_CONV = os.path.join(HERE, "..", "results", "segment_correlations_conventional")
CORRELATION_EMB = os.path.join(HERE, "..", "results", "segment_correlations_embodied")
MASTER_CSV = os.path.join(PROJECT_ROOT, "TranscriptAnalysis", "results", "gemini_analysis", "master_analysis.csv")

CONTENT_TYPES = [
    "hands_on_demonstration",
    "equipment_closeup",
    "presenter_talking",
    "slide_or_powerpoint",
    "diagram_or_whiteboard",
    "screen_or_software",
    "animation_or_graphic",
    "device_in_operation",
    "other",
]


# ── Data Loading ──────────────────────────────────────────────────────────────

def load_subgroup_map() -> Dict[str, str]:
    """Load video_id → subgroup mapping from master_analysis.csv."""
    subgroup_map = {}
    if not os.path.exists(MASTER_CSV):
        print(f"[WARN] Master CSV not found: {MASTER_CSV}")
        return subgroup_map
    with open(MASTER_CSV, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            subgroup_map[row["video_id"]] = row["subgroup"]
    return subgroup_map


def load_classification(label: str) -> Dict[str, List[Dict]]:
    """Load classification JSONs. Returns {video_id: [segments]}."""
    cls_dir = os.path.join(CLASSIFICATION_DIR, label)
    result = {}
    for path in glob.glob(os.path.join(cls_dir, "*.json")):
        with open(path) as f:
            data = json.load(f)
        if not data.get("complete"):
            continue
        result[data["video_id"]] = data["segments"]
    return result


def load_engagement(corr_dir: str) -> Dict[str, List[Dict]]:
    """Load fusion engagement JSONs. Returns {video_id: [segments]}."""
    result = {}
    for path in glob.glob(os.path.join(corr_dir, "*_segment_correlation.json")):
        with open(path) as f:
            data = json.load(f)
        vid = data.get("video_id", "")
        result[vid] = data.get("segments", [])
    return result


def compute_segment_engagement(eng_seg: Dict) -> Dict[str, float]:
    """Compute engagement metrics for a single fusion segment."""
    vis_list = eng_seg.get("visual_correlations", [])
    trans_list = eng_seg.get("transcript_correlations", [])

    def avg_score(corr_list):
        scores = []
        for c in corr_list:
            if not isinstance(c, dict):
                continue
            s = c.get("score", 0)
            if isinstance(s, list):
                s = s[0] if s else 0
            try:
                scores.append(float(s))
            except (ValueError, TypeError):
                pass
        return np.mean(scores) if scores else 0.0

    def pct_correlated(corr_list):
        if not corr_list:
            return 0.0
        corr_count = sum(1 for c in corr_list if isinstance(c, dict) and c.get("correlated", False))
        return corr_count / len(corr_list) * 100

    return {
        "visual_avg_score": avg_score(vis_list),
        "visual_pct_correlated": pct_correlated(vis_list),
        "transcript_avg_score": avg_score(trans_list),
        "transcript_pct_correlated": pct_correlated(trans_list),
        "n_visual": len(vis_list),
        "n_transcript": len(trans_list),
    }


# ── Join & Build Dataset ─────────────────────────────────────────────────────

def build_joined_dataset(subgroup_map: Dict[str, str]) -> List[Dict]:
    """Join classification + engagement data into one flat list of segment records."""
    # Load all data
    cls_conv = load_classification("conventional")
    cls_emb = load_classification("embodied")
    eng_conv = load_engagement(CORRELATION_CONV)
    eng_emb = load_engagement(CORRELATION_EMB)

    print(f"Loaded classification: {len(cls_conv)} conventional, {len(cls_emb)} embodied")
    print(f"Loaded engagement: {len(eng_conv)} conventional, {len(eng_emb)} embodied")

    records = []
    matched = 0
    unmatched = 0

    for label, cls_data, eng_data in [
        ("conventional", cls_conv, eng_conv),
        ("embodied", cls_emb, eng_emb),
    ]:
        for video_id, cls_segments in cls_data.items():
            eng_segments = eng_data.get(video_id, [])
            subgroup = subgroup_map.get(video_id, label)

            # Index engagement segments by segment_index
            eng_by_idx = {}
            for seg in eng_segments:
                idx = seg.get("segment_index")
                if idx is not None:
                    eng_by_idx[idx] = seg

            for cls_seg in cls_segments:
                idx = cls_seg["segment_index"]
                eng_seg = eng_by_idx.get(idx)

                if eng_seg is None:
                    unmatched += 1
                    continue

                engagement = compute_segment_engagement(eng_seg)

                # Skip segments with no engagement data (no comments evaluated)
                if engagement["n_visual"] == 0 and engagement["n_transcript"] == 0:
                    continue

                record = {
                    "video_id": video_id,
                    "subgroup": subgroup,
                    "segment_index": idx,
                    "start": cls_seg["start"],
                    "end": cls_seg["end"],
                    "content_type": cls_seg["content_type"],
                    "alignment_score": cls_seg["alignment_score"],
                    "narration": cls_seg["narration"],
                    "hands_visible": cls_seg["hands_visible"],
                    "instructional_density": cls_seg["instructional_density"],
                    **engagement,
                }
                records.append(record)
                matched += 1

    print(f"Joined: {matched} segments matched, {unmatched} unmatched")
    return records


# ── Analysis Functions ────────────────────────────────────────────────────────

def analyze_engagement_by_content_type(records: List[Dict], output_dir: str):
    """Analysis 1: Mean engagement metrics by content type, overall and by subgroup."""
    print("\n" + "=" * 70)
    print("ANALYSIS 1: Engagement by Content Type")
    print("=" * 70)

    # Overall
    by_ct = defaultdict(list)
    for r in records:
        by_ct[r["content_type"]].append(r)

    rows = []
    print(f"\n{'Content Type':<28} {'N':>6} {'Vis%':>7} {'VisSc':>7} {'Trn%':>7} {'TrnSc':>7}")
    print("-" * 70)
    for ct in CONTENT_TYPES:
        segs = by_ct.get(ct, [])
        if not segs:
            continue
        vis_pct = np.mean([s["visual_pct_correlated"] for s in segs])
        vis_sc = np.mean([s["visual_avg_score"] for s in segs])
        trn_pct = np.mean([s["transcript_pct_correlated"] for s in segs])
        trn_sc = np.mean([s["transcript_avg_score"] for s in segs])
        print(f"{ct:<28} {len(segs):>6} {vis_pct:>6.1f}% {vis_sc:>6.1f} {trn_pct:>6.1f}% {trn_sc:>6.1f}")
        rows.append({
            "content_type": ct, "n_segments": len(segs),
            "visual_corr_pct": round(vis_pct, 2), "visual_avg_score": round(vis_sc, 2),
            "transcript_corr_pct": round(trn_pct, 2), "transcript_avg_score": round(trn_sc, 2),
        })

    # Save CSV
    csv_path = os.path.join(output_dir, "engagement_by_content_type.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved: {csv_path}")

    # By subgroup
    for subgroup in ["conventional", "verbally_embodied", "visually_embodied"]:
        sub_records = [r for r in records if r["subgroup"] == subgroup]
        if not sub_records:
            continue
        by_ct_sub = defaultdict(list)
        for r in sub_records:
            by_ct_sub[r["content_type"]].append(r)

        rows_sub = []
        print(f"\n--- {subgroup.upper()} ({len(sub_records)} segments) ---")
        print(f"{'Content Type':<28} {'N':>6} {'Vis%':>7} {'VisSc':>7} {'Trn%':>7} {'TrnSc':>7}")
        print("-" * 70)
        for ct in CONTENT_TYPES:
            segs = by_ct_sub.get(ct, [])
            if not segs:
                continue
            vis_pct = np.mean([s["visual_pct_correlated"] for s in segs])
            vis_sc = np.mean([s["visual_avg_score"] for s in segs])
            trn_pct = np.mean([s["transcript_pct_correlated"] for s in segs])
            trn_sc = np.mean([s["transcript_avg_score"] for s in segs])
            print(f"{ct:<28} {len(segs):>6} {vis_pct:>6.1f}% {vis_sc:>6.1f} {trn_pct:>6.1f}% {trn_sc:>6.1f}")
            rows_sub.append({
                "subgroup": subgroup, "content_type": ct, "n_segments": len(segs),
                "visual_corr_pct": round(vis_pct, 2), "visual_avg_score": round(vis_sc, 2),
                "transcript_corr_pct": round(trn_pct, 2), "transcript_avg_score": round(trn_sc, 2),
            })

        csv_path = os.path.join(output_dir, f"engagement_by_content_type_{subgroup}.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows_sub[0].keys())
            writer.writeheader()
            writer.writerows(rows_sub)
        print(f"Saved: {csv_path}")


def analyze_content_distribution(records: List[Dict], output_dir: str):
    """Analysis 2: Content type distribution by subgroup."""
    print("\n" + "=" * 70)
    print("ANALYSIS 2: Content Type Distribution by Subgroup")
    print("=" * 70)

    rows = []
    for subgroup in ["conventional", "verbally_embodied", "visually_embodied"]:
        sub_records = [r for r in records if r["subgroup"] == subgroup]
        ct_counts = Counter(r["content_type"] for r in sub_records)
        total = len(sub_records)

        print(f"\n--- {subgroup.upper()} ({total} segments) ---")
        for ct in CONTENT_TYPES:
            count = ct_counts.get(ct, 0)
            pct = count / total * 100 if total else 0
            print(f"  {ct:<28} {count:>5} ({pct:>5.1f}%)")
            rows.append({
                "subgroup": subgroup, "content_type": ct,
                "count": count, "percentage": round(pct, 2),
            })

    csv_path = os.path.join(output_dir, "content_distribution_by_subgroup.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["subgroup", "content_type", "count", "percentage"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved: {csv_path}")


def analyze_alignment_vs_engagement(records: List[Dict], output_dir: str):
    """Analysis 3: Audio-visual alignment score vs engagement."""
    print("\n" + "=" * 70)
    print("ANALYSIS 3: Alignment Score vs Engagement")
    print("=" * 70)

    # Bin alignment into ranges
    bins = [(0, 20), (20, 40), (40, 60), (60, 80), (80, 100)]
    rows = []

    print(f"\n{'Alignment Range':<18} {'N':>6} {'Vis%':>7} {'VisSc':>7} {'Trn%':>7} {'TrnSc':>7}")
    print("-" * 60)
    for lo, hi in bins:
        segs = [r for r in records if lo <= r["alignment_score"] < hi or (hi == 100 and r["alignment_score"] == 100)]
        if not segs:
            continue
        vis_pct = np.mean([s["visual_pct_correlated"] for s in segs])
        vis_sc = np.mean([s["visual_avg_score"] for s in segs])
        trn_pct = np.mean([s["transcript_pct_correlated"] for s in segs])
        trn_sc = np.mean([s["transcript_avg_score"] for s in segs])
        label = f"{lo}-{hi}"
        print(f"{label:<18} {len(segs):>6} {vis_pct:>6.1f}% {vis_sc:>6.1f} {trn_pct:>6.1f}% {trn_sc:>6.1f}")
        rows.append({
            "alignment_range": label, "n_segments": len(segs),
            "visual_corr_pct": round(vis_pct, 2), "visual_avg_score": round(vis_sc, 2),
            "transcript_corr_pct": round(trn_pct, 2), "transcript_avg_score": round(trn_sc, 2),
        })

    # Pearson correlation
    if HAS_SCIPY:
        alignment = [r["alignment_score"] for r in records]
        vis_scores = [r["visual_avg_score"] for r in records]
        trn_scores = [r["transcript_avg_score"] for r in records]
        vis_pcts = [r["visual_pct_correlated"] for r in records]
        trn_pcts = [r["transcript_pct_correlated"] for r in records]

        r_vis, p_vis = sp_stats.pearsonr(alignment, vis_scores)
        r_trn, p_trn = sp_stats.pearsonr(alignment, trn_scores)
        r_vis_pct, p_vis_pct = sp_stats.pearsonr(alignment, vis_pcts)
        r_trn_pct, p_trn_pct = sp_stats.pearsonr(alignment, trn_pcts)
        print(f"\nPearson correlations:")
        print(f"  Alignment vs Visual Avg Score:  r={r_vis:.3f}, p={p_vis:.2e}")
        print(f"  Alignment vs Transcript Avg Score: r={r_trn:.3f}, p={p_trn:.2e}")
        print(f"  Alignment vs Visual Corr %:     r={r_vis_pct:.3f}, p={p_vis_pct:.2e}")
        print(f"  Alignment vs Transcript Corr %: r={r_trn_pct:.3f}, p={p_trn_pct:.2e}")

        rows.append({"alignment_range": "pearson_vis_score", "n_segments": len(records),
                      "visual_corr_pct": round(r_vis_pct, 4), "visual_avg_score": round(r_vis, 4),
                      "transcript_corr_pct": round(r_trn_pct, 4), "transcript_avg_score": round(r_trn, 4)})

    csv_path = os.path.join(output_dir, "alignment_vs_engagement.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {csv_path}")


def analyze_narration_hands(records: List[Dict], output_dir: str):
    """Analysis 4: Engagement by narration and hands-visible flags."""
    print("\n" + "=" * 70)
    print("ANALYSIS 4: Narration & Hands Visible vs Engagement")
    print("=" * 70)

    rows = []
    for flag_name, flag_key in [("Narration", "narration"), ("Hands Visible", "hands_visible")]:
        for val in [True, False]:
            segs = [r for r in records if r[flag_key] == val]
            if not segs:
                continue
            vis_pct = np.mean([s["visual_pct_correlated"] for s in segs])
            vis_sc = np.mean([s["visual_avg_score"] for s in segs])
            trn_pct = np.mean([s["transcript_pct_correlated"] for s in segs])
            trn_sc = np.mean([s["transcript_avg_score"] for s in segs])
            label = f"{flag_name}={'Yes' if val else 'No'}"
            print(f"{label:<28} N={len(segs):>5}  Vis%={vis_pct:>5.1f}  VisSc={vis_sc:>5.1f}  "
                  f"Trn%={trn_pct:>5.1f}  TrnSc={trn_sc:>5.1f}")
            rows.append({
                "condition": label, "n_segments": len(segs),
                "visual_corr_pct": round(vis_pct, 2), "visual_avg_score": round(vis_sc, 2),
                "transcript_corr_pct": round(trn_pct, 2), "transcript_avg_score": round(trn_sc, 2),
            })

        # t-test
        if HAS_SCIPY:
            yes = [r["visual_avg_score"] for r in records if r[flag_key]]
            no = [r["visual_avg_score"] for r in records if not r[flag_key]]
            if yes and no:
                t, p = sp_stats.ttest_ind(yes, no, equal_var=False)
                print(f"  t-test (visual avg score): t={t:.3f}, p={p:.2e}")
            yes_t = [r["transcript_avg_score"] for r in records if r[flag_key]]
            no_t = [r["transcript_avg_score"] for r in records if not r[flag_key]]
            if yes_t and no_t:
                t, p = sp_stats.ttest_ind(yes_t, no_t, equal_var=False)
                print(f"  t-test (transcript avg score): t={t:.3f}, p={p:.2e}")
        print()

    csv_path = os.path.join(output_dir, "narration_hands_vs_engagement.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {csv_path}")


def analyze_statistical_tests(records: List[Dict], output_dir: str):
    """Analysis 5: ANOVA and pairwise comparisons."""
    if not HAS_SCIPY:
        print("\n[SKIP] Statistical tests require scipy")
        return

    print("\n" + "=" * 70)
    print("ANALYSIS 5: Statistical Tests")
    print("=" * 70)

    results = []

    # 5a. One-way ANOVA: visual engagement across content types
    print("\n--- ANOVA: Visual Avg Score by Content Type ---")
    groups = defaultdict(list)
    for r in records:
        groups[r["content_type"]].append(r["visual_avg_score"])

    # Only include groups with enough data
    valid_groups = {k: v for k, v in groups.items() if len(v) >= 10}
    group_names = sorted(valid_groups.keys())
    group_data = [valid_groups[k] for k in group_names]

    if len(group_data) >= 2:
        f_stat, p_val = sp_stats.f_oneway(*group_data)
        print(f"  F={f_stat:.3f}, p={p_val:.2e}")
        print(f"  Groups: {', '.join(f'{k}(n={len(valid_groups[k])})' for k in group_names)}")
        results.append({"test": "ANOVA_visual_by_content_type", "statistic": round(f_stat, 4), "p_value": f"{p_val:.2e}"})

    # 5b. ANOVA: transcript engagement across content types
    print("\n--- ANOVA: Transcript Avg Score by Content Type ---")
    groups_t = defaultdict(list)
    for r in records:
        groups_t[r["content_type"]].append(r["transcript_avg_score"])
    valid_t = {k: v for k, v in groups_t.items() if len(v) >= 10}
    group_data_t = [valid_t[k] for k in sorted(valid_t.keys())]
    if len(group_data_t) >= 2:
        f_stat, p_val = sp_stats.f_oneway(*group_data_t)
        print(f"  F={f_stat:.3f}, p={p_val:.2e}")
        results.append({"test": "ANOVA_transcript_by_content_type", "statistic": round(f_stat, 4), "p_value": f"{p_val:.2e}"})

    # 5c. Key pairwise t-tests
    print("\n--- Pairwise t-tests (visual avg score) ---")
    pairs = [
        ("hands_on_demonstration", "slide_or_powerpoint"),
        ("hands_on_demonstration", "presenter_talking"),
        ("hands_on_demonstration", "diagram_or_whiteboard"),
        ("equipment_closeup", "slide_or_powerpoint"),
        ("device_in_operation", "presenter_talking"),
    ]
    for ct1, ct2 in pairs:
        g1 = valid_groups.get(ct1, [])
        g2 = valid_groups.get(ct2, [])
        if g1 and g2:
            t, p = sp_stats.ttest_ind(g1, g2, equal_var=False)
            d = (np.mean(g1) - np.mean(g2)) / np.sqrt((np.std(g1)**2 + np.std(g2)**2) / 2)  # Cohen's d
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            print(f"  {ct1} vs {ct2}: t={t:.3f}, p={p:.2e}, d={d:.3f} {sig}")
            results.append({
                "test": f"ttest_{ct1}_vs_{ct2}",
                "statistic": round(t, 4),
                "p_value": f"{p:.2e}",
            })

    # 5d. ANOVA: engagement by subgroup
    print("\n--- ANOVA: Visual Avg Score by Subgroup ---")
    sub_groups = defaultdict(list)
    for r in records:
        sub_groups[r["subgroup"]].append(r["visual_avg_score"])
    sub_data = [sub_groups[k] for k in sorted(sub_groups.keys())]
    if len(sub_data) >= 2:
        f_stat, p_val = sp_stats.f_oneway(*sub_data)
        print(f"  F={f_stat:.3f}, p={p_val:.2e}")
        for k in sorted(sub_groups.keys()):
            print(f"    {k}: n={len(sub_groups[k])}, mean={np.mean(sub_groups[k]):.2f}, std={np.std(sub_groups[k]):.2f}")
        results.append({"test": "ANOVA_visual_by_subgroup", "statistic": round(f_stat, 4), "p_value": f"{p_val:.2e}"})

    csv_path = os.path.join(output_dir, "statistical_tests.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["test", "statistic", "p_value"])
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved: {csv_path}")


def analyze_top_engaged_content(records: List[Dict], output_dir: str):
    """Analysis 6: What content types dominate the top-engaged segments?"""
    print("\n" + "=" * 70)
    print("ANALYSIS 6: Content Types in Top vs Bottom Engaged Segments")
    print("=" * 70)

    rows = []
    for subgroup in ["conventional", "verbally_embodied", "visually_embodied"]:
        sub = [r for r in records if r["subgroup"] == subgroup]
        if len(sub) < 20:
            continue

        # Sort by visual engagement
        sub_sorted = sorted(sub, key=lambda x: x["visual_avg_score"], reverse=True)
        n = len(sub_sorted)
        top_25 = sub_sorted[:n // 4]
        bot_25 = sub_sorted[-(n // 4):]

        top_ct = Counter(r["content_type"] for r in top_25)
        bot_ct = Counter(r["content_type"] for r in bot_25)

        print(f"\n--- {subgroup.upper()} (top 25% vs bottom 25% by visual score) ---")
        print(f"{'Content Type':<28} {'Top%':>7} {'Bot%':>7} {'Diff':>7}")
        print("-" * 55)
        for ct in CONTENT_TYPES:
            t_pct = top_ct.get(ct, 0) / len(top_25) * 100 if top_25 else 0
            b_pct = bot_ct.get(ct, 0) / len(bot_25) * 100 if bot_25 else 0
            diff = t_pct - b_pct
            if t_pct > 0 or b_pct > 0:
                print(f"{ct:<28} {t_pct:>6.1f}% {b_pct:>6.1f}% {diff:>+6.1f}%")
                rows.append({
                    "subgroup": subgroup, "content_type": ct,
                    "top_25_pct": round(t_pct, 2), "bottom_25_pct": round(b_pct, 2),
                    "difference": round(diff, 2),
                })

    csv_path = os.path.join(output_dir, "top_vs_bottom_content_types.csv")
    if rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nSaved: {csv_path}")


def analyze_instructional_density(records: List[Dict], output_dir: str):
    """Analysis 7: Instructional density vs engagement."""
    print("\n" + "=" * 70)
    print("ANALYSIS 7: Instructional Density vs Engagement")
    print("=" * 70)

    rows = []
    print(f"\n{'Density':>8} {'N':>6} {'Vis%':>7} {'VisSc':>7} {'Trn%':>7} {'TrnSc':>7}")
    print("-" * 50)
    for d in range(1, 6):
        segs = [r for r in records if r["instructional_density"] == d]
        if not segs:
            continue
        vis_pct = np.mean([s["visual_pct_correlated"] for s in segs])
        vis_sc = np.mean([s["visual_avg_score"] for s in segs])
        trn_pct = np.mean([s["transcript_pct_correlated"] for s in segs])
        trn_sc = np.mean([s["transcript_avg_score"] for s in segs])
        print(f"{d:>8} {len(segs):>6} {vis_pct:>6.1f}% {vis_sc:>6.1f} {trn_pct:>6.1f}% {trn_sc:>6.1f}")
        rows.append({
            "instructional_density": d, "n_segments": len(segs),
            "visual_corr_pct": round(vis_pct, 2), "visual_avg_score": round(vis_sc, 2),
            "transcript_corr_pct": round(trn_pct, 2), "transcript_avg_score": round(trn_sc, 2),
        })

    if HAS_SCIPY:
        densities = [r["instructional_density"] for r in records]
        vis_scores = [r["visual_avg_score"] for r in records]
        trn_scores = [r["transcript_avg_score"] for r in records]
        r_vis, p_vis = sp_stats.pearsonr(densities, vis_scores)
        r_trn, p_trn = sp_stats.pearsonr(densities, trn_scores)
        print(f"\nPearson: density vs visual score r={r_vis:.3f}, p={p_vis:.2e}")
        print(f"Pearson: density vs transcript score r={r_trn:.3f}, p={p_trn:.2e}")

    csv_path = os.path.join(output_dir, "instructional_density_vs_engagement.csv")
    if rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved: {csv_path}")


def save_full_dataset(records: List[Dict], output_dir: str):
    """Save the full joined dataset as CSV for further analysis."""
    csv_path = os.path.join(output_dir, "full_segment_dataset.csv")
    if not records:
        return
    fieldnames = list(records[0].keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    print(f"\nFull dataset saved: {csv_path} ({len(records)} rows)")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Analyze video classification vs engagement")
    parser.add_argument("--output_dir", default=os.path.join(HERE, "..", "results", "classification_analysis"),
                        help="Output directory for analysis results")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load subgroup mapping
    subgroup_map = load_subgroup_map()
    print(f"Subgroup map: {len(subgroup_map)} videos")

    # Build joined dataset
    records = build_joined_dataset(subgroup_map)
    print(f"Dataset: {len(records)} segment records")

    # Run analyses
    analyze_engagement_by_content_type(records, args.output_dir)
    analyze_content_distribution(records, args.output_dir)
    analyze_alignment_vs_engagement(records, args.output_dir)
    analyze_narration_hands(records, args.output_dir)
    analyze_statistical_tests(records, args.output_dir)
    analyze_top_engaged_content(records, args.output_dir)
    analyze_instructional_density(records, args.output_dir)
    save_full_dataset(records, args.output_dir)

    print("\n" + "=" * 70)
    print("All analyses complete.")
    print(f"Results saved to: {args.output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
