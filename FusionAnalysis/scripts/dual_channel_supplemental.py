#!/usr/bin/env python3
"""
Supplemental dual-channel breakdowns for the Fusion paper rebalance.

Produces:
  1. Instructional density x channel (visual, transcript, union) mean scores
  2. Top-quartile vs. bottom-quartile content distribution when videos are
     ranked by mean UNION score (instead of visual-only) within each subgroup
  3. Hand-visibility dual-channel means (already in paper, verified)

Run with: conda run -n llava python3 dual_channel_supplemental.py
"""

import os
import warnings

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings('ignore')

DATA_PATH = '/ocean/projects/cis240145p/byler/ben/embodiedAI/FusionAnalysis/results/classification_analysis/full_segment_dataset.csv'
OUTPUT_DIR = '/ocean/projects/cis240145p/byler/ben/embodiedAI/FusionAnalysis/results/classification_analysis'


def load():
    df = pd.read_csv(DATA_PATH)
    df['hands_visible'] = df['hands_visible'].astype(bool)
    df['narration'] = df['narration'].astype(bool)
    df['union_score'] = df[['visual_avg_score', 'transcript_avg_score']].max(axis=1)
    df['vis_corr'] = (df['visual_pct_correlated'] > 0).astype(int)
    df['tr_corr'] = (df['transcript_pct_correlated'] > 0).astype(int)
    return df


def density_channels(df):
    print('\n=== Instructional density x channel ===')
    rows = []
    for d in sorted(df['instructional_density'].unique()):
        sub = df[df['instructional_density'] == d]
        rows.append({
            'density': int(d),
            'n': len(sub),
            'vis_pct': sub['visual_pct_correlated'].mean(),
            'tr_pct': sub['transcript_pct_correlated'].mean(),
            'vis_score': sub['visual_avg_score'].mean(),
            'tr_score': sub['transcript_avg_score'].mean(),
            'union_score': sub['union_score'].mean(),
        })
    out = pd.DataFrame(rows).round(2)
    print(out.to_string(index=False))
    out.to_csv(os.path.join(OUTPUT_DIR, 'density_dual_channel.csv'), index=False)

    # ANOVAs
    groups_vis = [g['visual_avg_score'].values for _, g in df.groupby('instructional_density')]
    groups_tr = [g['transcript_avg_score'].values for _, g in df.groupby('instructional_density')]
    groups_u = [g['union_score'].values for _, g in df.groupby('instructional_density')]
    print(f"  Visual ANOVA:     F={stats.f_oneway(*groups_vis).statistic:.2f}, p={stats.f_oneway(*groups_vis).pvalue:.2e}")
    print(f"  Transcript ANOVA: F={stats.f_oneway(*groups_tr).statistic:.2f}, p={stats.f_oneway(*groups_tr).pvalue:.2e}")
    print(f"  Union ANOVA:      F={stats.f_oneway(*groups_u).statistic:.2f}, p={stats.f_oneway(*groups_u).pvalue:.2e}")


def hand_visibility_channels(df):
    print('\n=== Hand visibility x channel ===')
    rows = []
    for val in [True, False]:
        sub = df[df['hands_visible'] == val]
        rows.append({
            'hands_visible': val,
            'n': len(sub),
            'vis_pct': sub['visual_pct_correlated'].mean(),
            'tr_pct': sub['transcript_pct_correlated'].mean(),
            'union_pct': ((sub['vis_corr'] == 1) | (sub['tr_corr'] == 1)).mean() * 100,
            'vis_score': sub['visual_avg_score'].mean(),
            'tr_score': sub['transcript_avg_score'].mean(),
            'union_score': sub['union_score'].mean(),
        })
    out = pd.DataFrame(rows).round(2)
    print(out.to_string(index=False))
    out.to_csv(os.path.join(OUTPUT_DIR, 'hand_visibility_dual_channel.csv'), index=False)


def top_bottom_quartile_union(df):
    print('\n=== Top/bottom quartile content distribution by UNION score ===')
    # Video-level aggregation
    video_agg = df.groupby('video_id').agg(
        subgroup=('subgroup', 'first'),
        mean_visual=('visual_avg_score', 'mean'),
        mean_transcript=('transcript_avg_score', 'mean'),
        mean_union=('union_score', 'mean'),
    ).reset_index()

    content_types = sorted(df['content_type'].unique())
    all_rows = []

    for sg in ['conventional', 'verbally_embodied', 'visually_embodied']:
        sub_vids = video_agg[video_agg['subgroup'] == sg].sort_values('mean_union')
        if len(sub_vids) < 8:
            print(f"  skipping {sg}: only {len(sub_vids)} videos")
            continue
        q = int(np.ceil(len(sub_vids) / 4))
        bottom_ids = set(sub_vids.head(q)['video_id'])
        top_ids = set(sub_vids.tail(q)['video_id'])

        top_segs = df[df['video_id'].isin(top_ids)]
        bot_segs = df[df['video_id'].isin(bottom_ids)]

        print(f"\n  {sg}: top {len(top_ids)} videos / bottom {len(bottom_ids)} videos")
        print(f"    top mean union    = {top_segs['union_score'].mean():.2f}")
        print(f"    bottom mean union = {bot_segs['union_score'].mean():.2f}")

        for ct in content_types:
            top_pct = (top_segs['content_type'] == ct).mean() * 100
            bot_pct = (bot_segs['content_type'] == ct).mean() * 100
            all_rows.append({
                'subgroup': sg,
                'content_type': ct,
                'top_pct': round(top_pct, 2),
                'bottom_pct': round(bot_pct, 2),
                'delta': round(top_pct - bot_pct, 2),
            })

        # Print table for this subgroup
        sub_rows = [r for r in all_rows if r['subgroup'] == sg]
        sub_rows.sort(key=lambda r: -r['delta'])
        print(f"    {'content':<26} {'top%':>8} {'bot%':>8} {'delta':>8}")
        for r in sub_rows:
            print(f"    {r['content_type']:<26} {r['top_pct']:>8.2f} {r['bottom_pct']:>8.2f} {r['delta']:>+8.2f}")

    pd.DataFrame(all_rows).to_csv(
        os.path.join(OUTPUT_DIR, 'top_bottom_quartile_by_union.csv'), index=False
    )


if __name__ == '__main__':
    df = load()
    density_channels(df)
    hand_visibility_channels(df)
    top_bottom_quartile_union(df)
    print('\nDone.')
