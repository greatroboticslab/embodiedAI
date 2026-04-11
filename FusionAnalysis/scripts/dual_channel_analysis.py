#!/usr/bin/env python3
"""
Dual-channel analysis for the Fusion paper.
Computes transcript, union (visual OR transcript), and intersection (visual AND transcript)
correlation metrics by content type, subgroup, and temporal bin.

Run with: conda run -n llava python3 dual_channel_analysis.py
"""

import os
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.formula.api import ols as ols_formula
from statsmodels.stats.anova import anova_lm

warnings.filterwarnings('ignore')

DATA_PATH = '/ocean/projects/cis240145p/byler/ben/embodiedAI/FusionAnalysis/results/classification_analysis/full_segment_dataset.csv'
OUTPUT_DIR = '/ocean/projects/cis240145p/byler/ben/embodiedAI/FusionAnalysis/results/classification_analysis'


def load_and_prepare():
    df = pd.read_csv(DATA_PATH)
    df['hands_visible'] = df['hands_visible'].astype(bool)
    df['narration'] = df['narration'].astype(bool)

    # Binary correlated flags
    df['vis_corr'] = (df['visual_pct_correlated'] > 0).astype(int)
    df['tr_corr'] = (df['transcript_pct_correlated'] > 0).astype(int)
    df['union_corr'] = ((df['vis_corr'] == 1) | (df['tr_corr'] == 1)).astype(int)
    df['both_corr'] = ((df['vis_corr'] == 1) & (df['tr_corr'] == 1)).astype(int)

    # Union score = max of visual and transcript avg scores
    df['union_score'] = df[['visual_avg_score', 'transcript_avg_score']].max(axis=1)

    # Temporal bins
    max_seg = df.groupby('video_id')['segment_index'].transform('max')
    df['norm_pos'] = df['segment_index'] / max_seg
    df['temporal_bin'] = pd.cut(df['norm_pos'], bins=[0, 0.333, 0.667, 1.0],
                                labels=['early', 'middle', 'late'], include_lowest=True)

    print(f"Loaded {len(df)} segments, {df['video_id'].nunique()} videos")
    return df


def analysis_by_content_type(df):
    print("\n" + "=" * 80)
    print("ENGAGEMENT BY CONTENT TYPE — ALL CHANNELS")
    print("=" * 80)

    rows = []
    for ct in ['hands_on_demonstration', 'diagram_or_whiteboard', 'animation_or_graphic',
               'equipment_closeup', 'presenter_talking', 'slide_or_powerpoint',
               'device_in_operation', 'screen_or_software', 'other']:
        sub = df[df['content_type'] == ct]
        rows.append({
            'content_type': ct,
            'n': len(sub),
            'vis_corr_pct': sub['vis_corr'].mean() * 100,
            'vis_avg_score': sub['visual_avg_score'].mean(),
            'tr_corr_pct': sub['tr_corr'].mean() * 100,
            'tr_avg_score': sub['transcript_avg_score'].mean(),
            'union_pct': sub['union_corr'].mean() * 100,
            'union_score': sub['union_score'].mean(),
            'both_pct': sub['both_corr'].mean() * 100,
        })

    result = pd.DataFrame(rows)
    result.to_csv(os.path.join(OUTPUT_DIR, 'engagement_by_content_type_all_channels.csv'), index=False)
    print(result.to_string(index=False))

    # ANOVAs
    groups_tr = [g['transcript_avg_score'].values for _, g in df.groupby('content_type')]
    f_tr, p_tr = stats.f_oneway(*groups_tr)
    print(f"\nTranscript ANOVA by content type: F={f_tr:.1f}, p={p_tr:.2e}")

    groups_u = [g['union_score'].values for _, g in df.groupby('content_type')]
    f_u, p_u = stats.f_oneway(*groups_u)
    print(f"Union score ANOVA by content type: F={f_u:.1f}, p={p_u:.2e}")


def analysis_by_subgroup(df):
    print("\n" + "=" * 80)
    print("ENGAGEMENT BY SUBGROUP — ALL CHANNELS")
    print("=" * 80)

    rows = []
    for sg in ['conventional', 'verbally_embodied', 'visually_embodied']:
        sub = df[df['subgroup'] == sg]
        rows.append({
            'subgroup': sg,
            'n': len(sub),
            'vis_corr_pct': sub['vis_corr'].mean() * 100,
            'vis_avg_score': sub['visual_avg_score'].mean(),
            'tr_corr_pct': sub['tr_corr'].mean() * 100,
            'tr_avg_score': sub['transcript_avg_score'].mean(),
            'union_pct': sub['union_corr'].mean() * 100,
            'union_score': sub['union_score'].mean(),
            'both_pct': sub['both_corr'].mean() * 100,
        })

    result = pd.DataFrame(rows)
    result.to_csv(os.path.join(OUTPUT_DIR, 'engagement_by_subgroup_all_channels.csv'), index=False)
    print(result.to_string(index=False))

    # ANOVAs
    groups = [g['transcript_avg_score'].values for _, g in df.groupby('subgroup')]
    f, p = stats.f_oneway(*groups)
    print(f"\nTranscript ANOVA by subgroup: F={f:.1f}, p={p:.2e}")

    groups_u = [g['union_score'].values for _, g in df.groupby('subgroup')]
    f_u, p_u = stats.f_oneway(*groups_u)
    print(f"Union score ANOVA by subgroup: F={f_u:.1f}, p={p_u:.2e}")


def twoway_anova_transcript(df):
    print("\n" + "=" * 80)
    print("TWO-WAY ANOVA: transcript_avg_score ~ content_type * subgroup")
    print("=" * 80)

    model = ols_formula('transcript_avg_score ~ C(content_type) + C(subgroup) + C(content_type):C(subgroup)', data=df).fit()
    aov = anova_lm(model, typ=2)
    ss_total = aov['sum_sq'].sum()
    aov['eta_sq_pct'] = aov['sum_sq'] / ss_total * 100
    for idx, row in aov.iterrows():
        if idx != 'Residual':
            print(f"  {idx}: F={row['F']:.1f}, p={row['PR(>F)']:.2e}, eta2={row['eta_sq_pct']:.2f}%")


def temporal_all_channels(df):
    print("\n" + "=" * 80)
    print("TEMPORAL ANALYSIS — ALL CHANNELS")
    print("=" * 80)

    rows = []
    for sg in ['conventional', 'verbally_embodied', 'visually_embodied']:
        for tb in ['early', 'middle', 'late']:
            sub = df[(df['subgroup'] == sg) & (df['temporal_bin'] == tb)]
            rows.append({
                'subgroup': sg, 'temporal_bin': tb, 'n': len(sub),
                'vis_score': sub['visual_avg_score'].mean(),
                'tr_score': sub['transcript_avg_score'].mean(),
                'vis_pct': sub['vis_corr'].mean() * 100,
                'tr_pct': sub['tr_corr'].mean() * 100,
                'union_pct': sub['union_corr'].mean() * 100,
                'both_pct': sub['both_corr'].mean() * 100,
            })

    result = pd.DataFrame(rows)
    result.to_csv(os.path.join(OUTPUT_DIR, 'temporal_analysis_all_channels.csv'), index=False)
    print(result.to_string(index=False))


if __name__ == '__main__':
    df = load_and_prepare()
    analysis_by_content_type(df)
    analysis_by_subgroup(df)
    twoway_anova_transcript(df)
    temporal_all_channels(df)
    print("\nDone. Results saved to", OUTPUT_DIR)
