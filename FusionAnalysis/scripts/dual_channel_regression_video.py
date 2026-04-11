#!/usr/bin/env python3
"""
Dual-channel regression and video-level analyses for the Fusion paper.

Fits three parallel OLS models predicting engagement score from content type,
segment features, and subgroup:
  1. visual_avg_score
  2. transcript_avg_score
  3. union_score (max of visual and transcript)

Also extends the video-level aggregation with mean transcript and union scores,
and computes per-subgroup Pearson correlations of pct_hands_on against each
channel.

Run with: conda run -n llava python3 dual_channel_regression_video.py
"""

import os
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

warnings.filterwarnings('ignore', category=FutureWarning)

DATA_PATH = '/ocean/projects/cis240145p/byler/ben/embodiedAI/FusionAnalysis/results/classification_analysis/full_segment_dataset.csv'
OUTPUT_DIR = '/ocean/projects/cis240145p/byler/ben/embodiedAI/FusionAnalysis/results/classification_analysis'


def load_data():
    df = pd.read_csv(DATA_PATH)
    df['hands_visible'] = df['hands_visible'].astype(bool)
    df['narration'] = df['narration'].astype(bool)
    df['union_score'] = df[['visual_avg_score', 'transcript_avg_score']].max(axis=1)
    print(f"Loaded {len(df)} segments from {df['video_id'].nunique()} videos")
    return df


def build_design_matrix(df):
    """Reference categories: content_type = slide_or_powerpoint, subgroup = conventional."""
    content_dummies = pd.get_dummies(df['content_type'], prefix='ct', dtype=float)
    content_dummies = content_dummies.drop(columns=['ct_slide_or_powerpoint'])

    subgroup_dummies = pd.get_dummies(df['subgroup'], prefix='sg', dtype=float)
    subgroup_dummies = subgroup_dummies.drop(columns=['sg_conventional'])

    X = pd.concat([
        content_dummies,
        df['hands_visible'].astype(float).rename('hands_visible'),
        df['narration'].astype(float).rename('narration'),
        df['instructional_density'].astype(float).rename('instructional_density'),
        subgroup_dummies,
    ], axis=1)
    X = sm.add_constant(X)
    return X


def fit_regression(X, y, label):
    model = sm.OLS(y, X).fit()
    print(f"\n--- {label} ---")
    print(f"  R^2 = {model.rsquared:.4f}, F = {model.fvalue:.2f}, p = {model.f_pvalue:.2e}, N = {int(model.nobs)}")
    return model


def save_combined_regression_table(models_by_dv, path):
    """Write a side-by-side coefficient table for all three models."""
    dvs = list(models_by_dv.keys())
    predictors = list(models_by_dv[dvs[0]].params.index)

    rows = []
    for pred in predictors:
        row = {'predictor': pred}
        for dv in dvs:
            m = models_by_dv[dv]
            row[f'{dv}_coef'] = m.params[pred]
            row[f'{dv}_se'] = m.bse[pred]
            row[f'{dv}_t'] = m.tvalues[pred]
            row[f'{dv}_p'] = m.pvalues[pred]
        rows.append(row)

    summary = pd.DataFrame(rows)

    # Fit-statistics footer
    fit_rows = []
    for stat in ['R2', 'F', 'p', 'N']:
        row = {'predictor': stat}
        for dv in dvs:
            m = models_by_dv[dv]
            if stat == 'R2':
                val = m.rsquared
            elif stat == 'F':
                val = m.fvalue
            elif stat == 'p':
                val = m.f_pvalue
            else:
                val = int(m.nobs)
            row[f'{dv}_coef'] = val
            row[f'{dv}_se'] = np.nan
            row[f'{dv}_t'] = np.nan
            row[f'{dv}_p'] = np.nan
        fit_rows.append(row)

    out = pd.concat([summary, pd.DataFrame(fit_rows)], ignore_index=True)
    out.to_csv(path, index=False)
    return out


def print_pretty_coef_table(models_by_dv):
    predictors = list(models_by_dv['visual'].params.index)
    header = f"{'Predictor':<32} "
    for dv in ['visual', 'transcript', 'union']:
        header += f"{dv+' beta':>12} {'p':>10}  "
    print(header)
    print('-' * len(header))
    for pred in predictors:
        line = f"  {pred:<30} "
        for dv in ['visual', 'transcript', 'union']:
            m = models_by_dv[dv]
            coef = m.params[pred]
            p = m.pvalues[pred]
            line += f"{coef:>12.3f} {p:>10.2e}  "
        print(line)
    print('-' * len(header))
    for dv in ['visual', 'transcript', 'union']:
        m = models_by_dv[dv]
        print(f"  {dv}: R^2 = {m.rsquared:.4f}, F = {m.fvalue:.2f}, p = {m.f_pvalue:.2e}")


def analysis_dual_channel_regression(df):
    print("\n" + "=" * 80)
    print("DUAL-CHANNEL REGRESSION")
    print("=" * 80)

    X = build_design_matrix(df)
    models = {
        'visual': fit_regression(X, df['visual_avg_score'], 'visual_avg_score'),
        'transcript': fit_regression(X, df['transcript_avg_score'], 'transcript_avg_score'),
        'union': fit_regression(X, df['union_score'], 'union_score'),
    }

    print()
    print_pretty_coef_table(models)

    out_path = os.path.join(OUTPUT_DIR, 'regression_dual_channel.csv')
    save_combined_regression_table(models, out_path)
    print(f"\nSaved combined regression table to: {out_path}")

    # Full text summaries for the record
    txt_path = os.path.join(OUTPUT_DIR, 'regression_dual_channel_summary.txt')
    with open(txt_path, 'w') as f:
        f.write("Dual-Channel Regressions: visual / transcript / union score\n")
        f.write("Reference: content_type = slide_or_powerpoint, subgroup = conventional\n\n")
        for dv, m in models.items():
            f.write(f"\n========== {dv} ==========\n")
            f.write(m.summary().as_text())
            f.write("\n")
    print(f"Saved full regression summaries to: {txt_path}")

    return models


def analysis_video_level_dual(df):
    print("\n" + "=" * 80)
    print("VIDEO-LEVEL DUAL-CHANNEL ANALYSIS")
    print("=" * 80)

    video_agg = df.groupby('video_id').agg(
        subgroup=('subgroup', 'first'),
        mean_visual_score=('visual_avg_score', 'mean'),
        mean_transcript_score=('transcript_avg_score', 'mean'),
        mean_union_score=('union_score', 'mean'),
        mean_visual_pct=('visual_pct_correlated', 'mean'),
        mean_transcript_pct=('transcript_pct_correlated', 'mean'),
        total_segments=('segment_index', 'count'),
    ).reset_index()

    hands_on_pct = df.groupby('video_id').apply(
        lambda x: (x['content_type'] == 'hands_on_demonstration').mean() * 100,
        include_groups=False,
    ).rename('pct_hands_on')
    video_agg = video_agg.merge(hands_on_pct, on='video_id')
    video_agg = video_agg.round(4)

    out_path = os.path.join(OUTPUT_DIR, 'video_level_metrics_dual.csv')
    video_agg.to_csv(out_path, index=False)
    print(f"Saved video-level metrics to: {out_path}")
    print(f"N videos: {len(video_agg)}")

    # Correlations: pct_hands_on vs each channel, overall and per subgroup
    channels = {
        'visual': 'mean_visual_score',
        'transcript': 'mean_transcript_score',
        'union': 'mean_union_score',
    }

    rows = []
    print("\nPearson r (pct_hands_on vs each channel):")
    print(f"{'Subset':<22} {'N':>4}  " + "  ".join([f"{c+' r':>12} {'p':>10}" for c in channels]))
    print('-' * 90)

    for label, subset in [('overall', video_agg)] + [
        (sg, video_agg[video_agg['subgroup'] == sg]) for sg in sorted(video_agg['subgroup'].unique())
    ]:
        if len(subset) < 3:
            continue
        line = f"{label:<22} {len(subset):>4}  "
        row = {'subset': label, 'n': len(subset)}
        for cname, col in channels.items():
            r, p = stats.pearsonr(subset['pct_hands_on'], subset[col])
            line += f"{r:>12.3f} {p:>10.2e}  "
            row[f'{cname}_r'] = round(r, 4)
            row[f'{cname}_p'] = p
        print(line)
        rows.append(row)

    corr_path = os.path.join(OUTPUT_DIR, 'video_level_channel_correlations.csv')
    pd.DataFrame(rows).to_csv(corr_path, index=False)
    print(f"\nSaved per-channel correlations to: {corr_path}")

    # Descriptive stats by subgroup
    print("\nPer-subgroup means (video-level):")
    for sg in sorted(video_agg['subgroup'].unique()):
        sub = video_agg[video_agg['subgroup'] == sg]
        print(f"  {sg} (N={len(sub)}):")
        print(f"    visual={sub['mean_visual_score'].mean():.2f}, transcript={sub['mean_transcript_score'].mean():.2f}, union={sub['mean_union_score'].mean():.2f}")
        print(f"    pct_hands_on={sub['pct_hands_on'].mean():.2f} (SD={sub['pct_hands_on'].std():.2f})")

    return video_agg


if __name__ == '__main__':
    df = load_data()
    analysis_dual_channel_regression(df)
    analysis_video_level_dual(df)
    print("\nDone. Results in", OUTPUT_DIR)
