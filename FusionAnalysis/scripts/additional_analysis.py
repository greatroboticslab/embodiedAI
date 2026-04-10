#!/usr/bin/env python3
"""
Additional statistical analyses for the Fusion Analysis paper.

Analyses:
1. Two-Way ANOVA (Content Type x Subgroup) on visual_avg_score
2. Multiple Regression predicting visual_avg_score
3. Video-Level Aggregation with per-video metrics
4. Temporal Analysis (early/middle/late bins)
5. Visual-Transcript Correlation at segment and subgroup level
"""

import os
import warnings
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

warnings.filterwarnings('ignore', category=FutureWarning)

# Paths
DATA_PATH = '/ocean/projects/cis240145p/byler/ben/embodiedAI/FusionAnalysis/results/classification_analysis/full_segment_dataset.csv'
OUTPUT_DIR = '/ocean/projects/cis240145p/byler/ben/embodiedAI/FusionAnalysis/results/classification_analysis'

def load_data():
    df = pd.read_csv(DATA_PATH)
    # Ensure booleans
    df['hands_visible'] = df['hands_visible'].astype(bool)
    df['narration'] = df['narration'].astype(bool)
    print(f"Loaded {len(df)} segments from {df['video_id'].nunique()} videos")
    print(f"Subgroups: {df['subgroup'].value_counts().to_dict()}")
    print(f"Content types: {sorted(df['content_type'].unique())}")
    print()
    return df


# =============================================================================
# ANALYSIS 1: Two-Way ANOVA (Content Type x Subgroup)
# =============================================================================
def analysis_1_twoway_anova(df):
    print("=" * 80)
    print("ANALYSIS 1: Two-Way ANOVA — visual_avg_score ~ content_type * subgroup")
    print("=" * 80)
    print()

    # Fit the two-way ANOVA model with interaction
    model = ols('visual_avg_score ~ C(content_type) + C(subgroup) + C(content_type):C(subgroup)', data=df).fit()
    anova_table = anova_lm(model, typ=2)

    print("Two-Way ANOVA Table (Type II SS):")
    print("-" * 80)
    print(anova_table.to_string())
    print()

    # Compute eta-squared for each factor
    ss_total = anova_table['sum_sq'].sum()
    print("Effect Sizes (Eta-squared = SS_effect / SS_total):")
    print("-" * 50)
    for factor in anova_table.index:
        if factor != 'Residual':
            eta_sq = anova_table.loc[factor, 'sum_sq'] / ss_total
            print(f"  {factor}")
            print(f"    SS = {anova_table.loc[factor, 'sum_sq']:.2f}")
            print(f"    F  = {anova_table.loc[factor, 'F']:.4f}")
            print(f"    p  = {anova_table.loc[factor, 'PR(>F)']:.2e}")
            print(f"    eta² = {eta_sq:.4f} ({eta_sq*100:.2f}%)")
            print()

    # Also report cell means
    print("Cell Means (visual_avg_score by content_type x subgroup):")
    print("-" * 80)
    cell_means = df.groupby(['content_type', 'subgroup'])['visual_avg_score'].agg(['mean', 'count', 'std']).round(2)
    print(cell_means.to_string())
    print()

    return anova_table


# =============================================================================
# ANALYSIS 2: Multiple Regression
# =============================================================================
def analysis_2_regression(df):
    print("=" * 80)
    print("ANALYSIS 2: Multiple Regression — visual_avg_score ~ content_type + hands_visible + narration + instructional_density + subgroup")
    print("=" * 80)
    print()

    # Create dummy variables
    # Reference categories: content_type=slide_or_powerpoint, subgroup=conventional
    df_reg = df.copy()

    # Dummy code content_type (drop slide_or_powerpoint as reference)
    content_dummies = pd.get_dummies(df_reg['content_type'], prefix='ct', drop_first=False, dtype=float)
    ref_content = 'ct_slide_or_powerpoint'
    content_dummies = content_dummies.drop(columns=[ref_content])

    # Dummy code subgroup (drop conventional as reference)
    subgroup_dummies = pd.get_dummies(df_reg['subgroup'], prefix='sg', drop_first=False, dtype=float)
    ref_subgroup = 'sg_conventional'
    subgroup_dummies = subgroup_dummies.drop(columns=[ref_subgroup])

    # Build predictor matrix
    X = pd.concat([
        content_dummies,
        df_reg['hands_visible'].astype(float).rename('hands_visible'),
        df_reg['narration'].astype(float).rename('narration'),
        df_reg['instructional_density'].astype(float).rename('instructional_density'),
        subgroup_dummies
    ], axis=1)

    X = sm.add_constant(X)
    y = df_reg['visual_avg_score']

    model = sm.OLS(y, X).fit()

    print("Regression Summary:")
    print("-" * 80)
    print(model.summary().as_text())
    print()

    # Explicit coefficient table
    print("Coefficient Details:")
    print("-" * 80)
    print(f"{'Predictor':<35} {'Coef':>10} {'Std Err':>10} {'t':>10} {'p-value':>12}")
    print("-" * 80)
    for name in model.params.index:
        coef = model.params[name]
        se = model.bse[name]
        t = model.tvalues[name]
        p = model.pvalues[name]
        sig = ''
        if p < 0.001:
            sig = '***'
        elif p < 0.01:
            sig = '**'
        elif p < 0.05:
            sig = '*'
        print(f"  {name:<33} {coef:>10.4f} {se:>10.4f} {t:>10.4f} {p:>12.2e} {sig}")
    print()
    print(f"Reference: content_type = slide_or_powerpoint, subgroup = conventional")
    print(f"R² = {model.rsquared:.4f}")
    print(f"Adjusted R² = {model.rsquared_adj:.4f}")
    print(f"F-statistic = {model.fvalue:.4f}, p = {model.f_pvalue:.2e}")
    print(f"N = {int(model.nobs)}")
    print()

    # Save regression summary to text file
    reg_path = os.path.join(OUTPUT_DIR, 'regression_summary.txt')
    with open(reg_path, 'w') as f:
        f.write("Multiple Regression: visual_avg_score ~ content_type + hands_visible + narration + instructional_density + subgroup\n")
        f.write("Reference categories: content_type=slide_or_powerpoint, subgroup=conventional\n\n")
        f.write(model.summary().as_text())
    print(f"Saved regression summary to: {reg_path}")
    print()

    return model


# =============================================================================
# ANALYSIS 3: Video-Level Aggregation
# =============================================================================
def analysis_3_video_level(df):
    print("=" * 80)
    print("ANALYSIS 3: Video-Level Aggregation")
    print("=" * 80)
    print()

    # Compute per-video metrics
    video_agg = df.groupby('video_id').agg(
        subgroup=('subgroup', 'first'),
        mean_visual_avg_score=('visual_avg_score', 'mean'),
        mean_visual_pct_correlated=('visual_pct_correlated', 'mean'),
        total_segments=('segment_index', 'count'),
    ).reset_index()

    # Compute % hands_on_demonstration per video
    hands_on_pct = df.groupby('video_id').apply(
        lambda x: (x['content_type'] == 'hands_on_demonstration').mean() * 100,
        include_groups=False
    ).rename('pct_hands_on')
    video_agg = video_agg.merge(hands_on_pct, on='video_id')

    video_agg = video_agg.round(2)

    print(f"Video-level dataset: {len(video_agg)} videos")
    print()

    # Overall Pearson correlation: %_hands_on vs mean_visual_score
    r, p = stats.pearsonr(video_agg['pct_hands_on'], video_agg['mean_visual_avg_score'])
    print(f"Overall Pearson correlation (% hands_on vs mean visual score):")
    print(f"  r = {r:.4f}, p = {p:.2e}, N = {len(video_agg)}")
    print()

    # Per-subgroup correlation
    print("Per-subgroup correlations (% hands_on vs mean visual score):")
    print("-" * 60)
    for sg in sorted(video_agg['subgroup'].unique()):
        sub = video_agg[video_agg['subgroup'] == sg]
        if len(sub) >= 3:
            r_sg, p_sg = stats.pearsonr(sub['pct_hands_on'], sub['mean_visual_avg_score'])
            print(f"  {sg}: r = {r_sg:.4f}, p = {p_sg:.2e}, N = {len(sub)}")
        else:
            print(f"  {sg}: too few videos (N={len(sub)})")
    print()

    # Summary stats by subgroup
    print("Video-Level Summary by Subgroup:")
    print("-" * 80)
    for sg in sorted(video_agg['subgroup'].unique()):
        sub = video_agg[video_agg['subgroup'] == sg]
        print(f"  {sg} (N={len(sub)}):")
        print(f"    mean visual score:   {sub['mean_visual_avg_score'].mean():.2f} (SD={sub['mean_visual_avg_score'].std():.2f})")
        print(f"    mean visual pct:     {sub['mean_visual_pct_correlated'].mean():.2f} (SD={sub['mean_visual_pct_correlated'].std():.2f})")
        print(f"    mean % hands_on:     {sub['pct_hands_on'].mean():.2f} (SD={sub['pct_hands_on'].std():.2f})")
        print(f"    mean total segments: {sub['total_segments'].mean():.1f}")
    print()

    # Save CSV
    csv_path = os.path.join(OUTPUT_DIR, 'video_level_metrics.csv')
    video_agg.to_csv(csv_path, index=False)
    print(f"Saved video-level CSV to: {csv_path}")
    print()

    return video_agg


# =============================================================================
# ANALYSIS 4: Temporal Analysis
# =============================================================================
def analysis_4_temporal(df):
    print("=" * 80)
    print("ANALYSIS 4: Temporal Analysis (Early / Middle / Late)")
    print("=" * 80)
    print()

    # Compute normalized position within each video
    max_idx = df.groupby('video_id')['segment_index'].transform('max')
    df = df.copy()
    df['normalized_position'] = df['segment_index'] / max_idx

    # Assign temporal bins
    def assign_bin(pos):
        if pos <= 0.33:
            return 'early'
        elif pos <= 0.67:
            return 'middle'
        else:
            return 'late'

    df['temporal_bin'] = df['normalized_position'].apply(assign_bin)

    # ---- Overall temporal patterns ----
    print("Overall Mean by Temporal Bin:")
    print("-" * 60)
    overall = df.groupby('temporal_bin').agg(
        mean_visual_avg=('visual_avg_score', 'mean'),
        mean_visual_pct=('visual_pct_correlated', 'mean'),
        count=('visual_avg_score', 'count')
    ).round(2)
    # Reorder
    overall = overall.reindex(['early', 'middle', 'late'])
    print(overall.to_string())
    print()

    # ---- By subgroup x temporal bin ----
    print("Mean visual_avg_score by Subgroup x Temporal Bin:")
    print("-" * 70)
    by_sg_bin = df.groupby(['subgroup', 'temporal_bin']).agg(
        mean_visual_avg=('visual_avg_score', 'mean'),
        mean_visual_pct=('visual_pct_correlated', 'mean'),
        count=('visual_avg_score', 'count')
    ).round(2)
    # Reorder temporal bin
    by_sg_bin = by_sg_bin.reindex(['early', 'middle', 'late'], level='temporal_bin')
    print(by_sg_bin.to_string())
    print()

    # ---- Hands-on distribution across temporal bins ----
    print("Percentage of Hands-On Demonstration Segments by Temporal Bin and Subgroup:")
    print("-" * 70)
    df['is_hands_on'] = (df['content_type'] == 'hands_on_demonstration').astype(int)

    hands_on_by_bin = df.groupby(['subgroup', 'temporal_bin']).agg(
        pct_hands_on=('is_hands_on', 'mean'),
        total_segments=('is_hands_on', 'count')
    ).round(4)
    hands_on_by_bin['pct_hands_on'] = (hands_on_by_bin['pct_hands_on'] * 100).round(2)
    hands_on_by_bin = hands_on_by_bin.reindex(['early', 'middle', 'late'], level='temporal_bin')
    print(hands_on_by_bin.to_string())
    print()

    # Overall hands-on by bin
    print("Overall % Hands-On by Temporal Bin:")
    overall_ho = df.groupby('temporal_bin')['is_hands_on'].mean().reindex(['early', 'middle', 'late']) * 100
    for b in ['early', 'middle', 'late']:
        print(f"  {b}: {overall_ho[b]:.2f}%")
    print()

    # ---- ANOVA: visual_avg_score by temporal bin ----
    early = df[df['temporal_bin'] == 'early']['visual_avg_score']
    middle = df[df['temporal_bin'] == 'middle']['visual_avg_score']
    late = df[df['temporal_bin'] == 'late']['visual_avg_score']

    f_stat, p_val = stats.f_oneway(early, middle, late)
    print(f"One-Way ANOVA: visual_avg_score ~ temporal_bin")
    print(f"  F = {f_stat:.4f}, p = {p_val:.2e}")
    print(f"  N (early={len(early)}, middle={len(middle)}, late={len(late)})")
    print()

    # ---- Save temporal results to CSV ----
    # Combine all results
    temporal_results = df.groupby(['subgroup', 'temporal_bin']).agg(
        mean_visual_avg_score=('visual_avg_score', 'mean'),
        mean_visual_pct_correlated=('visual_pct_correlated', 'mean'),
        pct_hands_on=('is_hands_on', 'mean'),
        n_segments=('visual_avg_score', 'count')
    ).round(4)
    temporal_results['pct_hands_on'] = (temporal_results['pct_hands_on'] * 100).round(2)
    temporal_results = temporal_results.reindex(['early', 'middle', 'late'], level='temporal_bin')

    csv_path = os.path.join(OUTPUT_DIR, 'temporal_analysis.csv')
    temporal_results.to_csv(csv_path)
    print(f"Saved temporal analysis CSV to: {csv_path}")
    print()

    return df


# =============================================================================
# ANALYSIS 5: Visual-Transcript Correlation
# =============================================================================
def analysis_5_visual_transcript_correlation(df):
    print("=" * 80)
    print("ANALYSIS 5: Visual-Transcript Correlation")
    print("=" * 80)
    print()

    # Overall Pearson correlation
    r, p = stats.pearsonr(df['visual_avg_score'], df['transcript_avg_score'])
    print(f"Overall Pearson correlation (visual_avg_score vs transcript_avg_score):")
    print(f"  r = {r:.4f}, p = {p:.2e}, N = {len(df)}")
    print()

    # Per-subgroup
    print("Per-Subgroup Correlations:")
    print("-" * 60)
    results = []
    for sg in sorted(df['subgroup'].unique()):
        sub = df[df['subgroup'] == sg]
        r_sg, p_sg = stats.pearsonr(sub['visual_avg_score'], sub['transcript_avg_score'])
        print(f"  {sg}: r = {r_sg:.4f}, p = {p_sg:.2e}, N = {len(sub)}")
        results.append({
            'subgroup': sg,
            'pearson_r': round(r_sg, 4),
            'p_value': p_sg,
            'n': len(sub)
        })
    print()

    # Also report descriptive stats
    print("Descriptive Statistics for visual_avg_score and transcript_avg_score:")
    print("-" * 70)
    for sg in sorted(df['subgroup'].unique()):
        sub = df[df['subgroup'] == sg]
        print(f"  {sg} (N={len(sub)}):")
        print(f"    visual_avg_score:      mean={sub['visual_avg_score'].mean():.2f}, SD={sub['visual_avg_score'].std():.2f}")
        print(f"    transcript_avg_score:  mean={sub['transcript_avg_score'].mean():.2f}, SD={sub['transcript_avg_score'].std():.2f}")
    print()

    # Interpretation
    print("Interpretation:")
    if r > 0.3:
        print(f"  Moderate-to-strong positive correlation (r={r:.4f}): Visual and transcript")
        print(f"  engagement tend to co-occur — segments with high visual scores also have")
        print(f"  high transcript scores. This suggests complementary channels.")
    elif r > 0:
        print(f"  Weak positive correlation (r={r:.4f}): Visual and transcript engagement")
        print(f"  show a slight tendency to co-occur but are largely independent.")
    elif r > -0.3:
        print(f"  Weak negative correlation (r={r:.4f}): Visual and transcript engagement")
        print(f"  show a slight tendency to trade off but are largely independent.")
    else:
        print(f"  Moderate-to-strong negative correlation (r={r:.4f}): Visual and transcript")
        print(f"  engagement tend to trade off — high visual engagement co-occurs with low")
        print(f"  transcript engagement. This suggests competing channels.")
    print()

    # Save correlation results
    results_df = pd.DataFrame(results)
    results_df.loc[len(results_df)] = {
        'subgroup': 'overall',
        'pearson_r': round(r, 4),
        'p_value': p,
        'n': len(df)
    }
    csv_path = os.path.join(OUTPUT_DIR, 'visual_transcript_correlation.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"Saved visual-transcript correlation CSV to: {csv_path}")
    print()


# =============================================================================
# MAIN
# =============================================================================
def main():
    df = load_data()

    print()
    analysis_1_twoway_anova(df)
    analysis_2_regression(df)
    video_agg = analysis_3_video_level(df)
    analysis_4_temporal(df)
    analysis_5_visual_transcript_correlation(df)

    print("=" * 80)
    print("ALL ANALYSES COMPLETE")
    print("=" * 80)
    print()
    print("Output files saved to:")
    print(f"  {OUTPUT_DIR}/regression_summary.txt")
    print(f"  {OUTPUT_DIR}/video_level_metrics.csv")
    print(f"  {OUTPUT_DIR}/temporal_analysis.csv")
    print(f"  {OUTPUT_DIR}/visual_transcript_correlation.csv")


if __name__ == '__main__':
    main()
