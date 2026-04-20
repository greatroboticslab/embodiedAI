"""
Compute missing statistical tests for Paper 2 revision.

Gaps identified:
1. Content distribution across subgroups — chi-square test
2. Within-content-type quality (hands-on by subgroup) — ANOVA + t-tests
3. Narration effect — full t-test stats for all channels
4. Temporal dynamics — transcript and union ANOVAs
5. Content profiles (top/bottom quartile) — t-test on pct_hands_on
"""

import pandas as pd
import numpy as np
from scipy import stats

DATA = '/ocean/projects/cis240145p/byler/ben/embodiedAI/FusionAnalysis/results/classification_analysis'
df = pd.read_csv(f'{DATA}/full_segment_dataset.csv')

# Compute union and intersection metrics
df['union_score'] = df[['visual_avg_score', 'transcript_avg_score']].max(axis=1)
df['vis_corr'] = (df['visual_pct_correlated'] > 0).astype(int)
df['tr_corr'] = (df['transcript_pct_correlated'] > 0).astype(int)
df['union_corr'] = ((df['vis_corr'] == 1) | (df['tr_corr'] == 1)).astype(int)
df['both_corr'] = ((df['vis_corr'] == 1) & (df['tr_corr'] == 1)).astype(int)

print(f"Total segments: {len(df)}")
print(f"Subgroups: {df['subgroup'].value_counts().to_dict()}")
print()

# ============================================================
# 1. Content distribution across subgroups — Chi-square
# ============================================================
print("=" * 60)
print("1. CONTENT DISTRIBUTION — CHI-SQUARE TEST")
print("=" * 60)

contingency = pd.crosstab(df['content_type'], df['subgroup'])
chi2, p, dof, expected = stats.chi2_contingency(contingency)
# Cramér's V
n = contingency.sum().sum()
k = min(contingency.shape) - 1
cramers_v = np.sqrt(chi2 / (n * k))
print(f"Chi-square: χ² = {chi2:.1f}, df = {dof}, p < 10^{np.log10(p):.0f}")
print(f"Cramér's V = {cramers_v:.3f}")
print(f"Contingency table shape: {contingency.shape}")
print()

# ============================================================
# 2. Within-content-type: hands-on by subgroup — ANOVA + t-tests
# ============================================================
print("=" * 60)
print("2. WITHIN-CONTENT-TYPE: HANDS-ON BY SUBGROUP")
print("=" * 60)

hands_on = df[df['content_type'] == 'hands_on_demonstration']
print(f"Total hands-on segments: {len(hands_on)}")
for sg in ['conventional', 'verbally_embodied', 'visually_embodied']:
    sub = hands_on[hands_on['subgroup'] == sg]
    print(f"  {sg}: n={len(sub)}, vis_corr={sub['visual_pct_correlated'].mean():.1f}%, "
          f"tr_corr={sub['transcript_pct_correlated'].mean():.1f}%, "
          f"vis_score={sub['visual_avg_score'].mean():.1f}, "
          f"tr_score={sub['transcript_avg_score'].mean():.1f}")

for channel_name, channel_col in [('visual_avg_score', 'visual_avg_score'),
                                   ('transcript_avg_score', 'transcript_avg_score'),
                                   ('union_score', 'union_score'),
                                   ('visual_pct_correlated', 'visual_pct_correlated'),
                                   ('transcript_pct_correlated', 'transcript_pct_correlated')]:
    groups = [hands_on[hands_on['subgroup'] == sg][channel_col].values
              for sg in ['conventional', 'verbally_embodied', 'visually_embodied']]
    F, p = stats.f_oneway(*groups)
    print(f"\n  ANOVA on {channel_name}: F = {F:.1f}, p = {p:.2e}")
    if p < 0.05:
        # Pairwise t-tests
        labels = ['conventional', 'verbally_embodied', 'visually_embodied']
        for i in range(len(labels)):
            for j in range(i+1, len(labels)):
                t, p_pair = stats.ttest_ind(groups[i], groups[j], equal_var=False)
                d = (groups[i].mean() - groups[j].mean()) / np.sqrt((groups[i].var() + groups[j].var()) / 2)
                print(f"    {labels[i][:4]} vs {labels[j][:4]}: t = {t:.2f}, p = {p_pair:.2e}, d = {d:.3f}")

print()

# ============================================================
# 3. Narration effect — full t-tests for all channels
# ============================================================
print("=" * 60)
print("3. NARRATION EFFECT — FULL T-TESTS")
print("=" * 60)

narr_yes = df[df['narration'] == True]
narr_no = df[df['narration'] == False]
print(f"Narration present: n={len(narr_yes)}, absent: n={len(narr_no)}")

for channel_name, col in [('visual_pct_correlated', 'visual_pct_correlated'),
                           ('transcript_pct_correlated', 'transcript_pct_correlated'),
                           ('visual_avg_score', 'visual_avg_score'),
                           ('transcript_avg_score', 'transcript_avg_score'),
                           ('union_score', 'union_score'),
                           ('union_corr', 'union_corr')]:
    yes_vals = narr_yes[col].values
    no_vals = narr_no[col].values
    t, p = stats.ttest_ind(yes_vals, no_vals, equal_var=False)
    d = (yes_vals.mean() - no_vals.mean()) / np.sqrt((yes_vals.var() + no_vals.var()) / 2)
    print(f"  {channel_name}: present={yes_vals.mean():.2f}, absent={no_vals.mean():.2f}, "
          f"t = {t:.2f}, p = {p:.2e}, d = {d:.3f}")

print()

# ============================================================
# 4. Temporal dynamics — transcript and union ANOVAs
# ============================================================
print("=" * 60)
print("4. TEMPORAL DYNAMICS — TRANSCRIPT & UNION ANOVAs")
print("=" * 60)

# Assign temporal bins
def assign_bin(row):
    # Need video duration info — use max segment end per video
    return row

# Get max segment per video for normalization
video_max = df.groupby('video_id')['end'].max().reset_index()
video_max.columns = ['video_id', 'video_duration']
df2 = df.merge(video_max, on='video_id')
df2['normalized_pos'] = (df2['start'] + df2['end']) / 2 / df2['video_duration']
df2['temporal_bin'] = pd.cut(df2['normalized_pos'], bins=[0, 0.333, 0.667, 1.0],
                              labels=['Early', 'Middle', 'Late'], include_lowest=True)

for channel_name, col in [('visual_avg_score', 'visual_avg_score'),
                           ('transcript_avg_score', 'transcript_avg_score'),
                           ('union_score', 'union_score')]:
    groups = [df2[df2['temporal_bin'] == b][col].values for b in ['Early', 'Middle', 'Late']]
    F, p = stats.f_oneway(*groups)
    means = [g.mean() for g in groups]
    print(f"  {channel_name} by temporal bin: "
          f"E={means[0]:.1f}, M={means[1]:.1f}, L={means[2]:.1f}, "
          f"F = {F:.1f}, p = {p:.2e}")

# Also by subgroup
print("\n  By subgroup × temporal bin:")
for sg in ['conventional', 'verbally_embodied', 'visually_embodied']:
    sg_data = df2[df2['subgroup'] == sg]
    for channel_name, col in [('transcript_avg_score', 'transcript_avg_score'),
                               ('union_score', 'union_score')]:
        groups = [sg_data[sg_data['temporal_bin'] == b][col].values for b in ['Early', 'Middle', 'Late']]
        F, p = stats.f_oneway(*groups)
        means = [g.mean() for g in groups]
        print(f"    {sg[:4]} {channel_name}: E={means[0]:.1f}, M={means[1]:.1f}, L={means[2]:.1f}, "
              f"F = {F:.1f}, p = {p:.2e}")

print()

# ============================================================
# 5. Top/bottom quartile — t-test on pct_hands_on
# ============================================================
print("=" * 60)
print("5. TOP/BOTTOM QUARTILE — T-TESTS")
print("=" * 60)

# Compute video-level metrics
video_metrics = df.groupby(['video_id', 'subgroup']).agg(
    mean_vis_score=('visual_avg_score', 'mean'),
    mean_tr_score=('transcript_avg_score', 'mean'),
    mean_union_score=('union_score', 'mean'),
    pct_hands_on=('content_type', lambda x: (x == 'hands_on_demonstration').mean() * 100),
    pct_screen=('content_type', lambda x: (x == 'screen_or_software').mean() * 100),
    n_segments=('segment_index', 'count')
).reset_index()

for sg in ['verbally_embodied', 'conventional', 'visually_embodied']:
    sg_vids = video_metrics[video_metrics['subgroup'] == sg].copy()
    n = len(sg_vids)
    q25 = int(np.ceil(n * 0.25))

    sg_vids_sorted = sg_vids.sort_values('mean_union_score', ascending=False)
    top = sg_vids_sorted.head(q25)
    bottom = sg_vids_sorted.tail(q25)

    print(f"\n  {sg} (n={n}, quartile={q25}):")
    print(f"    Top quartile: mean_union={top['mean_union_score'].mean():.1f}, "
          f"pct_hands_on={top['pct_hands_on'].mean():.1f}%, "
          f"pct_screen={top['pct_screen'].mean():.1f}%")
    print(f"    Bottom quartile: mean_union={bottom['mean_union_score'].mean():.1f}, "
          f"pct_hands_on={bottom['pct_hands_on'].mean():.1f}%, "
          f"pct_screen={bottom['pct_screen'].mean():.1f}%")

    # t-test on pct_hands_on between quartiles
    t, p = stats.ttest_ind(top['pct_hands_on'].values, bottom['pct_hands_on'].values, equal_var=False)
    d = (top['pct_hands_on'].mean() - bottom['pct_hands_on'].mean()) / np.sqrt(
        (top['pct_hands_on'].var() + bottom['pct_hands_on'].var()) / 2)
    print(f"    pct_hands_on: t = {t:.2f}, p = {p:.2e}, d = {d:.3f}")

    # t-test on pct_screen
    t, p = stats.ttest_ind(top['pct_screen'].values, bottom['pct_screen'].values, equal_var=False)
    print(f"    pct_screen: t = {t:.2f}, p = {p:.2e}")

print()
print("Done.")
