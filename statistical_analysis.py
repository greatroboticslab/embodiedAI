"""
Statistical Analysis for ASEE Paper Revision
=============================================
Performs t-tests / Mann-Whitney U tests, computes Cohen's d effect sizes,
confidence intervals, and generates LaTeX-ready results tables.

Data sources:
  - CaptionAnalysis/output/Embodied_metrics.csv   (74 embodied videos)
  - CaptionAnalysis/output/Conventional_metrics.csv (56 conventional videos)
"""

import os
import numpy as np
import pandas as pd
from scipy import stats

# ─────────────────────────────────────────────
# 1.  LOAD DATA
# ─────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "CaptionAnalysis", "output")

emb = pd.read_csv(os.path.join(DATA_DIR, "Embodied_metrics.csv"))
con = pd.read_csv(os.path.join(DATA_DIR, "Conventional_metrics.csv"))

emb["group"] = "Embodied"
con["group"] = "Conventional"

# ─────────────────────────────────────────────
# 2.  COMPUTE DERIVED METRICS
# ─────────────────────────────────────────────
for df in [emb, con]:
    df["caption_density"] = df["caption_words"] / df["duration_s"]          # words/sec
    df["engagement_rate"] = df["comments"] / (df["duration_s"] / 60.0)      # comments/min
    df["corr_efficiency"] = df["corr_n_ge_60"] / df["corr_n"]              # ratio ≥60%

all_data = pd.concat([emb, con], ignore_index=True)

# ─────────────────────────────────────────────
# 3.  STATISTICAL TESTS
# ─────────────────────────────────────────────
METRICS = {
    "corr_avg":          "Avg. Correlation Score",
    "caption_words":     "Caption Word Count",
    "comments":          "Comment Count",
    "duration_s":        "Video Duration (s)",
    "caption_density":   "Caption Density (words/s)",
    "engagement_rate":   "Engagement Rate (comm/min)",
    "corr_efficiency":   "Correlation Efficiency",
}

ALPHA = 0.05


def cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Cohen's d for two independent samples."""
    nx, ny = len(x), len(y)
    pooled_std = np.sqrt(((nx - 1) * x.std(ddof=1)**2 +
                          (ny - 1) * y.std(ddof=1)**2) / (nx + ny - 2))
    if pooled_std == 0:
        return 0.0
    return (x.mean() - y.mean()) / pooled_std


def rank_biserial_r(U: float, n1: int, n2: int) -> float:
    """Rank-biserial correlation as effect size for Mann-Whitney U."""
    return 1 - (2 * U) / (n1 * n2)


def mean_diff_ci(x: np.ndarray, y: np.ndarray, confidence: float = 0.95):
    """95% CI for the difference of means (Welch approximation)."""
    diff = x.mean() - y.mean()
    se = np.sqrt(x.var(ddof=1) / len(x) + y.var(ddof=1) / len(y))
    # Welch–Satterthwaite degrees of freedom
    num = (x.var(ddof=1) / len(x) + y.var(ddof=1) / len(y))**2
    denom = ((x.var(ddof=1) / len(x))**2 / (len(x) - 1) +
             (y.var(ddof=1) / len(y))**2 / (len(y) - 1))
    df = num / denom if denom != 0 else len(x) + len(y) - 2
    t_crit = stats.t.ppf((1 + confidence) / 2, df)
    return diff - t_crit * se, diff + t_crit * se


def interpret_d(d: float) -> str:
    """Interpret Cohen's d magnitude."""
    d_abs = abs(d)
    if d_abs < 0.2:
        return "Negligible"
    elif d_abs < 0.5:
        return "Small"
    elif d_abs < 0.8:
        return "Medium"
    else:
        return "Large"


def sig_label(p: float) -> str:
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return "n.s."


results = []

print("=" * 90)
print("STATISTICAL ANALYSIS: Embodied vs Conventional Learning Videos")
print("=" * 90)
print(f"  Embodied videos:     n = {len(emb)}")
print(f"  Conventional videos: n = {len(con)}")
print(f"  Significance level:  α = {ALPHA}")
print()

for col, label in METRICS.items():
    x_emb = emb[col].dropna().values
    x_con = con[col].dropna().values

    # ── Normality check ──
    _, p_norm_emb = stats.shapiro(x_emb) if len(x_emb) <= 5000 else (0, 0)
    _, p_norm_con = stats.shapiro(x_con) if len(x_con) <= 5000 else (0, 0)
    is_normal = (p_norm_emb > ALPHA) and (p_norm_con > ALPHA)

    # ── Select test ──
    if is_normal:
        test_name = "Independent t-test"
        stat_val, p_val = stats.ttest_ind(x_emb, x_con, equal_var=False)  # Welch's
        d = cohens_d(x_emb, x_con)
        effect_size = d
        effect_label = f"Cohen's d = {d:.3f} ({interpret_d(d)})"
    else:
        test_name = "Mann-Whitney U"
        stat_val, p_val = stats.mannwhitneyu(x_emb, x_con, alternative='two-sided')
        r = rank_biserial_r(stat_val, len(x_emb), len(x_con))
        effect_size = r
        d = cohens_d(x_emb, x_con)  # also report Cohen's d for reference
        effect_label = f"r = {r:.3f}, Cohen's d = {d:.3f} ({interpret_d(d)})"

    ci_lo, ci_hi = mean_diff_ci(x_emb, x_con)

    results.append({
        "Metric": label,
        "Emb Mean": np.mean(x_emb),
        "Emb SD": np.std(x_emb, ddof=1),
        "Con Mean": np.mean(x_con),
        "Con SD": np.std(x_con, ddof=1),
        "Test": test_name,
        "Statistic": stat_val,
        "p-value": p_val,
        "Cohen_d": d,
        "Effect Interpretation": interpret_d(d),
        "CI_lo": ci_lo,
        "CI_hi": ci_hi,
        "Sig": sig_label(p_val),
    })

    print(f"── {label} ──")
    print(f"   Embodied:      M = {np.mean(x_emb):.4f},  SD = {np.std(x_emb, ddof=1):.4f}")
    print(f"   Conventional:  M = {np.mean(x_con):.4f},  SD = {np.std(x_con, ddof=1):.4f}")
    print(f"   Normality:     Emb p = {p_norm_emb:.4f}, Con p = {p_norm_con:.4f}  →  {'Normal' if is_normal else 'Non-normal'}")
    print(f"   Test:          {test_name}")
    print(f"   Statistic:     {stat_val:.4f}")
    print(f"   p-value:       {p_val:.6f}  {sig_label(p_val)}")
    print(f"   Effect size:   {effect_label}")
    print(f"   95% CI (diff): [{ci_lo:.4f}, {ci_hi:.4f}]")
    print()

# ─────────────────────────────────────────────
# 4.  YOUTUBE COMMENT TABLE ANALYSIS (20 videos from LaTeX tables)
# ─────────────────────────────────────────────
print("=" * 90)
print("YOUTUBE COMMENT TABLE ANALYSIS (10 Embodied + 10 Conventional)")
print("=" * 90)

# Data from LaTeX tables
yt_emb_total  = np.array([54, 271, 128, 24, 623, 214, 559, 1332, 248, 506])
yt_con_total  = np.array([980, 6, 3, 1408, 129, 109, 54, 15, 29, 20])

# LLaVA correlated
yt_emb_llava  = np.array([44, 180, 67, 19, 465, 129, 448, 899, 147, 266])
yt_con_llava  = np.array([615, 1, 1, 422, 75, 83, 47, 11, 13, 5])

# MiniCPM correlated
yt_emb_mini   = np.array([32, 182, 80, 21, 438, 100, 417, 829, 128, 290])
yt_con_mini   = np.array([809, 3, 1, 700, 85, 76, 37, 10, 15, 8])

# Integrated correlated
yt_emb_integ  = np.array([49, 240, 96, 23, 551, 166, 486, 1090, 199, 398])
yt_con_integ  = np.array([834, 4, 1, 911, 82, 81, 46, 11, 16, 5])

# Sentiment (Integrated) - Embodied
yt_emb_pos = np.array([33, 124, 41, 16, 318, 113, 141, 515, 139, 224])
yt_emb_neu = np.array([16, 89, 42, 7, 204, 39, 273, 514, 44, 121])
yt_emb_neg = np.array([0, 27, 13, 0, 29, 14, 72, 61, 16, 53])

# Sentiment (Integrated) - Conventional
yt_con_pos = np.array([215, 3, 1, 513, 35, 39, 17, 7, 8, 5])
yt_con_neu = np.array([310, 1, 0, 313, 47, 30, 20, 4, 7, 0])
yt_con_neg = np.array([309, 0, 0, 85, 10, 12, 9, 0, 1, 0])

# Correlation ratios
def make_ratio(corr, total):
    return corr / total

ratios = {
    "Corr. Ratio (LLaVA)":      (make_ratio(yt_emb_llava, yt_emb_total), make_ratio(yt_con_llava, yt_con_total)),
    "Corr. Ratio (MiniCPM)":     (make_ratio(yt_emb_mini,  yt_emb_total), make_ratio(yt_con_mini,  yt_con_total)),
    "Corr. Ratio (Integrated)":  (make_ratio(yt_emb_integ, yt_emb_total), make_ratio(yt_con_integ, yt_con_total)),
    "Positive Sent. Ratio":      (yt_emb_pos / yt_emb_integ, yt_con_pos / yt_con_integ),
    "Neutral Sent. Ratio":       (yt_emb_neu / yt_emb_integ, yt_con_neu / yt_con_integ),
    "Negative Sent. Ratio":      (yt_emb_neg / yt_emb_integ, yt_con_neg / yt_con_integ),
}

yt_results = []

for label, (x_e, x_c) in ratios.items():
    x_e = x_e[~np.isnan(x_e)]
    x_c = x_c[~np.isnan(x_c)]

    _, p_norm_e = stats.shapiro(x_e)
    _, p_norm_c = stats.shapiro(x_c)
    is_normal = (p_norm_e > ALPHA) and (p_norm_c > ALPHA)

    if is_normal:
        test_name = "Independent t-test"
        stat_val, p_val = stats.ttest_ind(x_e, x_c, equal_var=False)
    else:
        test_name = "Mann-Whitney U"
        stat_val, p_val = stats.mannwhitneyu(x_e, x_c, alternative='two-sided')

    d = cohens_d(x_e, x_c)
    ci_lo, ci_hi = mean_diff_ci(x_e, x_c)

    yt_results.append({
        "Metric": label,
        "Emb Mean": np.mean(x_e),
        "Emb SD": np.std(x_e, ddof=1),
        "Con Mean": np.mean(x_c),
        "Con SD": np.std(x_c, ddof=1),
        "Test": test_name,
        "Statistic": stat_val,
        "p-value": p_val,
        "Cohen_d": d,
        "Effect Interpretation": interpret_d(d),
        "CI_lo": ci_lo,
        "CI_hi": ci_hi,
        "Sig": sig_label(p_val),
    })

    print(f"── {label} ──")
    print(f"   Embodied:      M = {np.mean(x_e):.4f},  SD = {np.std(x_e, ddof=1):.4f}")
    print(f"   Conventional:  M = {np.mean(x_c):.4f},  SD = {np.std(x_c, ddof=1):.4f}")
    print(f"   Test:          {test_name}")
    print(f"   p-value:       {p_val:.6f}  {sig_label(p_val)}")
    print(f"   Cohen's d:     {d:.3f} ({interpret_d(d)})")
    print(f"   95% CI (diff): [{ci_lo:.4f}, {ci_hi:.4f}]")
    print()


# ─────────────────────────────────────────────
# 5.  SAVE RESULTS
# ─────────────────────────────────────────────
all_results = results + yt_results
df_results = pd.DataFrame(all_results)
output_path = os.path.join(SCRIPT_DIR, "statistical_results.csv")
df_results.to_csv(output_path, index=False)
print(f"Results saved to: {output_path}")
print()

# ─────────────────────────────────────────────
# 6.  GENERATE LATEX TABLE
# ─────────────────────────────────────────────
print("=" * 90)
print("LATEX TABLE (copy-paste into paper)")
print("=" * 90)

latex_lines = []
latex_lines.append(r"\begin{table}[H]")
latex_lines.append(r"    \centering")
latex_lines.append(r"    \caption{Statistical Comparison of Embodied vs.\ Conventional Videos}")
latex_lines.append(r"    \label{tab:stat_tests}")
latex_lines.append(r"    \scriptsize")
latex_lines.append(r"    \begin{tabular}{l c c c c c c}")
latex_lines.append(r"        \toprule")
latex_lines.append(r"        \textbf{Metric} & \textbf{Emb.\ Mean$\pm$SD} & \textbf{Con.\ Mean$\pm$SD} & \textbf{Test} & \textbf{$p$-value} & \textbf{Cohen's $d$} & \textbf{Sig.} \\")
latex_lines.append(r"        \midrule")

# User study results
latex_lines.append(r"        \multicolumn{7}{l}{\textbf{User Study (130 Videos)}} \\")
latex_lines.append(r"        \midrule")
for r in results:
    emb_str = f"${r['Emb Mean']:.2f} \\pm {r['Emb SD']:.2f}$"
    con_str = f"${r['Con Mean']:.2f} \\pm {r['Con SD']:.2f}$"
    test_short = "t" if "t-test" in r["Test"] else "U"
    p_str = f"${r['p-value']:.4f}$" if r['p-value'] >= 0.0001 else "$<0.0001$"
    d_str = f"${r['Cohen_d']:.3f}$"
    interp = r["Effect Interpretation"]
    sig = r["Sig"]
    metric_escaped = r["Metric"].replace("%", r"\%")
    latex_lines.append(f"        {metric_escaped} & {emb_str} & {con_str} & {test_short} & {p_str} & {d_str} ({interp}) & {sig} \\\\")

latex_lines.append(r"        \midrule")
latex_lines.append(r"        \multicolumn{7}{l}{\textbf{YouTube Comment Analysis (20 Videos)}} \\")
latex_lines.append(r"        \midrule")
for r in yt_results:
    emb_str = f"${r['Emb Mean']:.4f} \\pm {r['Emb SD']:.4f}$"
    con_str = f"${r['Con Mean']:.4f} \\pm {r['Con SD']:.4f}$"
    test_short = "t" if "t-test" in r["Test"] else "U"
    p_str = f"${r['p-value']:.4f}$" if r['p-value'] >= 0.0001 else "$<0.0001$"
    d_str = f"${r['Cohen_d']:.3f}$"
    interp = r["Effect Interpretation"]
    sig = r["Sig"]
    metric_escaped = r["Metric"].replace("%", r"\%")
    latex_lines.append(f"        {metric_escaped} & {emb_str} & {con_str} & {test_short} & {p_str} & {d_str} ({interp}) & {sig} \\\\")

latex_lines.append(r"        \bottomrule")
latex_lines.append(r"    \end{tabular}")
latex_lines.append(r"\end{table}")

latex_output = "\n".join(latex_lines)
print(latex_output)
print()

# Save LaTeX snippet to file
latex_path = os.path.join(SCRIPT_DIR, "statistical_table.tex")
with open(latex_path, "w", encoding="utf-8") as f:
    f.write(latex_output)
print(f"LaTeX table saved to: {latex_path}")
