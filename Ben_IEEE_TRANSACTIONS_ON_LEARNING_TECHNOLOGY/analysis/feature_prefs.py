import pandas as pd
from pathlib import Path
from collections import Counter

BASE = Path(__file__).resolve().parent.parent

df1 = pd.read_excel(BASE / "Video Watching and Reflection Study Form 1 (Responses).xlsx")
df2 = pd.read_excel(BASE / "_ENG3530 Video Watching and Reflection (Responses).xlsx")
df1.columns = ["timestamp", "consent", "favorable", "unfavorable", "why_good", "why_bad", "most_useful", "closing"]
df2.columns = ["timestamp", "name", "favorable", "unfavorable", "why_good", "why_bad", "most_useful"]

FEATURES = [
    ("Hands-on Experiences", "Hands-on"),
    ("Explanation", "Explanation"),
    ("Diagrams and Charts", "Diagrams"),
    ("Easy to Follow Writting", "Writing"),
    ("Body Language", "Body Language"),
    ("Sentiment", "Sentiment"),
    ("Thoroughness", "Thoroughness"),
    ("Others", "Others"),
]

def tally(series, label):
    n = series.notna().sum()
    c = Counter()
    for v in series.dropna():
        for key, name in FEATURES:
            if key in v:
                c[name] += 1
    print(f"\n{label}: {n} respondents answered the feature question")
    for key, name in FEATURES:
        if c[name]:
            print(f"  {name:<14} {c[name]:3d}  ({100*c[name]/n:.1f}%)")
    return c, n

c1, n1 = tally(df1["most_useful"], "Form 1 (robotics)")
c2, n2 = tally(df2["most_useful"], "Form 2 (ENG3530 speaker design)")

print("\nCombined:")
ntot = n1 + n2
for key, name in FEATURES:
    tot = c1.get(name, 0) + c2.get(name, 0)
    print(f"  {name:<14} {tot:3d}  ({100*tot/ntot:.1f}%)")

# Form 1 consent / duplicates
print("\nForm 1 consent counts:", dict(df1["consent"].value_counts(dropna=False)))
print("Form 1 rows with favorable answered:", df1["favorable"].notna().sum())
print("Form 1 rows with consent NaN but favorable answered:",
      ((df1["consent"].isna()) & (df1["favorable"].notna())).sum())
print("Form 1 rows where consent declined but favorable answered:",
      ((df1["consent"] == "I decline to participate") & (df1["favorable"].notna())).sum())
print("Form 2 unique names:", df2["name"].nunique(), "of", len(df2))
