import pandas as pd
from pathlib import Path

pd.set_option("display.max_colwidth", None)
BASE = Path(__file__).resolve().parent.parent

f1 = BASE / "Video Watching and Reflection Study Form 1 (Responses).xlsx"
f2 = BASE / "_ENG3530 Video Watching and Reflection (Responses).xlsx"

df1 = pd.read_excel(f1)
df2 = pd.read_excel(f2)

# Short aliases
df1.columns = ["timestamp", "consent", "favorable", "unfavorable", "why_good", "why_bad", "most_useful", "closing"]
df2.columns = ["timestamp", "name", "favorable", "unfavorable", "why_good", "why_bad", "most_useful"]

print("########## FORM 1 (Study Form 1, n=%d) ##########" % len(df1))
print("\n-- consent values --")
print(df1["consent"].value_counts(dropna=False))
print("\n-- most_useful values --")
print(df1["most_useful"].value_counts(dropna=False))
print("\n-- favorable (first 5 rows) --")
for i, v in enumerate(df1["favorable"].head(5)):
    print(f"[{i}] {v!r}\n")
print("\n-- unfavorable (first 3 rows) --")
for i, v in enumerate(df1["unfavorable"].head(3)):
    print(f"[{i}] {v!r}\n")
print("\n-- why_good (first 3 rows) --")
for i, v in enumerate(df1["why_good"].head(3)):
    print(f"[{i}] {v!r}\n")
print("\n-- why_bad (first 2 rows) --")
for i, v in enumerate(df1["why_bad"].head(2)):
    print(f"[{i}] {v!r}\n")

print("\n\n########## FORM 2 (ENG3530, n=%d) ##########" % len(df2))
print("\n-- most_useful values --")
print(df2["most_useful"].value_counts(dropna=False))
print("\n-- favorable (first 5 rows) --")
for i, v in enumerate(df2["favorable"].head(5)):
    print(f"[{i}] {v!r}\n")
print("\n-- why_good (first 3 rows) --")
for i, v in enumerate(df2["why_good"].head(3)):
    print(f"[{i}] {v!r}\n")
