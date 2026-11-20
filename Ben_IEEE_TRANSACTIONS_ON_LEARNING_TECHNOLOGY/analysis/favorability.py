import re
import pandas as pd
from pathlib import Path
from collections import Counter

BASE = Path(__file__).resolve().parent.parent

# Confirmed modality labels for all 20 surveyed videos.
# In-dataset labels from FusionAnalysis/results/video_classification;
# the remaining 10 confirmed by the author (2026-07-05).
LABELS = {
    # conventional
    "n3TeLpbw_jk": "conventional",
    "XRsJRqipSx8": "conventional",
    "iKZ05QIN8mc": "conventional",
    "umKfROrX0aE": "conventional",
    "3-r5u4Vz3oY": "conventional",
    "j-YCQV4TYWo": "conventional",
    "KcV3AiUxlCI": "conventional",
    "cA9JyEqelC8": "conventional",
    # embodied
    "N5DbiWFjbh8": "embodied",
    "3dQjIeYoIdM": "embodied",
    "KDwqqlqdYFQ": "embodied",
    "9fFjlxUUQZg": "embodied",
    "dV8rpGUoxns": "embodied",
    "cggDZE7gZNw": "embodied",
    "-qCO7NUpLc0": "embodied",
    "f-M58vllFc8": "embodied",
    "4DGZOaQOzJU": "embodied",
    "S-OgcqWoGIY": "embodied",
    "UPNfNQXh6Y4": "embodied",
    "ezTsxAiL-8U": "embodied",
}

def extract_ids(text):
    if not isinstance(text, str):
        return []
    t = text.replace("%3D", "=").replace("%3d", "=").replace("%26", "&")
    return re.findall(r"(?:v=|youtu\.be/)([A-Za-z0-9_-]{11})", t)

df1 = pd.read_excel(BASE / "Video Watching and Reflection Study Form 1 (Responses).xlsx")
df2 = pd.read_excel(BASE / "_ENG3530 Video Watching and Reflection (Responses).xlsx")
df1.columns = ["timestamp", "consent", "favorable", "unfavorable", "why_good", "why_bad", "most_useful", "closing"]
df2.columns = ["timestamp", "name", "favorable", "unfavorable", "why_good", "why_bad", "most_useful"]

fav, unfav = Counter(), Counter()
for df in (df1, df2):
    for v in df["favorable"]:
        fav.update(extract_ids(v))
    for v in df["unfavorable"]:
        unfav.update(extract_ids(v))

unknown = (set(fav) | set(unfav)) - set(LABELS)
assert not unknown, f"unlabeled ids: {unknown}"

rows = []
for vid in sorted(LABELS, key=lambda v: (LABELS[v], -fav[v])):
    rows.append((vid, LABELS[vid], fav[vid], unfav[vid]))

print(f"{'video_id':<14}{'label':<14}{'fav':>5}{'unfav':>7}")
for vid, lab, f, u in rows:
    print(f"{vid:<14}{lab:<14}{f:>5}{u:>7}")

for lab in ("conventional", "embodied"):
    vids = [v for v in LABELS if LABELS[v] == lab]
    F = sum(fav[v] for v in vids)
    U = sum(unfav[v] for v in vids)
    n = len(vids)
    print(f"\n{lab}: {n} videos")
    print(f"  favorable votes:   {F:4d}  ({F/n:.1f} per video)")
    print(f"  unfavorable votes: {U:4d}  ({U/n:.1f} per video)")
    print(f"  favorable share F/(F+U): {F/(F+U)*100:.1f}%")

# total ballots cast
tot_f = sum(fav.values())
tot_u = sum(unfav.values())
print(f"\ntotal favorable selections: {tot_f}, unfavorable: {tot_u}")

# top-voted videos overall
print("\nTop favorable videos:")
for vid, n in fav.most_common(6):
    print(f"  {vid} ({LABELS[vid]}): {n}")
print("Top unfavorable videos:")
for vid, n in unfav.most_common(6):
    print(f"  {vid} ({LABELS[vid]}): {n}")
