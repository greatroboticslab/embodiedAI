import re
import pandas as pd
from pathlib import Path
from collections import Counter

BASE = Path(__file__).resolve().parent.parent
ROOT = BASE.parent  # D:\Projects\embodiedAI

# Video ID extraction from messy URL text (handles url-encoded %3D forms too)
def extract_ids(text):
    if not isinstance(text, str):
        return []
    t = text.replace("%3D", "=").replace("%3d", "=").replace("%26", "&")
    ids = re.findall(r"(?:v=|youtu\.be/)([A-Za-z0-9_-]{11})", t)
    return ids

df1 = pd.read_excel(BASE / "Video Watching and Reflection Study Form 1 (Responses).xlsx")
df2 = pd.read_excel(BASE / "_ENG3530 Video Watching and Reflection (Responses).xlsx")
df1.columns = ["timestamp", "consent", "favorable", "unfavorable", "why_good", "why_bad", "most_useful", "closing"]
df2.columns = ["timestamp", "name", "favorable", "unfavorable", "why_good", "why_bad", "most_useful"]

fav_ids, unfav_ids = Counter(), Counter()
for df in (df1, df2):
    for v in df["favorable"]:
        fav_ids.update(extract_ids(v))
    for v in df["unfavorable"]:
        unfav_ids.update(extract_ids(v))

all_ids = set(fav_ids) | set(unfav_ids)
print(f"Unique video IDs mentioned: {len(all_ids)}")

# Load classification from the main project
conv_dir = ROOT / "FusionAnalysis/results/video_classification/conventional"
emb_dir = ROOT / "FusionAnalysis/results/video_classification/embodied"
conv_ids = {p.stem for p in conv_dir.glob("*.json")} if conv_dir.exists() else set()
emb_ids = {p.stem for p in emb_dir.glob("*.json")} if emb_dir.exists() else set()
print(f"Dataset: {len(conv_ids)} conventional, {len(emb_ids)} embodied")

# Subgroup info from master CSV
master = ROOT / "TranscriptAnalysis/results/gemini_analysis/master_analysis.csv"
sub = {}
if master.exists():
    m = pd.read_csv(master)
    print("master cols:", list(m.columns)[:15])
    idcol = "video_id" if "video_id" in m.columns else m.columns[0]
    if "subgroup" in m.columns:
        sub = dict(zip(m[idcol], m["subgroup"]))

def cat(vid):
    if vid in conv_ids:
        return "conventional"
    if vid in emb_ids:
        return f"embodied ({sub.get(vid, '?')})"
    return "NOT IN DATASET"

print("\n%-14s %-30s fav  unfav" % ("video_id", "category"))
for vid in sorted(all_ids, key=lambda v: (cat(v), -fav_ids[v])):
    print("%-14s %-30s %3d  %3d" % (vid, cat(vid), fav_ids[vid], unfav_ids[vid]))

# Aggregate by category
agg = Counter()
for vid, n in fav_ids.items():
    agg[("fav", cat(vid).split(" ")[0])] += n
for vid, n in unfav_ids.items():
    agg[("unfav", cat(vid).split(" ")[0])] += n
print("\nAggregate votes:")
for k, v in sorted(agg.items()):
    print(k, v)
