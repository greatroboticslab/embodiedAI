#!/usr/bin/env python3
"""
Collect detailed YouTube metrics per video and save as JSON.
Format matches embodiment_project_collect_metrics.py style.

Metrics:
- PctCorrelated: (UniqueCommentsCorrelated / TotalComments)
- Engagement: (TotalComments / Duration(min))
- TopTopicPct: (CommentsOnTopTopic / TotalComments)
- TopTopicAvgScore: (AvgScore of CommentsOnTopTopic > 60)
"""

import os
import json
import argparse
import glob
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict
from collections import defaultdict

@dataclass
class VideoMetrics:
    video_id: str
    duration_seconds: float
    total_comments: int
    unique_correlated: int
    pct_correlated: float
    engagement: float
    top_topic_id: str
    top_topic_count: int
    top_topic_avg_score: float
    top_topic_pct: float

@dataclass
class Aggregate:
    label: str
    num_videos: int
    avg_pct_correlated: float
    avg_engagement: float
    avg_top_topic_pct: float
    avg_top_topic_avg_score: float

def load_durations(metrics_dir):
    durations = {}
    for filename in ["Conventional_metrics.json", "Embodied_metrics.json"]:
        path = os.path.join(metrics_dir, filename)
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for v in data.get("videos", []):
                        vid = v.get("video_id")
                        dur = v.get("duration_seconds") 
                        if dur is not None:
                            durations[vid] = dur
            except Exception as e:
                print(f"[WARN] Failed to load durations from {path}: {e}")
    return durations

def get_total_comments(video_id, comments_root):
    # Try <comments_root>/<video_id>.json
    p = os.path.join(comments_root, f"{video_id}.json")
    if os.path.exists(p):
        try:
            with open(p, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if not content: return 0
                try:
                    data = json.loads(content)
                    if isinstance(data, list): return len(data)
                    if isinstance(data, dict): return 1
                except:
                    return len(content.splitlines())
        except:
            return 0
    return 0

def process_video(video_id, duration, comments_root, correlation_root, min_score=60):
    total = get_total_comments(video_id, comments_root)
    
    unique_corr = set()
    topic_stats = defaultdict(lambda: {"count_high": 0, "sum_high": 0})
    
    # Read correlation results
    res_pattern = os.path.join(correlation_root, video_id, "*.results.jsonl")
    files = glob.glob(res_pattern)
    if files:
        # Use first file
        try:
            with open(files[0], "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                        tid = rec.get("topic_id")
                        for c in rec.get("candidates", []):
                            entry = (c.get("url"), c.get("comment"))
                            score = int(c.get("score", 0))
                            if c.get("correlated"):
                                unique_corr.add(entry)
                                if score > min_score:
                                    topic_stats[tid]["count_high"] += 1
                                    topic_stats[tid]["sum_high"] += score
                    except:
                        pass
        except:
            pass

    # Unique Correlated
    u_count = len(unique_corr)
    pct_corr = u_count / total if total > 0 else 0.0
    
    # Engagement
    dur_min = duration / 60.0 if duration > 0 else 1.0
    eng = total / dur_min
    
    # Top Topic
    top_tid = "None"
    top_count = 0
    top_avg = 0.0
    
    if topic_stats:
        best = max(topic_stats.items(), key=lambda x: x[1]["count_high"])
        top_tid = best[0]
        top_count = best[1]["count_high"]
        s_sum = best[1]["sum_high"]
        if top_count > 0:
            top_avg = s_sum / top_count
            
    top_pct = top_count / total if total > 0 else 0.0
    
    return VideoMetrics(
        video_id=video_id,
        duration_seconds=duration,
        total_comments=total,
        unique_correlated=u_count,
        pct_correlated=pct_corr,
        engagement=eng,
        top_topic_id=top_tid,
        top_topic_count=top_count,
        top_topic_avg_score=top_avg,
        top_topic_pct=top_pct
    )

def mean(vals):
    if not vals: return 0.0
    return sum(vals) / len(vals)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--comments_root", required=True)
    parser.add_argument("--correlation_root", required=True)
    parser.add_argument("--metrics_root", required=True, help="Path to folder containing Conventional_metrics.json for durations")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--label", required=True, help="Conventional or Embodied")
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    durations = load_durations(args.metrics_root)
    
    # Discover videos in correlation root
    # correlation_root typically has subfolders <video_id>
    video_ids = [d for d in os.listdir(args.correlation_root) if os.path.isdir(os.path.join(args.correlation_root, d))]
    
    metrics_list = []
    
    print(f"Processing {len(video_ids)} videos for {args.label}...")
    
    for vid in video_ids:
        dur = durations.get(vid, 0)
        # Assuming comments_root has <video_id>.json directly
        m = process_video(vid, dur, args.comments_root, args.correlation_root)
        if m.total_comments > 0:
             metrics_list.append(m)
             
    # Aggregate
    agg = Aggregate(
        label=args.label,
        num_videos=len(metrics_list),
        avg_pct_correlated=mean([m.pct_correlated for m in metrics_list]),
        avg_engagement=mean([m.engagement for m in metrics_list]),
        avg_top_topic_pct=mean([m.top_topic_pct for m in metrics_list]),
        avg_top_topic_avg_score=mean([m.top_topic_avg_score for m in metrics_list])
    )
    
    out_file = os.path.join(args.out_dir, f"{args.label}_youtube_metrics.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "label": args.label,
            "aggregate": asdict(agg),
            "videos": [asdict(m) for m in metrics_list]
        }, f, indent=2)
        
    print(f"Saved {out_file}")

if __name__ == "__main__":
    main()
