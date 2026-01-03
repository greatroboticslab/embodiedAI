#!/usr/bin/env python3
"""
Collect metrics for YouTube comments analysis.
Generates a CSV with:
- Video Metadata (ID, Label, Duration, Total Comments)
- Global Correlation (Unique correlated comments, % Correlated)
- Engagement (Comments / Minute)
- Top Topic Stats (Topic with most high-scoring comments > 60)

Usage:
    python collect_youtube_metrics.py
"""

import os
import json
import csv
import glob
from collections import defaultdict

# Base paths relative to this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR)) # embodiedAI
TRANSCRIPT_ROOT = os.path.join(PROJECT_ROOT, "TranscriptAnalysis")
RESULTS_YOUTUBE = os.path.join(TRANSCRIPT_ROOT, "results_youtube")
VIDEO_ANALYSIS_COMMENTS = os.path.join(PROJECT_ROOT, "VideoAnalysis", "data", "comments")
METRICS_DIR = os.path.join(TRANSCRIPT_ROOT, "results", "metrics")
OUTPUT_DIR = os.path.join(RESULTS_YOUTUBE, "metrics")
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "youtube_metrics.csv")

MIN_SCORE = 60

def load_durations():
    """Load duration_seconds from Conventional and Embodied metrics JSONs."""
    durations = {}
    for filename in ["Conventional_metrics.json", "Embodied_metrics.json"]:
        path = os.path.join(METRICS_DIR, filename)
        if not os.path.exists(path):
            print(f"[WARN] Metrics file not found: {path}")
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                for video in data.get("videos", []):
                    vid = video.get("video_id")
                    dur = video.get("duration_seconds")
                    if dur is None: dur = 0
                    if vid:
                        durations[vid] = dur
        except Exception as e:
            print(f"[ERROR] Reading {filename}: {e}")
    return durations

def get_total_comments(video_id, label_lower):
    """
    Count total comments from the source JSON file.
    Path: .../VideoAnalysis/data/comments/<label>/<video_id>.json
    """
    # map label 'conv' -> 'conventional', 'emb' -> 'embodied' if needed, 
    # but directory listing showed 'embodied' and likely 'conventional' (or 'con'?)
    # Let's try likely folder names.
    
    # Based on previous listing: results_youtube has 'correlation_conventional' and 'correlation_embodied'
    # VideoAnalysis/data/comments likely has 'conventional' and 'embodied' or 'con'/'emb'
    # We will try both.
    
    candidates = []
    if label_lower.startswith("conv"):
        candidates.append(os.path.join(VIDEO_ANALYSIS_COMMENTS, "conventional", f"{video_id}.json"))
        candidates.append(os.path.join(VIDEO_ANALYSIS_COMMENTS, "con", f"{video_id}.json"))
    else:
        candidates.append(os.path.join(VIDEO_ANALYSIS_COMMENTS, "embodied", f"{video_id}.json"))
        candidates.append(os.path.join(VIDEO_ANALYSIS_COMMENTS, "emb", f"{video_id}.json"))
        
    for p in candidates:
        if os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    # It might be JSONL or a JSON list
                    content = f.read().strip()
                    if not content: return 0
                    
                    # Try list
                    try:
                        data = json.loads(content)
                        if isinstance(data, list): return len(data)
                        if isinstance(data, dict): return 1 # unlikely but possible
                    except:
                        # Try JSONL
                        return len(content.splitlines())
            except Exception as e:
                print(f"[ERR] Error reading comments for {video_id}: {e}")
                return 0
    
    print(f"[WARN] Source comment file not found for {video_id} in {candidates}")
    return 0

def process_video(video_id, label, durations, results_dir):
    # 1. Get Duration
    duration = durations.get(video_id, 0)
    
    # 2. Get Total Comments
    label_lower = "conventional" if label == "Conv" else "embodied"
    total_comments = get_total_comments(video_id, label_lower)
    
    if total_comments == 0:
        return None 

    # 3. Analyze Correlation Results
    # Locate result file: <video_id>_topiccorr_*.results.jsonl
    result_files = glob.glob(os.path.join(results_dir, video_id, "*.results.jsonl"))
    if not result_files:
        print(f"[SKIP] No result file for {video_id}")
        return None
    
    # Use the first one found (usually only one)
    res_path = result_files[0]
    
    unique_correlated = set() # (url, comment)
    
    # For Top Topic
    topic_stats = defaultdict(lambda: {"count_high": 0, "sum_score_high": 0})
    
    with open(res_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
                tid = rec.get("topic_id")
                candidates = rec.get("candidates", [])
                
                for c in candidates:
                    is_corr = c.get("correlated", False)
                    score = int(c.get("score", 0))
                    cid = (c.get("url", ""), c.get("comment", ""))
                    
                    if is_corr:
                        unique_correlated.add(cid)
                    
                    # Top Topic Logic: correlated AND score > MIN_SCORE
                    if is_corr and score > MIN_SCORE:
                        topic_stats[tid]["count_high"] += 1
                        topic_stats[tid]["sum_score_high"] += score
                        
            except:
                continue

    # 4. Calculate Metrics
    
    # -- Global Correlation
    unique_corr_count = len(unique_correlated)
    pct_correlated = (unique_corr_count / total_comments) if total_comments > 0 else 0
    
    # -- Engagement (Comments / Minute)
    # Duration is in seconds
    duration_min = duration / 60.0 if duration > 0 else 1.0
    engagement_rate = total_comments / duration_min
    
    # -- Top Topic
    # Find topic with max 'count_high'
    top_tid = "None"
    top_count = 0
    top_avg = 0.0
    
    if topic_stats:
        # Sort by count_high desc
        sorted_topics = sorted(topic_stats.items(), key=lambda x: x[1]["count_high"], reverse=True)
        best = sorted_topics[0]
        
        top_tid = best[0]
        top_count = best[1]["count_high"]
        sum_score = best[1]["sum_score_high"]
        if top_count > 0:
            top_avg = sum_score / top_count
            
    top_topic_pct = (top_count / total_comments) if total_comments > 0 else 0
    
    return {
        "Label": label,
        "VideoID": video_id,
        "DurationSec": duration,
        "TotalComments": total_comments,
        "UniqueCorrelated": unique_corr_count,
        "PctCorrelated": pct_correlated,
        "Engagement": engagement_rate,
        "TopTopicID": top_tid,
        "TopTopicCount": top_count,
        "TopTopicAvgScore": top_avg,
        "TopTopicPct": top_topic_pct
    }

def main():
    print(f"Loading durations from: {METRICS_DIR}")
    durations = load_durations()
    print(f"Loaded {len(durations)} video durations.")
    
    rows = []
    
    # Process Conventional
    conv_dir = os.path.join(RESULTS_YOUTUBE, "correlation_conventional")
    if os.path.exists(conv_dir):
        print("Processing Conventional...")
        for vid in os.listdir(conv_dir):
            if os.path.isdir(os.path.join(conv_dir, vid)):
                row = process_video(vid, "Conv", durations, conv_dir)
                if row: rows.append(row)
    
    # Process Embodied
    emb_dir = os.path.join(RESULTS_YOUTUBE, "correlation_embodied")
    if os.path.exists(emb_dir):
        print("Processing Embodied...")
        for vid in os.listdir(emb_dir):
            if os.path.isdir(os.path.join(emb_dir, vid)):
                row = process_video(vid, "Emb", durations, emb_dir)
                if row: rows.append(row)
                
    # Write CSV
    headers = [
        "Label", "VideoID", "DurationSec", "TotalComments", 
        "UniqueCorrelated", "PctCorrelated", "Engagement", 
        "TopTopicID", "TopTopicCount", "TopTopicAvgScore", "TopTopicPct"
    ]
    
    print(f"Writing metrics to {OUTPUT_CSV}...")
    with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)
        
    print("Done.")

if __name__ == "__main__":
    main()
