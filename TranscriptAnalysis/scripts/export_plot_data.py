#!/usr/bin/env python3
"""
Read Conventional and Embodied Transcript metrics JSON files, calculate Mean and SEM,
and export a summary CSV for LaTeX PGFPlots.
"""

import json
import csv
import math
import statistics
import argparse
import os

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_stats(data_list):
    if not data_list:
        return 0.0, 0.0
    # Filter out None values just in case
    data_list = [x for x in data_list if x is not None]
    if not data_list:
        return 0.0, 0.0
        
    m = statistics.mean(data_list)
    try:
        s = statistics.stdev(data_list) if len(data_list) > 1 else 0.0
    except statistics.StatisticsError:
        s = 0.0
    sem = s / math.sqrt(len(data_list))
    return m, sem

def get_metrics_summary(data_dict, label):
    videos = data_dict.get("videos", [])
    
    # Extract raw lists
    transcript_words = []
    topics = []
    comments = []
    correlations = []
    durations = []
    
    # Derived
    word_density = []   # words/sec
    topic_density = []  # topics/sec
    engagement = []     # comments/sec
    efficiency = []     # fraction of high-conf correlations

    for v in videos:
        wc = v.get("transcript_word_count", 0)
        tc = v.get("topics_count", 0)
        cc = v.get("comments_count", 0)
        dur = v.get("duration_seconds", 0)
        corr_val = v.get("corr_avg")
        c_n = v.get("corr_n", 0)
        c_ge = v.get("corr_n_ge_threshold", 0)
        
        transcript_words.append(wc)
        topics.append(tc)
        comments.append(cc)
        if corr_val is not None:
            correlations.append(corr_val)
        if dur is not None and dur > 0:
            durations.append(dur)
            # Convert to per-minute for readability
            word_density.append((wc / dur) * 60)
            engagement.append((cc / dur) * 60)
            topic_density.append((tc / dur) * 60)
        
        if c_n > 0:
            efficiency.append(c_ge / c_n)
        else:
            efficiency.append(0.0)

    return {
        "label": label,
        "TranscriptWords": get_stats(transcript_words),
        "Topics": get_stats(topics),
        "Comments": get_stats(comments),
        "Correlation": get_stats(correlations),
        "Duration": get_stats(durations),
        "WordDensity": get_stats(word_density),
        "TopicDensity": get_stats(topic_density),
        "Engagement": get_stats(engagement),
        "Efficiency": get_stats(efficiency)
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--conventional', required=True)
    parser.add_argument('--embodied', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    conv = load_json(args.conventional)
    emb = load_json(args.embodied)

    s1 = get_metrics_summary(conv, "Conv")
    s2 = get_metrics_summary(emb, "Emb")

    # Columns
    # Label, 
    # CorrMean, CorrSEM, 
    # WordMean, WordSEM, 
    # TopicMean, TopicSEM, 
    # CommMean, CommSEM, 
    # DurMean, DurSEM, 
    # DensMean, DensSEM,   (Words/min)
    # TopDensMean, TopDensSEM, (Topics/min)
    # EngMean, EngSEM, (Comments/min)
    # EffMean, EffSEM

    headers = [
        "Label",
        "CorrMean", "CorrSEM",
        "WordMean", "WordSEM",
        "TopicMean", "TopicSEM",
        "CommMean", "CommSEM",
        "DurMean", "DurSEM",
        "DensMean", "DensSEM",
        "TopDensMean", "TopDensSEM",
        "EngMean", "EngSEM",
        "EffMean", "EffSEM"
    ]

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(headers)
        
        for s in [s1, s2]:
            row = [
                s["label"],
                f"{s['Correlation'][0]:.4f}", f"{s['Correlation'][1]:.4f}",
                f"{s['TranscriptWords'][0]:.4f}", f"{s['TranscriptWords'][1]:.4f}",
                f"{s['Topics'][0]:.4f}", f"{s['Topics'][1]:.4f}",
                f"{s['Comments'][0]:.4f}", f"{s['Comments'][1]:.4f}",
                f"{s['Duration'][0]:.4f}", f"{s['Duration'][1]:.4f}",
                f"{s['WordDensity'][0]:.4f}", f"{s['WordDensity'][1]:.4f}",
                f"{s['TopicDensity'][0]:.4f}", f"{s['TopicDensity'][1]:.4f}",
                f"{s['Engagement'][0]:.4f}", f"{s['Engagement'][1]:.4f}",
                f"{s['Efficiency'][0]:.4f}", f"{s['Efficiency'][1]:.4f}",
            ]
            w.writerow(row)
            
    print(f"Exported to {args.output}")

if __name__ == "__main__":
    main()
