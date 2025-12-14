#!/usr/bin/env python3
"""
Read Conventional and Embodied metrics JSON files, calculate Mean and SEM,
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

def get_metrics(data, label):
    videos = data.get('videos', [])
    metrics = {
        'corr_avg': [],
        'caption_word_count': [],
        'comments_count': []
    }
    
    # New metrics lists
    durs = []
    densities = []
    engagements = []
    efficiencies = []
    
    for v in videos:
        dur = v.get("duration_seconds", 0)
        wc = v.get("caption_word_count", 0)
        cc = v.get("comments_count", 0)
        c_n = v.get("corr_n", 0)
        c_ge = v.get("corr_n_ge_threshold", 0)

        # Word Density (Words per sec)
        if dur and dur > 0:
            densities.append(wc / dur)
            engagements.append(cc / dur)
        else:
            # If duration is missing/zero, skip or set 0? 
            # Skipping is safer for "Average Density"
            pass
            
        # Efficiency (Ratio 0-1)
        if c_n and c_n > 0:
            efficiencies.append(c_ge / c_n)
        else:
            efficiencies.append(0.0)

    # Calculate basic metrics lists
    corrs = [v.get("corr_avg", 0) for v in videos if v.get("corr_avg") is not None]
    words = [v.get("caption_word_count", 0) for v in videos]
    comms = [v.get("comments_count", 0) for v in videos]
    durs = [v.get("duration_seconds", 0) for v in videos if v.get("duration_seconds") is not None]

    def stats(data):
        if not data: return 0.0, 0.0
        m = statistics.mean(data)
        s = statistics.stdev(data) if len(data) > 1 else 0.0
        sem = s / math.sqrt(len(data))
        return m, sem

    result = {
        "label": label,
        "corr": stats(corrs),
        "word": stats(words),
        "comm": stats(comms),
        "dur": stats(durs),
        "dens": stats(densities),
        "eng": stats(engagements),
        "eff": stats(efficiencies),
    }
    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--conventional', required=True, help='Path to Conventional metrics JSON')
    parser.add_argument('--embodied', required=True, help='Path to Embodied metrics JSON')
    parser.add_argument('--output', required=True, help='Path to output CSV')
    args = parser.parse_args()

    conv_data = load_json(args.conventional)
    emb_data = load_json(args.embodied)

    conv_stats = get_metrics(conv_data, "Conv")
    emb_stats = get_metrics(emb_data, "Emb")

    # Write CSV
    # Columns: Label, 
    # CorrMean, CorrSEM, WordMean, WordSEM, CommMean, CommSEM, 
    # DurMean, DurSEM, DensMean, DensSEM, EngMean, EngSEM, EffMean, EffSEM
    
    header = ["Label", 
              "CorrMean", "CorrSEM", "WordMean", "WordSEM", "CommMean", "CommSEM",
              "DurMean", "DurSEM", "DensMean", "DensSEM", "EngMean", "EngSEM", "EffMean", "EffSEM"]
    
    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        
        for s in [conv_stats, emb_stats]:
            row = [
                s["label"],
                f"{s['corr'][0]:.4f}", f"{s['corr'][1]:.4f}",
                f"{s['word'][0]:.4f}", f"{s['word'][1]:.4f}",
                f"{s['comm'][0]:.4f}", f"{s['comm'][1]:.4f}",
                f"{s['dur'][0]:.4f}",  f"{s['dur'][1]:.4f}",
                f"{s['dens'][0]:.4f}", f"{s['dens'][1]:.4f}",
                f"{s['eng'][0]:.4f}",  f"{s['eng'][1]:.4f}",
                f"{s['eff'][0]:.4f}",  f"{s['eff'][1]:.4f}",
            ]
            writer.writerow(row)
    
    print(f"Exported summary to {args.output}")

if __name__ == "__main__":
    main()
