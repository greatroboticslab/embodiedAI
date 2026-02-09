#!/usr/bin/env python3
"""
collect_fusion_metrics.py

Collects comprehensive metrics from FusionAnalysis correlation results and generates Excel files.
Processes correlation JSON files to extract video-level metrics including:
- Video metadata (duration, segments, comments)
- Visual correlation statistics
- Transcript correlation statistics
- Union metrics (visual OR transcript)

Usage:
    python collect_fusion_metrics.py --label embodied --output ../results/fusion_metrics_embodied.xlsx
    python collect_fusion_metrics.py --label conventional --output ../results/fusion_metrics_conventional.xlsx
    python collect_fusion_metrics.py --label both --output_dir ../results
"""

import os
import json
import argparse
import pandas as pd
from typing import Dict, List, Any
import numpy as np
from pathlib import Path


def load_json(path: str) -> Dict:
    """Load JSON file safely."""
    if not os.path.exists(path):
        return {}
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"[ERR] Failed to load {path}: {e}")
        return {}


def calculate_video_metrics(video_data: Dict, score_threshold: int = 60) -> Dict[str, Any]:
    """
    Calculate comprehensive metrics for a single video from its correlation data.
    
    Args:
        video_data: Loaded JSON data from segment_correlation.json file
        score_threshold: Score threshold for "high score" classification (default: 60)
    
    Returns:
        Dictionary containing all calculated metrics
    """
    metrics = {}
    
    # Basic video info
    metrics['video_id'] = video_data.get('video_id', '')
    metrics['duration_seconds'] = video_data.get('duration', 0)
    
    segments = video_data.get('segments', [])
    
    # Filter out segments with no frames (too short to extract frames)
    # These segments should not be counted in correlation metrics
    valid_segments = [seg for seg in segments if seg.get('frames', [])]
    
    metrics['num_segments'] = len(valid_segments)
    metrics['total_segments_in_file'] = len(segments)  # For reference
    
    stats = video_data.get('stats', {})
    metrics['total_comments_used'] = stats.get('total_comments', 0)
    metrics['top_k_limit'] = stats.get('top_k_limit', 0)
    
    # Calculate total expected pairs: valid_segments × comments
    total_expected_pairs = len(valid_segments) * metrics['total_comments_used']
    
    # Initialize correlation metrics
    visual_scores = []
    visual_correlated = 0
    visual_high_score = 0
    
    transcript_scores = []
    transcript_correlated = 0
    transcript_high_score = 0
    
    # Track union metrics - count ALL pairs (segment-comment combinations)
    union_correlated_pairs = 0
    union_high_score_pairs = 0
    
    # Process each segment
    segments_with_visual = 0
    segments_with_transcript = 0
    
    # Track all comments for word count calculation
    all_comments = set()
    
    for seg in valid_segments:
        visual_list = seg.get('visual_correlations', [])
        transcript_list = seg.get('transcript_correlations', [])
        
        # Track if this segment has any correlations
        has_visual = False
        has_transcript = False
        
        # Build a map of comments for this segment to track union
        # Key: comment text, Value: {v_corr, t_corr, v_score, t_score, v_exists, t_exists}
        segment_comment_map = {}
        
        # Process visual correlations
        for corr in visual_list:
            if not isinstance(corr, dict):
                continue
            
            score = corr.get('score', 0)
            if isinstance(score, list):
                score = score[0] if score else 0
            try:
                score = int(float(score))
            except (ValueError, TypeError):
                score = 0
            
            visual_scores.append(score)
            
            is_correlated = corr.get('correlated', False)
            if is_correlated:
                visual_correlated += 1
                has_visual = True
            
            if score >= score_threshold:
                visual_high_score += 1
            
            # Track for union calculation
            comment = corr.get('comment', '')
            if isinstance(comment, list):
                comment = ' '.join([str(x) for x in comment])
            comment = str(comment)
            all_comments.add(comment)
            
            if comment not in segment_comment_map:
                segment_comment_map[comment] = {
                    'v_corr': False, 't_corr': False, 
                    'v_score': 0, 't_score': 0,
                    'v_exists': False, 't_exists': False
                }
            segment_comment_map[comment]['v_corr'] = is_correlated
            segment_comment_map[comment]['v_score'] = score
            segment_comment_map[comment]['v_exists'] = True
        
        # Process transcript correlations
        for corr in transcript_list:
            if not isinstance(corr, dict):
                continue
            
            score = corr.get('score', 0)
            if isinstance(score, list):
                score = score[0] if score else 0
            try:
                score = int(float(score))
            except (ValueError, TypeError):
                score = 0
            
            transcript_scores.append(score)
            
            is_correlated = corr.get('correlated', False)
            if is_correlated:
                transcript_correlated += 1
                has_transcript = True
            
            if score >= score_threshold:
                transcript_high_score += 1
            
            # Track for union calculation
            comment = corr.get('comment', '')
            if isinstance(comment, list):
                comment = ' '.join([str(x) for x in comment])
            comment = str(comment)
            all_comments.add(comment)
            
            if comment not in segment_comment_map:
                segment_comment_map[comment] = {
                    'v_corr': False, 't_corr': False, 
                    'v_score': 0, 't_score': 0,
                    'v_exists': False, 't_exists': False
                }
            segment_comment_map[comment]['t_corr'] = is_correlated
            segment_comment_map[comment]['t_score'] = score
            segment_comment_map[comment]['t_exists'] = True
        
        # Calculate union for this segment
        # Each comment in this segment represents one pair
        for comment_data in segment_comment_map.values():
            # Union: visual OR transcript is correlated
            if comment_data['v_corr'] or comment_data['t_corr']:
                union_correlated_pairs += 1
            
            # Union high score: visual OR transcript score >= threshold
            if comment_data['v_score'] >= score_threshold or comment_data['t_score'] >= score_threshold:
                union_high_score_pairs += 1
        
        if has_visual:
            segments_with_visual += 1
        if has_transcript:
            segments_with_transcript += 1
    
    # Visual metrics - use total_expected_pairs as denominator
    metrics['visual_total_pairs'] = total_expected_pairs
    metrics['visual_correlated_count'] = visual_correlated
    metrics['visual_correlated_pct'] = (visual_correlated / total_expected_pairs * 100) if total_expected_pairs else 0
    metrics['visual_high_score_count'] = visual_high_score
    metrics['visual_high_score_pct'] = (visual_high_score / total_expected_pairs * 100) if total_expected_pairs else 0
    metrics['visual_avg_score'] = np.mean(visual_scores) if visual_scores else 0
    metrics['visual_median_score'] = np.median(visual_scores) if visual_scores else 0
    
    # Transcript metrics - use total_expected_pairs as denominator
    metrics['transcript_total_pairs'] = total_expected_pairs
    metrics['transcript_correlated_count'] = transcript_correlated
    metrics['transcript_correlated_pct'] = (transcript_correlated / total_expected_pairs * 100) if total_expected_pairs else 0
    metrics['transcript_high_score_count'] = transcript_high_score
    metrics['transcript_high_score_pct'] = (transcript_high_score / total_expected_pairs * 100) if total_expected_pairs else 0
    metrics['transcript_avg_score'] = np.mean(transcript_scores) if transcript_scores else 0
    metrics['transcript_median_score'] = np.median(transcript_scores) if transcript_scores else 0
    
    # Union metrics - use total_expected_pairs as denominator
    metrics['union_total_pairs'] = total_expected_pairs
    metrics['union_correlated_count'] = union_correlated_pairs
    metrics['union_correlated_pct'] = (union_correlated_pairs / total_expected_pairs * 100) if total_expected_pairs else 0
    metrics['union_high_score_count'] = union_high_score_pairs
    metrics['union_high_score_pct'] = (union_high_score_pairs / total_expected_pairs * 100) if total_expected_pairs else 0
    
    # Segment-level statistics
    metrics['avg_visual_per_segment'] = len(visual_scores) / len(valid_segments) if valid_segments else 0
    metrics['avg_transcript_per_segment'] = len(transcript_scores) / len(valid_segments) if valid_segments else 0
    metrics['segments_with_visual_corr'] = segments_with_visual
    metrics['segments_with_transcript_corr'] = segments_with_transcript
    
    # Comment statistics - calculate average comment length
    if all_comments:
        comment_lengths = [len(comment.split()) for comment in all_comments]
        metrics['avg_comment_word_count'] = np.mean(comment_lengths)
        metrics['median_comment_word_count'] = np.median(comment_lengths)
    else:
        metrics['avg_comment_word_count'] = 0
        metrics['median_comment_word_count'] = 0
    
    return metrics


def collect_metrics_for_label(results_dir: str, label: str, score_threshold: int = 60) -> pd.DataFrame:
    """
    Collect metrics for all videos of a given label (embodied or conventional).
    
    Args:
        results_dir: Path to results directory
        label: 'embodied' or 'conventional'
        score_threshold: Score threshold for high score classification
    
    Returns:
        DataFrame with metrics for all videos
    """
    # Determine the subdirectory based on label
    if label.lower() == 'embodied':
        subdir = 'segment_correlations_embodied'
    elif label.lower() == 'conventional':
        subdir = 'segment_correlations_conventional'
    else:
        raise ValueError(f"Invalid label: {label}. Must be 'embodied' or 'conventional'")
    
    correlation_dir = os.path.join(results_dir, subdir)
    
    if not os.path.exists(correlation_dir):
        print(f"[ERR] Directory not found: {correlation_dir}")
        return pd.DataFrame()
    
    # Find all correlation JSON files
    json_files = [f for f in os.listdir(correlation_dir) if f.endswith('_segment_correlation.json')]
    
    print(f"[INFO] Found {len(json_files)} correlation files in {subdir}")
    
    all_metrics = []
    
    for i, filename in enumerate(json_files, 1):
        filepath = os.path.join(correlation_dir, filename)
        
        if i % 10 == 0:
            print(f"  Processing {i}/{len(json_files)}...")
        
        video_data = load_json(filepath)
        if not video_data:
            print(f"  [WARN] Skipping {filename} - failed to load")
            continue
        
        metrics = calculate_video_metrics(video_data, score_threshold)
        metrics['label'] = label.capitalize()
        all_metrics.append(metrics)
    
    print(f"[INFO] Processed {len(all_metrics)} videos successfully")
    
    # Create DataFrame
    df = pd.DataFrame(all_metrics)
    
    # Reorder columns for better readability
    column_order = [
        'label', 'video_id', 'duration_seconds', 'num_segments', 'total_segments_in_file',
        'total_comments_used', 'top_k_limit',
        'avg_comment_word_count', 'median_comment_word_count',
        'visual_total_pairs', 'visual_correlated_count', 'visual_correlated_pct',
        'visual_high_score_count', 'visual_high_score_pct',
        'visual_avg_score', 'visual_median_score',
        'transcript_total_pairs', 'transcript_correlated_count', 'transcript_correlated_pct',
        'transcript_high_score_count', 'transcript_high_score_pct',
        'transcript_avg_score', 'transcript_median_score',
        'union_total_pairs', 'union_correlated_count', 'union_correlated_pct',
        'union_high_score_count', 'union_high_score_pct',
        'avg_visual_per_segment', 'avg_transcript_per_segment',
        'segments_with_visual_corr', 'segments_with_transcript_corr'
    ]
    
    # Only reorder columns that exist
    existing_cols = [col for col in column_order if col in df.columns]
    df = df[existing_cols]
    
    return df


def main():
    parser = argparse.ArgumentParser(description="Collect FusionAnalysis metrics and generate Excel files")
    parser.add_argument('--results_dir', default='../results', help='Path to results directory')
    parser.add_argument('--label', default='both', choices=['embodied', 'conventional', 'both'],
                        help='Which label to process (embodied, conventional, or both)')
    parser.add_argument('--output', help='Output Excel file path (used when label is not "both")')
    parser.add_argument('--output_dir', help='Output directory for Excel files (used when label is "both")')
    parser.add_argument('--score_threshold', type=int, default=60,
                        help='Score threshold for "high score" classification')
    
    args = parser.parse_args()
    
    # Resolve paths
    results_dir = os.path.abspath(args.results_dir)
    
    if args.label == 'both':
        # Process both labels
        output_dir = args.output_dir if args.output_dir else results_dir
        output_dir = os.path.abspath(output_dir)
        
        for label in ['embodied', 'conventional']:
            print(f"\n{'='*60}")
            print(f"Processing {label.upper()} videos")
            print(f"{'='*60}")
            
            df = collect_metrics_for_label(results_dir, label, args.score_threshold)
            
            if df.empty:
                print(f"[WARN] No data collected for {label}")
                continue
            
            output_file = os.path.join(output_dir, f'fusion_metrics_{label}.xlsx')
            df.to_excel(output_file, index=False, engine='openpyxl')
            print(f"[SUCCESS] Saved {len(df)} videos to {output_file}")
    
    else:
        # Process single label
        print(f"\n{'='*60}")
        print(f"Processing {args.label.upper()} videos")
        print(f"{'='*60}")
        
        df = collect_metrics_for_label(results_dir, args.label, args.score_threshold)
        
        if df.empty:
            print(f"[WARN] No data collected for {args.label}")
            return
        
        # Determine output path
        if args.output:
            output_file = os.path.abspath(args.output)
        else:
            output_file = os.path.join(results_dir, f'fusion_metrics_{args.label}.xlsx')
        
        df.to_excel(output_file, index=False, engine='openpyxl')
        print(f"[SUCCESS] Saved {len(df)} videos to {output_file}")


if __name__ == "__main__":
    main()
