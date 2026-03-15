#!/usr/bin/env python3
"""
compute_speech_ratio.py

Computes speech_ratio = speech_duration_s / video_duration_s for each video
by parsing the Whisper-generated SRT files already present in the transcript folders.

What each column means in the output:
  speech_duration_s  — total seconds the teacher was speaking (sum of all SRT segments)
  video_duration_s   — total video length (from existing metrics CSV)
  speech_ratio       — proportion of the video that has speech  (0 = silent, 1 = always talking)
  words_per_min      — transcript word count / speech minutes (sanity check on SRT quality)
  hallucination_flag — 1 if Whisper likely hallucinated (high speech_ratio, very low words_per_min)

Why speech_ratio matters:
  Many embodied videos show physical tasks without narration — the teacher's hands are busy.
  Those videos will have a low speech_ratio and a sparse transcript, which explains why
  their comment-topic correlation scores are lower than conventional videos.
  Introducing speech_ratio as an explanatory variable de-confounds this in the paper.

Why the hallucination flag is needed:
  Whisper sometimes hallucinates repeated phrases ("Thank you.", "you", etc.) over
  silent/music segments, producing long SRT blocks with almost no real words.
  Such videos get a falsely high speech_ratio. The flag lets us decide how to handle them.
  Detection rule: speech_ratio > 0.5 AND words_per_min < 30
  (normal speech is ~120-180 wpm; < 30 wpm suggests hallucination)

Outputs (written to results/speech_ratio/):
  speech_ratio_conventional.csv
  speech_ratio_embodied.csv
  speech_ratio_combined.csv   <- most useful for scatter plots

Usage:
  cd TranscriptAnalysis/scripts/speech_ratio/
  python compute_speech_ratio.py
"""

import os
import re
import csv
from statistics import mean, median

# ── Paths (relative to this script's location) ────────────────────────────────
HERE         = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR  = os.path.dirname(HERE)
BASE         = os.path.dirname(SCRIPTS_DIR)   # TranscriptAnalysis/

TRANSCRIPTS_CONV = os.path.join(BASE, "data", "transcripts_conventional")
TRANSCRIPTS_EMB  = os.path.join(BASE, "data", "transcripts_embodied")

METRICS_CONV = os.path.join(BASE, "results", "conventional", "Conventional_metrics.csv")
METRICS_EMB  = os.path.join(BASE, "results", "embodied",     "Embodied_metrics.csv")

OUT_DIR = os.path.join(BASE, "results", "speech_ratio")

# Words-per-minute below this threshold (combined with speech_ratio > 0.5)
# is treated as likely Whisper hallucination
HALLUCINATION_WPM_THRESHOLD = 30


# ── SRT parsing ───────────────────────────────────────────────────────────────

def parse_srt_speech_duration(srt_path: str) -> float:
    """
    Parse a Whisper SRT file and return total speech duration in seconds.

    SRT format (one entry):
        1
        00:00:00,640 --> 00:00:05,920
        Hey, what's up guys?...

    Steps:
      1. Find every timestamp line (HH:MM:SS,mmm --> HH:MM:SS,mmm).
      2. Convert start/end to seconds.
      3. Merge overlapping intervals (defensive — Whisper rarely overlaps but can).
      4. Sum the merged intervals.
    """
    # Matches lines like: 00:00:05,920 --> 00:00:13,120
    pattern = re.compile(
        r'(\d{2}):(\d{2}):(\d{2}),(\d{3})\s+-->\s+(\d{2}):(\d{2}):(\d{2}),(\d{3})'
    )

    intervals = []
    with open(srt_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            m = pattern.match(line.strip())
            if m:
                h1, m1, s1, ms1, h2, m2, s2, ms2 = map(int, m.groups())
                start = h1 * 3600 + m1 * 60 + s1 + ms1 / 1000
                end   = h2 * 3600 + m2 * 60 + s2 + ms2 / 1000
                if end > start:
                    intervals.append((start, end))

    if not intervals:
        return 0.0

    # Merge overlapping/touching intervals
    intervals.sort()
    merged = [list(intervals[0])]
    for start, end in intervals[1:]:
        if start <= merged[-1][1]:          # overlaps with previous segment
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])

    return sum(end - start for start, end in merged)


# ── Load existing metrics ─────────────────────────────────────────────────────

def load_metrics_map(metrics_csv: str) -> dict:
    """
    Load existing per-video metrics CSV.
    Returns {video_id: {'duration_s': float, 'transcript_words': int}}
    Only includes rows where duration_s > 0.
    """
    result = {}
    if not os.path.isfile(metrics_csv):
        print(f"[WARN] Metrics CSV not found: {metrics_csv}")
        return result

    with open(metrics_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            vid   = row.get('video_id', '').strip()
            dur   = row.get('duration_s', '').strip()
            words = row.get('transcript_words', '').strip()
            if not vid or not dur:
                continue
            try:
                d = float(dur)
                if d > 0:
                    result[vid] = {
                        'duration_s':       d,
                        'transcript_words': int(words) if words else 0,
                    }
            except ValueError:
                pass
    return result


# ── Core computation ──────────────────────────────────────────────────────────

def compute_speech_ratios(transcripts_dir: str, metrics_map: dict, label: str) -> list:
    """
    For every SRT file in transcripts_dir, compute speech metrics.

    Returns a list of dicts, one per video, with keys:
      video_id, label, speech_duration_s, video_duration_s,
      speech_ratio, words_per_min, hallucination_flag
    """
    rows = []

    if not os.path.isdir(transcripts_dir):
        print(f"[WARN] Transcripts dir not found: {transcripts_dir}")
        return rows

    srt_files = sorted(fn for fn in os.listdir(transcripts_dir) if fn.endswith('.srt'))

    missing_duration = 0
    for fn in srt_files:
        vid      = fn[:-4]                             # strip .srt
        srt_path = os.path.join(transcripts_dir, fn)

        speech_s = parse_srt_speech_duration(srt_path)

        info   = metrics_map.get(vid)
        dur_s  = info['duration_s']       if info else None
        words  = info['transcript_words'] if info else 0

        if dur_s is None:
            missing_duration += 1

        # speech_ratio: capped at 1.0 (SRT segments occasionally extend slightly past video end)
        ratio = min(speech_s / dur_s, 1.0) if dur_s else None

        # words_per_min: how fast the teacher talked during speech segments
        # Used to detect Whisper hallucination (very low wpm despite long speech duration)
        wpm = (words / speech_s * 60) if speech_s > 0 else 0.0

        # Hallucination flag: high speech_ratio but almost no real words
        # Typical normal speech: 120-180 wpm; threshold 30 wpm is very conservative
        hall_flag = int(ratio is not None and ratio > 0.5 and wpm < HALLUCINATION_WPM_THRESHOLD)

        rows.append({
            'video_id':          vid,
            'label':             label,
            'speech_duration_s': round(speech_s, 3),
            'video_duration_s':  round(dur_s, 3) if dur_s is not None else '',
            'speech_ratio':      round(ratio, 4) if ratio is not None else '',
            'words_per_min':     round(wpm, 1),
            'transcript_words':  words,
            'hallucination_flag': hall_flag,
        })

    if missing_duration:
        print(f"  [{label}] {missing_duration} SRT files had no matching duration in metrics CSV")

    return rows


# ── Output ────────────────────────────────────────────────────────────────────

def write_csv(path: str, rows: list):
    if not rows:
        print(f"[WARN] No data to write: {path}")
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows):>3} rows → {path}")


def print_summary(label: str, rows: list):
    valid = [r for r in rows if r['speech_ratio'] != '']
    if not valid:
        print(f"  {label}: no valid ratios")
        return
    ratios = [r['speech_ratio'] for r in valid]
    flags  = sum(r['hallucination_flag'] for r in valid)
    print(f"  {label} (n={len(valid)}): "
          f"mean={mean(ratios):.3f}, "
          f"median={median(ratios):.3f}, "
          f"min={min(ratios):.3f}, "
          f"max={max(ratios):.3f} "
          f"| hallucination_flag={flags} videos")
    if flags:
        flagged = [r['video_id'] for r in valid if r['hallucination_flag']]
        print(f"    Flagged video IDs: {flagged}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Load existing per-video metrics (for duration_s and transcript_words)
    metrics_conv = load_metrics_map(METRICS_CONV)
    metrics_emb  = load_metrics_map(METRICS_EMB)
    print(f"Loaded metrics: {len(metrics_conv)} conventional, {len(metrics_emb)} embodied")

    # Compute speech ratios from SRT files
    rows_conv = compute_speech_ratios(TRANSCRIPTS_CONV, metrics_conv, "Conventional")
    rows_emb  = compute_speech_ratios(TRANSCRIPTS_EMB,  metrics_emb,  "Embodied")

    # Write outputs
    write_csv(os.path.join(OUT_DIR, "speech_ratio_conventional.csv"), rows_conv)
    write_csv(os.path.join(OUT_DIR, "speech_ratio_embodied.csv"),     rows_emb)
    write_csv(os.path.join(OUT_DIR, "speech_ratio_combined.csv"),     rows_conv + rows_emb)

    # Print summary stats
    print("\nSummary:")
    print_summary("Conventional", rows_conv)
    print_summary("Embodied",     rows_emb)


if __name__ == '__main__':
    main()
