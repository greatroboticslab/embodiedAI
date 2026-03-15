#!/usr/bin/env python3
"""
gemini_analyze_transcripts.py

Sends each video transcript to the Gemini API to extract:
  1. Keypoints  — main topics the video teaches, with strength scores
  2. Embodied phrases — language rooted in physical action or demonstration

This is an audio-only analysis. No visual/frame data is used.

Output per video (results/gemini_analysis/conventional/ and embodied/):
  <video_id>.json   — full Gemini output (keypoints + embodied phrases + summary)

Aggregate CSV output (one row per video):
  results/gemini_analysis/conventional_summary.csv
  results/gemini_analysis/embodied_summary.csv
  results/gemini_analysis/combined_summary.csv   <- use this for scatter plots

Usage:
  cd TranscriptAnalysis/scripts/gemini_analysis/
  python gemini_analyze_transcripts.py --api_key YOUR_KEY_HERE

  # Dry run (parse prompts but don't call API — for testing):
  python gemini_analyze_transcripts.py --api_key YOUR_KEY_HERE --dry_run

  # Process only one set:
  python gemini_analyze_transcripts.py --api_key YOUR_KEY_HERE --label conventional

  # Use a specific model:
  python gemini_analyze_transcripts.py --api_key YOUR_KEY_HERE --model gemini-2.0-flash

Resume behavior:
  Already-processed videos (existing .json in output dir) are skipped automatically.
  Re-run the same command to resume after interruption.

Rate limiting:
  Free tier: ~15 requests/min. Default sleep is 4s between calls (~15/min).
  Adjust with --sleep if you have a paid tier.
"""

import os
import re
import json
import time
import csv
import argparse
from typing import Optional

# ── Paths ─────────────────────────────────────────────────────────────────────
HERE        = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.dirname(HERE)
BASE        = os.path.dirname(SCRIPTS_DIR)   # TranscriptAnalysis/

TRANSCRIPTS_CONV = os.path.join(BASE, "data", "transcripts_conventional")
TRANSCRIPTS_EMB  = os.path.join(BASE, "data", "transcripts_embodied")
OUT_DIR          = os.path.join(BASE, "results", "gemini_analysis")

SETS = [
    ("conventional", TRANSCRIPTS_CONV),
    ("embodied",     TRANSCRIPTS_EMB),
]

# ── Prompt ────────────────────────────────────────────────────────────────────

PROMPT_TEMPLATE = """You are analyzing a transcript from a robotics education YouTube video.
Extract two things: (1) keypoints, and (2) embodied language phrases.

DEFINITIONS
-----------
Keypoint: A substantive concept, skill, or topic the video teaches.
Not every word — only topics the speaker devotes meaningful time to.

Embodied phrase: Any phrase where the speaker uses language rooted in
physical action, direct demonstration, or sensory experience.
Four categories:
  - visual_reference      : "look at this", "as you can see", "right here on screen"
  - action_narration      : "I'm now attaching", "connect the wire to", "I'm soldering"
  - sensory_description   : "you can feel it click", "notice how it sounds", "it feels firm"
  - procedural_instruction: "hold it steady", "make sure this is aligned", "tighten until"

STRENGTH SCORE SCALE (for keypoints)
--------------------------------------
1 — Mentioned once briefly, no development
2 — Mentioned 2-3 times, limited coverage
3 — Covered in a dedicated section, multiple mentions
4 — Introduced, developed, and returned to; central to the video
5 — Primary focus of the entire video, explained from multiple angles, reinforced at end

OUTPUT
------
Return valid JSON only. No markdown fences, no explanation outside the JSON object.

{{
  "keypoints": [
    {{
      "name": "<short name>",
      "description": "<1-2 sentences on what this keypoint covers>",
      "mention_count": <integer>,
      "positions": <list from ["early", "mid", "late"] — which thirds of the transcript it appears in>,
      "reinforced_at_end": <true or false>,
      "strength_score": <integer 1-5>,
      "strength_reasoning": "<brief explanation of the score>",
      "representative_quotes": ["<verbatim short quote>", "<verbatim short quote>"]
    }}
  ],
  "embodied_phrases": [
    {{
      "phrase": "<verbatim phrase from transcript>",
      "category": "<visual_reference|action_narration|sensory_description|procedural_instruction>",
      "position_pct": <float 0.0-100.0, approximate position in the transcript>,
      "context": "<10-20 words of surrounding text>"
    }}
  ],
  "embodied_summary": {{
    "total_count": <integer>,
    "by_category": {{
      "visual_reference": <integer>,
      "action_narration": <integer>,
      "sensory_description": <integer>,
      "procedural_instruction": <integer>
    }},
    "verbal_embodied_richness": "<high|medium|low|none>",
    "verbal_embodied_reasoning": "<1-2 sentence explanation>"
  }}
}}

TRANSCRIPT:
{transcript_text}"""


# ── Transcript loading ────────────────────────────────────────────────────────

def load_transcript(transcripts_dir: str, video_id: str) -> Optional[str]:
    """Load transcript text for a video. Returns None if not found."""
    path = os.path.join(transcripts_dir, f"{video_id}.txt")
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read().strip()
    # Strip the first 3 lines (video ID, URL, category label added by transcription script)
    lines = text.splitlines()
    if len(lines) > 3:
        text = "\n".join(lines[3:]).strip()
    return text if text else None


def discover_video_ids(transcripts_dir: str) -> list:
    """Return sorted list of video IDs from .txt files in transcripts_dir."""
    ids = []
    for fn in os.listdir(transcripts_dir):
        if fn.endswith(".txt") and not fn.endswith("_timestamped.txt"):
            ids.append(fn[:-4])
    return sorted(ids)


# ── Gemini API ────────────────────────────────────────────────────────────────

def init_gemini(api_key: str, model_name: str):
    """
    Initialize Gemini client. Supports both SDK versions:
      - google-generativeai  (pip install google-generativeai)
      - google-genai         (pip install google-genai)
    Tries google-genai first (newer), falls back to google-generativeai.
    """
    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        print(f"Using google-genai SDK with model: {model_name}")
        return ("genai_new", client, model_name)
    except Exception:
        # Falls back if package missing or incompatible with current Python version
        pass

    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)
        print(f"Using google-generativeai SDK with model: {model_name}")
        return ("genai_old", model, model_name)
    except ImportError:
        raise SystemExit(
            "No Gemini SDK found. Install one of:\n"
            "  pip install google-genai\n"
            "  pip install google-generativeai"
        )


def call_gemini(client_info: tuple, prompt: str) -> str:
    """Call Gemini and return raw response text."""
    sdk_type, client_or_model, model_name = client_info

    if sdk_type == "genai_new":
        response = client_or_model.models.generate_content(
            model=model_name,
            contents=prompt,
        )
        return response.text

    else:  # genai_old
        response = client_or_model.generate_content(prompt)
        return response.text


# ── JSON parsing ──────────────────────────────────────────────────────────────

def parse_gemini_json(raw: str) -> Optional[dict]:
    """
    Parse JSON from Gemini response.
    Handles cases where Gemini wraps output in markdown fences despite instructions.
    Returns None if parsing fails entirely.
    """
    text = raw.strip()

    # Strip markdown fences if present: ```json ... ``` or ``` ... ```
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fence_match:
        text = fence_match.group(1).strip()

    # Try parsing directly
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try finding the outermost JSON object
    brace_start = text.find("{")
    brace_end   = text.rfind("}")
    if brace_start != -1 and brace_end != -1:
        try:
            return json.loads(text[brace_start:brace_end + 1])
        except json.JSONDecodeError:
            pass

    return None


# ── Output helpers ────────────────────────────────────────────────────────────

def save_video_result(out_dir: str, video_id: str, label: str,
                      parsed: Optional[dict], raw: str):
    """Save per-video JSON result."""
    set_dir = os.path.join(out_dir, label)
    os.makedirs(set_dir, exist_ok=True)
    result = {
        "video_id":  video_id,
        "label":     label,
        "parsed":    parsed,
        "raw_response": raw,
    }
    path = os.path.join(set_dir, f"{video_id}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return path


def extract_summary_row(video_id: str, label: str, parsed: Optional[dict]) -> dict:
    """
    Flatten the parsed Gemini output into a single CSV row.
    Handles missing/null parsed output gracefully.
    """
    row = {
        "video_id":                     video_id,
        "label":                        label,
        "keypoint_count":               "",
        "avg_strength_score":           "",
        "max_strength_score":           "",
        "keypoints_reinforced_at_end":  "",
        "embodied_total":               "",
        "embodied_visual_reference":    "",
        "embodied_action_narration":    "",
        "embodied_sensory":             "",
        "embodied_procedural":          "",
        "verbal_embodied_richness":     "",
        "parse_success":                0,
    }

    if not parsed:
        return row

    row["parse_success"] = 1

    # Keypoints
    kps = parsed.get("keypoints") or []
    row["keypoint_count"] = len(kps)
    if kps:
        scores = [k.get("strength_score") for k in kps if isinstance(k.get("strength_score"), (int, float))]
        row["avg_strength_score"]          = round(sum(scores) / len(scores), 2) if scores else ""
        row["max_strength_score"]          = max(scores) if scores else ""
        row["keypoints_reinforced_at_end"] = sum(1 for k in kps if k.get("reinforced_at_end"))

    # Embodied summary
    es = parsed.get("embodied_summary") or {}
    row["embodied_total"]            = es.get("total_count", "")
    by_cat = es.get("by_category") or {}
    row["embodied_visual_reference"] = by_cat.get("visual_reference",      "")
    row["embodied_action_narration"] = by_cat.get("action_narration",      "")
    row["embodied_sensory"]          = by_cat.get("sensory_description",   "")
    row["embodied_procedural"]       = by_cat.get("procedural_instruction","")
    row["verbal_embodied_richness"]  = es.get("verbal_embodied_richness",  "")

    return row


SUMMARY_FIELDS = [
    "video_id", "label",
    "keypoint_count", "avg_strength_score", "max_strength_score", "keypoints_reinforced_at_end",
    "embodied_total", "embodied_visual_reference", "embodied_action_narration",
    "embodied_sensory", "embodied_procedural",
    "verbal_embodied_richness", "parse_success",
]


def write_summary_csv(path: str, rows: list):
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows):>3} rows → {path}")


# ── Main processing loop ──────────────────────────────────────────────────────

def process_set(label: str, transcripts_dir: str, client_info,
                out_dir: str, sleep_s: float, dry_run: bool) -> list:
    """
    Process all videos in one set (conventional or embodied).
    Returns list of summary rows.
    """
    set_out_dir = os.path.join(out_dir, label)
    os.makedirs(set_out_dir, exist_ok=True)

    video_ids = discover_video_ids(transcripts_dir)
    print(f"\n[{label}] Found {len(video_ids)} videos")

    summary_rows = []
    skipped = 0
    errors  = 0

    for i, vid in enumerate(video_ids, 1):
        out_path = os.path.join(set_out_dir, f"{vid}.json")

        # Resume: skip already-processed videos
        if os.path.isfile(out_path):
            # Load existing result to include in summary
            try:
                with open(out_path) as f:
                    existing = json.load(f)
                summary_rows.append(extract_summary_row(vid, label, existing.get("parsed")))
            except Exception:
                pass
            skipped += 1
            continue

        # Load transcript
        transcript = load_transcript(transcripts_dir, vid)
        if not transcript:
            print(f"  [{i}/{len(video_ids)}] {vid} — no transcript, skipping")
            summary_rows.append(extract_summary_row(vid, label, None))
            continue

        word_count = len(transcript.split())
        prompt     = PROMPT_TEMPLATE.format(transcript_text=transcript)

        if dry_run:
            print(f"  [{i}/{len(video_ids)}] {vid} — DRY RUN ({word_count} words)")
            summary_rows.append(extract_summary_row(vid, label, None))
            continue

        # Call Gemini
        print(f"  [{i}/{len(video_ids)}] {vid} ({word_count} words) ... ", end="", flush=True)
        try:
            raw    = call_gemini(client_info, prompt)
            parsed = parse_gemini_json(raw)
            status = "OK" if parsed else "PARSE_FAIL"
            print(status)
        except Exception as e:
            print(f"ERROR: {e}")
            raw    = ""
            parsed = None
            errors += 1

        save_video_result(out_dir, vid, label, parsed, raw)
        summary_rows.append(extract_summary_row(vid, label, parsed))

        # Rate limiting
        time.sleep(sleep_s)

    if skipped:
        print(f"  [{label}] Skipped {skipped} already-processed videos (resume)")
    if errors:
        print(f"  [{label}] {errors} API errors — re-run to retry")

    return summary_rows


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Gemini transcript analysis — keypoints + embodied phrases")
    ap.add_argument("--api_key",  required=True, help="Gemini API key")
    ap.add_argument("--model",    default="gemini-2.0-flash",
                    help="Gemini model name (default: gemini-2.0-flash)")
    ap.add_argument("--label",    default="both",
                    choices=["conventional", "embodied", "both"],
                    help="Which set to process (default: both)")
    ap.add_argument("--sleep",    type=float, default=4.0,
                    help="Seconds to sleep between API calls (default: 4.0 for free tier)")
    ap.add_argument("--dry_run",  action="store_true",
                    help="Load transcripts and build prompts but do not call the API")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    # Initialize Gemini client (once, reused for all calls)
    if not args.dry_run:
        client_info = init_gemini(args.api_key, args.model)
    else:
        client_info = None
        print("DRY RUN mode — no API calls will be made")

    # Determine which sets to process
    sets_to_run = SETS if args.label == "both" else [s for s in SETS if s[0] == args.label]

    all_rows = []
    for label, transcripts_dir in sets_to_run:
        rows = process_set(
            label          = label,
            transcripts_dir= transcripts_dir,
            client_info    = client_info,
            out_dir        = OUT_DIR,
            sleep_s        = args.sleep,
            dry_run        = args.dry_run,
        )
        write_summary_csv(
            os.path.join(OUT_DIR, f"{label}_summary.csv"), rows
        )
        all_rows.extend(rows)

    # Combined summary (useful for scatter plots and stats)
    if len(sets_to_run) == 2:
        write_summary_csv(os.path.join(OUT_DIR, "combined_summary.csv"), all_rows)

    print("\nDone.")
    if args.dry_run:
        print("Re-run without --dry_run to process with the API.")


if __name__ == "__main__":
    main()
