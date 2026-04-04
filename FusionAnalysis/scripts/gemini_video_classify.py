#!/usr/bin/env python3
"""
gemini_video_classify.py

Uploads videos to Gemini and asks it to classify each 10-second segment by:
  1. Content type (9 categories)
  2. Audio-visual alignment score (0-100)
  3. Narration present (true/false)
  4. Hands visible (true/false)
  5. Instructional density (1-5)

Videos are processed in 5-minute chunks to keep Gemini output reliable (~30 segments per call).
Each video is uploaded once via the File API, then referenced in multiple chunked calls.

Output per video:
  <video_id>.json — array of segment classifications

Usage:
  python gemini_video_classify.py --api_key YOUR_KEY_HERE

  # Process only one label:
  python gemini_video_classify.py --api_key YOUR_KEY_HERE --label conventional

  # Dry run (no API calls):
  python gemini_video_classify.py --api_key YOUR_KEY_HERE --dry_run

  # Use a specific model:
  python gemini_video_classify.py --api_key YOUR_KEY_HERE --model gemini-2.0-flash

Resume behavior:
  Already-processed videos (existing .json in output dir) are skipped automatically.
  Partially-processed videos (some chunks done) resume from the last completed chunk.
  Re-run the same command to resume after interruption.

Rate limiting:
  Default sleep is 4s between calls (~15 req/min).
  Adjust with --sleep if you have a paid tier.
"""

import os
import re
import json
import time
import math
import argparse
from typing import Optional, Dict, List, Any

# ── Paths ─────────────────────────────────────────────────────────────────────
HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))

VIDEOS_CONV = os.path.join(PROJECT_ROOT, "VideoAnalysis", "rawvideos", "conventional_videos")
VIDEOS_EMB = os.path.join(PROJECT_ROOT, "VideoAnalysis", "rawvideos", "embodied_videos")

RESULTS_DIR = os.path.join(HERE, "..", "results", "video_classification")

# ── Constants ─────────────────────────────────────────────────────────────────
CHUNK_DURATION = 300  # 5 minutes in seconds
SEGMENT_DURATION = 10  # 10 seconds per segment

CONTENT_TYPES = [
    "hands_on_demonstration",
    "equipment_closeup",
    "presenter_talking",
    "slide_or_powerpoint",
    "diagram_or_whiteboard",
    "screen_or_software",
    "animation_or_graphic",
    "device_in_operation",
    "other",
]

PROMPT_TEMPLATE = """You are analyzing an educational video used in a university engineering course. The video may cover any technical topic related to engineering or robotics education, including but not limited to: electronics, circuits, batteries, soldering, process control, sensors, motor systems, audio amplifiers, computer vision, programming, drone building, robot assembly, or other hands-on technical subjects.

Watch ONLY the portion of this video from {start_time} to {end_time}. Divide this portion into exactly {num_segments} consecutive 10-second intervals and classify each one. Return a JSON array with exactly {num_segments} elements.

For each 10-second interval, provide:

1. "segment_index": sequential number starting from {start_index}
2. "start": start time in seconds
3. "end": end time in seconds
4. "content_type": one of the following categories (pick the BEST fit for what dominates the visual frame):
   - "hands_on_demonstration": a person's hands are actively working — assembling, wiring, soldering, connecting, measuring, adjusting, or physically manipulating any equipment or components
   - "equipment_closeup": static or zoomed-in view of hardware, components, circuit boards, devices, or equipment that is NOT being actively manipulated (just shown or displayed)
   - "presenter_talking": an instructor or presenter is visible on camera, speaking or gesturing, but not performing hands-on work (e.g., lecturing to camera, standing at a podium)
   - "slide_or_powerpoint": PowerPoint slides, title cards, text overlays, bullet points, equations displayed on screen
   - "diagram_or_whiteboard": hand-drawn diagrams, schematics, circuit diagrams, whiteboard illustrations, technical drawings
   - "screen_or_software": computer screen showing an IDE, terminal, simulation, web browser, CAD tool, or any software interface
   - "animation_or_graphic": animated explanations, 3D renders, motion graphics, visual effects, animated diagrams
   - "device_in_operation": a device, robot, drone, motor, circuit, or system actively running, being tested, or demonstrating its function (not being built — already operating)
   - "other": transitions, intro/outro sequences, sponsor segments, music-only sections, B-roll footage, or anything that does not fit the above categories
5. "alignment_score": 0-100 integer. How well does the audio narration match what is shown visually? 100 means the narrator is describing exactly what is visible on screen. 0 means the audio and visuals are completely unrelated, or there is no audio. If there is no narration in this interval, score 0.
6. "narration": true if someone is speaking during this interval, false if it is silent or music-only
7. "hands_visible": true if a person's hands are visible and actively doing something on screen, false otherwise
8. "instructional_density": 1-5 integer. How much teaching content is packed into this interval? 5 = dense instruction with a new concept being taught or a step being explained in detail. 1 = no instructional content (intro animation, transition, filler, or dead time).

Return ONLY a valid JSON array. No explanation, no markdown fences, no text before or after the array.

Example format:
[
  {{"segment_index": 1, "start": 0, "end": 10, "content_type": "hands_on_demonstration", "alignment_score": 85, "narration": true, "hands_visible": true, "instructional_density": 4}},
  {{"segment_index": 2, "start": 10, "end": 20, "content_type": "slide_or_powerpoint", "alignment_score": 90, "narration": true, "hands_visible": false, "instructional_density": 3}}
]
"""


# ── SDK Init ──────────────────────────────────────────────────────────────────

def init_gemini(api_key: str, model_name: str):
    """Initialize Gemini SDK. Try google-genai (newer) first, fall back to google-generativeai."""
    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        print(f"Using google-genai SDK with model: {model_name}")
        return ("genai_new", client, model_name)
    except Exception:
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


def upload_video(client_info: tuple, video_path: str):
    """Upload video to Gemini File API. Returns the file reference."""
    sdk_type, client_or_model, model_name = client_info

    if sdk_type == "genai_new":
        from google import genai
        # Upload using the client
        uploaded = client_or_model.files.upload(file=video_path)
        print(f"  Uploaded: {uploaded.name} ({uploaded.state})")
        # Wait for processing
        while uploaded.state.name == "PROCESSING":
            time.sleep(5)
            uploaded = client_or_model.files.get(name=uploaded.name)
        if uploaded.state.name == "FAILED":
            raise RuntimeError(f"Upload failed for {video_path}: {uploaded.state}")
        return uploaded

    else:  # genai_old
        import google.generativeai as genai
        uploaded = genai.upload_file(video_path)
        print(f"  Uploaded: {uploaded.name} ({uploaded.state.name})")
        while uploaded.state.name == "PROCESSING":
            time.sleep(5)
            uploaded = genai.get_file(uploaded.name)
        if uploaded.state.name == "FAILED":
            raise RuntimeError(f"Upload failed for {video_path}: {uploaded.state}")
        return uploaded


def call_gemini_with_video(client_info: tuple, video_file, prompt: str) -> str:
    """Call Gemini with a video file reference and text prompt."""
    sdk_type, client_or_model, model_name = client_info

    if sdk_type == "genai_new":
        response = client_or_model.models.generate_content(
            model=model_name,
            contents=[video_file, prompt],
        )
        return response.text

    else:  # genai_old
        response = client_or_model.generate_content([video_file, prompt])
        return response.text


def delete_uploaded_file(client_info: tuple, video_file):
    """Delete uploaded file from Gemini to free storage."""
    sdk_type, client_or_model, model_name = client_info
    try:
        if sdk_type == "genai_new":
            client_or_model.files.delete(name=video_file.name)
        else:
            import google.generativeai as genai
            genai.delete_file(video_file.name)
        print(f"  Deleted uploaded file: {video_file.name}")
    except Exception as e:
        print(f"  [WARN] Could not delete {video_file.name}: {e}")


# ── JSON Parsing ──────────────────────────────────────────────────────────────

def parse_gemini_json(raw: str) -> Optional[list]:
    """Parse JSON array from Gemini response."""
    text = raw.strip()

    # Strip markdown fences if present
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fence_match:
        text = fence_match.group(1).strip()

    # Try parsing directly
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
        return None
    except json.JSONDecodeError:
        pass

    # Try finding outermost array
    bracket_start = text.find("[")
    bracket_end = text.rfind("]")
    if bracket_start != -1 and bracket_end != -1:
        try:
            result = json.loads(text[bracket_start:bracket_end + 1])
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass

    return None


def validate_segment(seg: dict, expected_index: int) -> dict:
    """Validate and normalize a single segment classification."""
    validated = {}
    validated["segment_index"] = seg.get("segment_index", expected_index)
    validated["start"] = seg.get("start", (expected_index - 1) * SEGMENT_DURATION)
    validated["end"] = seg.get("end", expected_index * SEGMENT_DURATION)

    ct = seg.get("content_type", "other")
    validated["content_type"] = ct if ct in CONTENT_TYPES else "other"

    alignment = seg.get("alignment_score", 0)
    try:
        alignment = int(float(alignment))
    except (ValueError, TypeError):
        alignment = 0
    validated["alignment_score"] = max(0, min(100, alignment))

    validated["narration"] = bool(seg.get("narration", False))
    validated["hands_visible"] = bool(seg.get("hands_visible", False))

    density = seg.get("instructional_density", 1)
    try:
        density = int(float(density))
    except (ValueError, TypeError):
        density = 1
    validated["instructional_density"] = max(1, min(5, density))

    return validated


# ── Video Duration ────────────────────────────────────────────────────────────

def load_duration_map() -> Dict[str, float]:
    """Load pre-computed video durations from JSON."""
    duration_path = os.path.join(HERE, "..", "results", "video_durations.json")
    if os.path.exists(duration_path):
        with open(duration_path, "r") as f:
            return json.load(f)
    return {}


# Global duration cache (loaded once)
_DURATION_MAP: Dict[str, float] = {}


def get_video_duration(video_path: str, video_id: str) -> float:
    """Get video duration using ffprobe, falling back to pre-computed map."""
    global _DURATION_MAP
    # Try ffprobe first
    try:
        import subprocess
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", video_path],
            capture_output=True, text=True, timeout=30
        )
        dur = float(result.stdout.strip())
        if dur > 0:
            return dur
    except Exception:
        pass

    # Fallback to pre-computed duration map
    if not _DURATION_MAP:
        _DURATION_MAP = load_duration_map()

    if video_id in _DURATION_MAP:
        return _DURATION_MAP[video_id]

    return 0.0


# ── Main Processing ───────────────────────────────────────────────────────────

def process_video(client_info: tuple, video_path: str, video_id: str,
                  output_dir: str, sleep_sec: float, dry_run: bool) -> bool:
    """Process a single video: upload, classify in chunks, save results."""
    output_path = os.path.join(output_dir, f"{video_id}.json")

    # Load existing partial results if resuming
    existing_segments = []
    if os.path.exists(output_path):
        try:
            with open(output_path, "r") as f:
                data = json.load(f)
            if data.get("complete", False):
                print(f"  [SKIP] Already complete: {video_id}")
                return True
            existing_segments = data.get("segments", [])
            print(f"  [RESUME] Found {len(existing_segments)} existing segments")
        except Exception:
            existing_segments = []

    # Get duration
    duration = get_video_duration(video_path, video_id)
    if duration <= 0:
        print(f"  [WARN] Could not determine duration for {video_id}, skipping")
        return False

    total_segments = math.ceil(duration / SEGMENT_DURATION)
    num_chunks = math.ceil(duration / CHUNK_DURATION)
    print(f"  Duration: {duration:.0f}s, {total_segments} segments, {num_chunks} chunks")

    if dry_run:
        print(f"  [DRY RUN] Would process {num_chunks} chunks")
        return True

    # Upload video
    print(f"  Uploading video...")
    try:
        video_file = upload_video(client_info, video_path)
    except Exception as e:
        print(f"  [ERR] Upload failed: {e}")
        return False

    # Determine which chunks are already done
    last_done_index = max((s["segment_index"] for s in existing_segments), default=0)
    all_segments = list(existing_segments)

    try:
        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * CHUNK_DURATION
            chunk_end = min((chunk_idx + 1) * CHUNK_DURATION, duration)
            start_index = chunk_idx * (CHUNK_DURATION // SEGMENT_DURATION) + 1

            # Skip already-processed chunks
            chunk_num_segments = math.ceil((chunk_end - chunk_start) / SEGMENT_DURATION)
            end_index = start_index + chunk_num_segments - 1
            if end_index <= last_done_index:
                continue

            # Format time strings for prompt
            start_min = int(chunk_start // 60)
            start_sec = int(chunk_start % 60)
            end_min = int(chunk_end // 60)
            end_sec = int(chunk_end % 60)
            start_str = f"{start_min}:{start_sec:02d}"
            end_str = f"{end_min}:{end_sec:02d}"

            prompt = PROMPT_TEMPLATE.format(
                start_time=start_str,
                end_time=end_str,
                start_index=start_index,
                num_segments=chunk_num_segments,
            )

            print(f"  Chunk {chunk_idx + 1}/{num_chunks}: {start_str} - {end_str} "
                  f"(segments {start_index}-{end_index})...")

            # Call Gemini with retry on transient errors
            raw_response = None
            for attempt in range(3):
                try:
                    raw_response = call_gemini_with_video(client_info, video_file, prompt)
                    break
                except Exception as e:
                    err_str = str(e)
                    if ("503" in err_str or "UNAVAILABLE" in err_str or
                            "500" in err_str or "deadline" in err_str.lower()):
                        wait = 30 * (attempt + 1)
                        print(f"  [RETRY] Attempt {attempt + 1}/3 failed: {e}")
                        print(f"  Waiting {wait}s before retry...")
                        time.sleep(wait)
                    else:
                        print(f"  [ERR] API call failed for chunk {chunk_idx + 1}: {e}")
                        break
            if raw_response is None:
                print(f"  [ERR] All retries failed for chunk {chunk_idx + 1}")
                # Save what we have so far
                break

            # Parse response
            segments = parse_gemini_json(raw_response)
            if segments is None:
                print(f"  [ERR] Failed to parse JSON for chunk {chunk_idx + 1}")
                print(f"  Raw response (first 500 chars): {raw_response[:500]}")
                break

            # Validate each segment
            expected_idx = start_index
            for seg in segments:
                validated = validate_segment(seg, expected_idx)
                all_segments.append(validated)
                expected_idx += 1

            print(f"    Got {len(segments)} segments")

            # Save after each chunk (for resume)
            save_results(output_path, video_id, duration, total_segments,
                         all_segments, complete=False)

            time.sleep(sleep_sec)

    finally:
        # Always clean up the uploaded file
        delete_uploaded_file(client_info, video_file)

    # Mark complete if we got all segments
    is_complete = len(all_segments) >= total_segments * 0.9  # Allow 10% tolerance
    save_results(output_path, video_id, duration, total_segments,
                 all_segments, complete=is_complete)

    if is_complete:
        print(f"  [OK] Complete: {len(all_segments)} segments")
    else:
        print(f"  [PARTIAL] Got {len(all_segments)}/{total_segments} segments")

    return is_complete


def save_results(output_path: str, video_id: str, duration: float,
                 total_segments: int, segments: list, complete: bool):
    """Save classification results to JSON."""
    data = {
        "video_id": video_id,
        "duration": duration,
        "total_expected_segments": total_segments,
        "segments_classified": len(segments),
        "complete": complete,
        "segments": segments,
    }
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Classify video segments using Gemini video understanding"
    )
    parser.add_argument("--api_key", required=True, help="Gemini API key")
    parser.add_argument("--model", default="gemini-2.0-flash",
                        help="Gemini model name (default: gemini-2.0-flash)")
    parser.add_argument("--label", default="both",
                        choices=["embodied", "conventional", "both"],
                        help="Which video set to process")
    parser.add_argument("--output_dir", default=RESULTS_DIR,
                        help="Output directory for classification JSONs")
    parser.add_argument("--sleep", type=float, default=4.0,
                        help="Seconds between API calls (default: 4)")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print what would be done without calling API")
    args = parser.parse_args()

    # Init SDK
    client_info = init_gemini(args.api_key, args.model)

    # Determine which video sets to process
    video_sets = []
    if args.label in ("conventional", "both"):
        video_sets.append(("conventional", VIDEOS_CONV))
    if args.label in ("embodied", "both"):
        video_sets.append(("embodied", VIDEOS_EMB))

    for label, video_dir in video_sets:
        label_output = os.path.join(args.output_dir, label)
        os.makedirs(label_output, exist_ok=True)

        # Find all mp4 files
        videos = sorted([
            f for f in os.listdir(video_dir)
            if f.endswith(".mp4")
        ])

        print(f"\n{'=' * 60}")
        print(f"Processing {label.upper()}: {len(videos)} videos")
        print(f"Output: {label_output}")
        print(f"{'=' * 60}")

        success = 0
        fail = 0

        for i, filename in enumerate(videos, 1):
            video_id = filename.replace(".mp4", "")
            video_path = os.path.join(video_dir, filename)

            print(f"\n[{i}/{len(videos)}] {video_id}")

            ok = process_video(
                client_info, video_path, video_id,
                label_output, args.sleep, args.dry_run,
            )

            if ok:
                success += 1
            else:
                fail += 1

        print(f"\n{label.upper()} done: {success} success, {fail} failed")

    print("\nAll done.")


if __name__ == "__main__":
    main()
