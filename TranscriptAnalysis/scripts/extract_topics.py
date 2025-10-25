#!/usr/bin/env python3
"""
Topic extraction (LLM-only) from Whisper transcripts.

Now includes:
- --llm_retries (default 3): re-ask the model up to N times BEFORE any fallback.
- llm_raw attempts recorded in output JSON:
  * whole-video: {"mode":"whole_simple"|"whole_timestamped","attempts":[...]}
  * chunked: {"mode":"chunked","chunks":[{"chunk_index":i,"attempts":[...]}, ...]}
"""

from __future__ import annotations
from config import Config
import os, json, argparse, hashlib, datetime as dt, re
from os.path import join, isfile, basename, splitext
from typing import List, Dict, Any, Tuple, Callable, Optional

# -------------------- Optional project integration --------------------
try:
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from utils.llm_utils import generate_response, stream_parser  # type: ignore
    _HAS_PROJECT_LLM = True
except Exception:
    _HAS_PROJECT_LLM = False

# -------------------- LLM prompts --------------------
PROMPT_VERSION = "topics_v1.0"

EXTRACTION_SYSTEM_PROMPT = (
    "You are a careful analyst. You only use the provided transcript segment. "
    "Extract coherent topics that are actually discussed. Avoid speculation."
)

EXTRACTION_USER_INSTRUCTIONS = (
    "You are given a video transcript segment with timestamps.\n\n"
    "TASK: Extract 3-8 coherent topics discussed in THIS SEGMENT ONLY.\n"
    "For each topic, return a JSON object with: \n"
    "- title: 3-8 words, specific.\n"
    "- description: 2-4 sentences summarizing what the speaker says or shows; be faithful to the text.\n"
    "- evidence_quotes: 1-3 short verbatim quotes (10-25 words) from this segment that justify the topic.\n"
    "- time_range: start-end as HH:MM:SS.mmm–HH:MM:SS.mmm WITHIN THIS SEGMENT.\n\n"
    "RULES: Use only the given text; merge repeated subpoints; omit trivial filler (greetings, ads).\n"
    "Return ONLY a JSON array of objects. No prose before/after.\n"
)

CONSOLIDATION_USER_INSTRUCTIONS = (
    "You are given a list of topics from multiple consecutive segments of the same video.\n"
    "Your goal: deduplicate, stitch adjacent/overlapping topics, and return a final set of 5-12 topics for the video.\n"
    "Prefer specific, non-redundant titles. Keep descriptions faithful.\n"
    "Return ONLY a JSON array with objects: {title, description, evidence_quotes, start_s, end_s}.\n"
)

WHOLE_SYSTEM = (
    "You are an expert video outliner. Produce a clean, high-level list of topics "
    "for the entire video."
)

WHOLE_USER_TMPL = """You will see the FULL transcript of one video with timestamp anchors on each line:
[HH:MM:SS → HH:MM:SS] sentence…

GOAL: Produce a VIDEO-LEVEL OUTLINE of {kmin}-{kmax} high-level topics that cover the main phases of the talk.

RULES:
- Prefer broad titles (5–10 words), not micro-points.
- Topics must be chronological and non-overlapping.
- Each topic should span minutes when possible and collectively cover most of the runtime.
- IMPORTANT ON TIMING: Use ONLY the provided time anchors; SNAP start_second and end_second to the nearest anchors.
  Do NOT invent timestamps and do NOT output times earlier than the first or later than the last anchor.
- Return ONLY a JSON array with objects: {{ "title", "description", "start_s", "end_s" }}.

FULL TRANSCRIPT:
{full_text}
"""

STRICT_WHOLE_SUFFIX = (
    "\n\nFORMAT RULES (strict):\n"
    "- Return ONLY a JSON array, no prose, no code fences.\n"
    "- Make sure to include timestamps.\n"
    "- Each item must have keys exactly: title, description, start_s, end_s.\n"
)

# Whole-video SIMPLE (no timestamps; provides coverage over S-indices)
WHOLE_SIMPLE_SYSTEM = (
    "You are an expert video outliner. Produce a clean, high-level table of contents "
    "for the entire video. Use only the provided transcript lines."
)

WHOLE_SIMPLE_USER_TMPL = """You will see the FULL transcript as numbered lines.
Each line starts with an S-index in square brackets: [S00012] Text…

GOAL: Produce a VIDEO-LEVEL OUTLINE of {kmin}-{kmax} HIGH-LEVEL topics that cover the main phases of the talk.

RULES:
- Prefer broad titles (5–10 words), not micro-points or single sentences.
- Topics should be chronological and non-overlapping in concept.
- For each topic, select the FULL transcript portions it covers using as few S-INDEX RANGES as possible.
- Return ONLY a JSON array. Each item MUST be:
  {{
    "title": str,
    "description": str,
    "coverage": [{{"start_idx": int, "end_idx": int}}]  // inclusive, 0-based S-indices
  }}
- Do NOT include any prose or code fences.

FULL TRANSCRIPT (numbered lines):
{full_text}
"""

STRICT_SIMPLE_SUFFIX = (
    "\n\nFORMAT RULES (strict):\n"
    "- Return ONLY a JSON array, no prose, no code fences.\n"
    '- Each item must have keys exactly: title, description, coverage (array of {"start_idx", "end_idx"}).\n'
)

# -------------------- LLM wrappers --------------------
def llm_complete(model: str, prompt: str, system: Optional[str] = None) -> str:
    """Single call to your streaming LLM util."""
    if not _HAS_PROJECT_LLM:
        raise RuntimeError("No project LLM utilities found. Replace llm_complete() with your own call.")
    merged = (system + " " if system else "") + prompt
    stream = generate_response(model, merged)
    return "".join(stream_parser(stream)).strip()

def _truncate(txt: Optional[str], max_chars: int) -> Optional[str]:
    if txt is None:
        return None
    return txt if len(txt) <= max_chars else (txt[:max_chars].rstrip() + " …(truncated)")

def ask_json_with_retries(
    model: str,
    system: Optional[str],
    base_prompt: str,
    parse_fn: Callable[[str], Any],
    tries: int,
    strict_suffix: Optional[str] = None,
    attempt_hint: str = "\n\nReturn ONLY a JSON array. No prose."
) -> Tuple[Any, List[str]]:
    attempts_raw: List[str] = []
    if tries < 1:
        tries = 1

    for i in range(tries):
        # last attempt → add strict suffix if present; otherwise add a light hint on retries
        if i == tries - 1 and strict_suffix:
            prompt = base_prompt + strict_suffix
        else:
            prompt = base_prompt + (attempt_hint if i > 0 else "")

        raw = llm_complete(model, prompt, system=system)
        attempts_raw.append(raw)
        try:
            parsed = parse_fn(raw)
            return parsed, attempts_raw
        except Exception:
            continue

    raise ValueError("Failed to parse JSON after retries")


# -------------------- Utilities --------------------
def hms_from_seconds(s: float) -> str:
    s = max(0.0, float(s))
    ms = int(round((s - int(s)) * 1000))
    sec = int(s) % 60
    minute = (int(s) // 60) % 60
    hour = int(s) // 3600
    return f"{hour:02d}:{minute:02d}:{sec:02d}.{ms:03d}"

def seconds_from_hms(hms: str) -> float:
    s = hms.strip().replace(",", ".")
    parts = s.split(":")
    try:
        if len(parts) == 1:  return float(parts[0])
        if len(parts) == 2:  return int(parts[0]) * 60 + float(parts[1])
        if len(parts) == 3:  return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    except Exception:
        pass
    raise ValueError(f"Bad time format: {hms}")

def stable_id(i: int) -> str:
    return f"T{i+1}"

def digest_prompt(prompt_text: str) -> str:
    return hashlib.md5(prompt_text.encode("utf-8")).hexdigest()[:10]

# -------------------- Chunking --------------------
def chunk_segments(segments: List[Dict[str, Any]], max_duration_s: int, max_chars: int):
    """Yield chunks as dicts: {start, end, items, text} built from [hh:mm:ss.mmm → hh:mm:ss.mmm] lines."""
    cur = []; cur_chars = 0; cur_start = None
    for seg in segments:
        s, e, txt = float(seg["start"]), float(seg["end"]), (seg.get("text") or "").strip()
        line = f"[{hms_from_seconds(s)} → {hms_from_seconds(e)}] {txt}\n"
        if cur_start is None:
            cur_start = s
        if (e - cur_start) > max_duration_s or (cur_chars + len(line)) > max_chars:
            if cur:
                yield {"start": cur_start, "end": cur[-1]["end"], "items": cur, "text": "".join(it["line"] for it in cur)}
            cur, cur_chars, cur_start = [], 0, s
        cur.append({"start": s, "end": e, "text": txt, "line": line})
        cur_chars += len(line)
    if cur:
        yield {"start": cur_start, "end": cur[-1]["end"], "items": cur, "text": "".join(it["line"] for it in cur)}

# -------------------- Parsing / validation --------------------
class JSONArrayError(Exception): pass

def _coerce_json_array(text: str):
    if not text or not text.strip():
        raise JSONArrayError("empty reply")
    t = text.strip()
    t = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", t, flags=re.I)
    m = re.search(r"\[\s*[\s\S]*\]", t)
    if m:
        try:
            arr = json.loads(m.group(0))
            if isinstance(arr, list):
                return arr
        except Exception:
            pass
    objs = re.findall(r"\{[\s\S]*?\}", t)
    if objs:
        try:
            arr = json.loads("[" + ",".join(objs) + "]")
            if isinstance(arr, list):
                return arr
        except Exception:
            pass
    try:
        obj = json.loads(t)
        if isinstance(obj, dict): return [obj]
        if isinstance(obj, list): return obj
    except Exception:
        pass
    raise JSONArrayError("LLM did not return a JSON array")

def parse_topics_json(raw: str) -> list[dict]:
    arr = _coerce_json_array(raw)
    out = []
    for t in arr:
        if not isinstance(t, dict): continue
        title = (t.get("title") or t.get("topic") or "").strip()
        desc  = (t.get("description") or t.get("summary") or "").strip()
        quotes = t.get("evidence_quotes") or t.get("quotes") or []
        if isinstance(quotes, str): quotes = [quotes]
        quotes = [q.strip() for q in quotes if isinstance(q, str) and q.strip()]
        start_local = end_local = None
        tr = (t.get("time_range") or "").strip().replace("–", "-")
        if tr and "-" in tr:
            a, b = tr.split("-", 1)
            start_local = seconds_from_hms(a)
            end_local   = seconds_from_hms(b)
        else:
            for k in ("start_local","start"):
                if k in t:
                    try: start_local = float(t[k]); break
                    except: pass
            for k in ("end_local","end"):
                if k in t:
                    try: end_local = float(t[k]); break
                    except: pass
        if not title or not desc or start_local is None or end_local is None: continue
        if end_local <= start_local: continue
        out.append({
            "title": title, "description": desc,
            "evidence_quotes": quotes[:3],
            "start_local": float(start_local), "end_local": float(end_local),
        })
    return out

def shift_to_absolute(topics_local: List[Dict[str, Any]], chunk_start: float) -> List[Dict[str, Any]]:
    out = []
    for t in topics_local:
        out.append({
            "title": t["title"], "description": t["description"],
            "evidence_quotes": t.get("evidence_quotes", []),
            "start_s": float(t["start_local"]) + chunk_start,
            "end_s": float(t["end_local"]) + chunk_start,
        })
    return out

def consolidate_topics(all_topics: List[Dict[str, Any]], target_count_min=5, target_count_max=12) -> List[Dict[str, Any]]:
    if not all_topics: return []
    all_topics.sort(key=lambda x: (x["title"].lower(), x["start_s"]))
    merged: List[Dict[str, Any]] = []
    for t in all_topics:
        if not merged: merged.append(dict(t)); continue
        last = merged[-1]
        same_title = t["title"].strip().lower() == last["title"].strip().lower()
        overlap = t["start_s"] <= (last["end_s"] + 2.0) and same_title
        if overlap:
            last["end_s"] = max(last["end_s"], t["end_s"])
            if len(last.get("evidence_quotes", [])) < 3 and t.get("evidence_quotes"):
                last["evidence_quotes"] = (last.get("evidence_quotes", []) + t.get("evidence_quotes", []))[:3]
        else:
            merged.append(dict(t))
    topics = merged
    while len(topics) > target_count_max and len(topics) > target_count_min:
        idx = min(range(len(topics)), key=lambda i: topics[i]["end_s"] - topics[i]["start_s"])
        if idx == 0:
            topics[1]["start_s"] = min(topics[1]["start_s"], topics[0]["start_s"]); topics.pop(0)
        else:
            topics[idx-1]["end_s"] = max(topics[idx-1]["end_s"], topics[idx]["end_s"]); topics.pop(idx)
    for i, t in enumerate(topics): t["id"] = stable_id(i)
    return topics

# -------------------- Whole-transcript helpers --------------------
def pack_full_transcript(segments: List[Dict[str, Any]], max_chars: int) -> str:
    lines = [(s.get("text") or "").strip() for s in segments]
    text = "\n".join([ln for ln in lines if ln])
    if len(text) <= max_chars: return text
    step = max(2, int(len(text) / max_chars))
    thinned = "\n".join(lines[::step])
    while len(thinned) > max_chars and step < len(lines):
        step += 1; thinned = "\n".join(lines[::step])
    return thinned

def salvage_simple_outline(raw: str) -> list[dict]:
    lines = [ln.strip() for ln in raw.splitlines()]
    topics = []; cur = None
    def flush():
        nonlocal cur
        if cur and cur["title"]:
            cur["description"] = " ".join(cur["description"]).strip()
            topics.append({"id":"", "title":cur["title"], "description":cur["description"]})
        cur = None
    for ln in lines:
        if not ln: continue
        if ln.startswith("#"):
            flush(); cur = {"title": ln.lstrip("# ").strip(), "description": []}; continue
        m = re.match(r"^\*\*(.+?)\*\*$", ln)
        if m:
            flush(); cur = {"title": m.group(1).strip(), "description": []}; continue
        if cur is None: cur = {"title": "", "description": []}
        if ln.startswith(("*", "-", "•")):
            cur["description"].append(ln.lstrip("*-• ").strip())
        else:
            cur["description"].append(ln)
    flush()
    for i,t in enumerate(topics): t["id"] = f"T{i+1}"
    return topics

def pack_full_transcript_numbered(segments: List[Dict[str, Any]], max_chars: int) -> str:
    lines = [f"[S{idx:05d}] {(s.get('text') or '').strip()}" for idx, s in enumerate(segments)]
    text = "\n".join([ln for ln in lines if ln and not ln.endswith('[S')])
    if len(text) <= max_chars: return text
    step = max(2, int(len(text) / max_chars))
    thinned = "\n".join(lines[::step])
    while len(thinned) > max_chars and step < max(3, len(lines)//2):
        step += 1; thinned = "\n".join(lines[::step])
    return thinned

def _clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))

def _ranges_to_indices(ranges: List[Dict[str, Any]], n: int) -> List[int]:
    out = set()
    for r in ranges or []:
        try:
            a = int(r.get("start_idx")); b = int(r.get("end_idx"))
        except Exception:
            continue
        if a > b: a, b = b, a
        a = _clamp(a, 0, n-1); b = _clamp(b, 0, n-1)
        out.update(range(a, b+1))
    return sorted(out)

def _ranges_to_times(ranges: List[Dict[str, Any]], segments: List[Dict[str, Any]]) -> List[Dict[str, float]]:
    n = len(segments)
    times = []
    for r in ranges or []:
        try:
            a = _clamp(int(r.get("start_idx")), 0, n-1)
            b = _clamp(int(r.get("end_idx")),   0, n-1)
        except Exception:
            continue
        if a > b: a, b = b, a
        start_s = float(segments[a].get("start", 0.0))
        end_s   = float(segments[b].get("end",   start_s))
        if end_s > start_s:
            times.append({"start_s": start_s, "end_s": end_s})
    return times

# -------------------- Whole-video SIMPLE (with retries) --------------------
def outline_entire_video_simple(
    model: str,
    segments: List[Dict[str, Any]],
    kmin: int, kmax: int,
    max_chars: int,
    tries: int,
    vid: Optional[str] = None,
    debug_dir: Optional[str] = None,
    coverage_text_max_chars: int = 8000
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    full_text = pack_full_transcript_numbered(segments, max_chars=max_chars)
    base_user = WHOLE_SIMPLE_USER_TMPL.format(kmin=kmin, kmax=kmax, full_text=full_text)

    def _parse_array(text: str) -> list[dict]:
        arr = _coerce_json_array(text)
        out = []
        for o in arr:
            if not isinstance(o, dict): continue
            title = (o.get("title") or "").strip()
            desc  = (o.get("description") or "").strip()
            cov   = o.get("coverage") or []
            ranges = []
            if isinstance(cov, list):
                for r in cov:
                    if not isinstance(r, dict): continue
                    if "start_idx" in r and "end_idx" in r:
                        ranges.append({"start_idx": r["start_idx"], "end_idx": r["end_idx"]})
            if title and desc:
                out.append({"id":"", "title":title, "description":desc, "coverage": ranges})
        return out

    # Re-ask with strict on last attempt
    topics = []
    attempts_raw: List[str] = []
    try:
        topics, attempts_raw = ask_json_with_retries(
            model=model,
            system=WHOLE_SIMPLE_SYSTEM,
            base_prompt=base_user,
            parse_fn=_parse_array,
            tries=tries,
            strict_suffix=STRICT_SIMPLE_SUFFIX,
            attempt_hint="\n\nReturn ONLY a JSON array. No prose. Keys: title, description, coverage."
        )
    except Exception:
        # as a last-ditch salvage from last attempt
        if attempts_raw:
            salv = salvage_simple_outline(attempts_raw[-1])
            if salv:
                topics = [{"id":"", "title":t["title"], "description":t["description"], "coverage": []} for t in salv]
        if not topics:
            raise

    # derive coverage_* fields
    n = len(segments)
    for i, t in enumerate(topics):
        t["id"] = stable_id(i)
        ranges = t.get("coverage") or []
        norm_ranges = []
        for r in ranges:
            try: a = int(r["start_idx"]); b = int(r["end_idx"])
            except Exception: continue
            if a > b: a, b = b, a
            a = _clamp(a, 0, n-1); b = _clamp(b, 0, n-1)
            norm_ranges.append({"start_idx": a, "end_idx": b})
        norm_ranges.sort(key=lambda x: (x["start_idx"], x["end_idx"]))
        merged = []
        for r in norm_ranges:
            if not merged or r["start_idx"] > merged[-1]["end_idx"] + 1:
                merged.append(dict(r))
            else:
                merged[-1]["end_idx"] = max(merged[-1]["end_idx"], r["end_idx"])
        t["coverage_ranges"] = merged
        t.pop("coverage", None)

        idxs = _ranges_to_indices(merged, n)
        t["coverage_segments"] = idxs
        t["coverage_times"] = _ranges_to_times(merged, segments)

        parts = []
        for idx in idxs:
            txt = (segments[idx].get("text") or "").strip()
            if txt:
                parts.append(txt)
        cov_text = "\n".join(parts).strip()
        if len(cov_text) > coverage_text_max_chars:
            cov_text = cov_text[:coverage_text_max_chars].rstrip() + " …"
        t["coverage_text"] = cov_text

        t.setdefault("start_s", None)
        t.setdefault("end_s", None)

    raw_info = {
        "mode": "whole_simple",
        "attempts": attempts_raw
    }
    return topics, raw_info

# -------------------- Whole-video timestamped (with retries) --------------------
def outline_entire_video(
    model: str,
    segments: List[Dict[str, Any]],
    kmin: int, kmax: int,
    max_chars: int,
    tries: int,
    vid: Optional[str] = None,
    debug_dir: Optional[str] = None
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    full_text = pack_full_transcript(segments, max_chars=max_chars)
    base_user = WHOLE_USER_TMPL.format(kmin=kmin, kmax=kmax, full_text=full_text)

    def _parse_array(text: str) -> list[dict]:
        arr = _coerce_json_array(text)
        out: List[Dict[str, Any]] = []
        for o in arr:
            if not isinstance(o, dict): continue
            title = (o.get("title") or "").strip()
            desc  = (o.get("description") or "").strip()
            try:
                s = float(o.get("start_s")); e = float(o.get("end_s"))
            except Exception:
                continue
            if title and desc and e > s:
                quotes = o.get("evidence_quotes", [])
                if isinstance(quotes, str): quotes = [quotes]
                quotes = [q for q in quotes if isinstance(q, str) and q.strip()][:3]
                out.append({"id":"", "title": title, "description": desc,
                            "start_s": s, "end_s": e, "evidence_quotes": quotes})
        out.sort(key=lambda x: x["start_s"])
        for i,t in enumerate(out): t["id"] = stable_id(i)
        return out

    topics, attempts_raw = ask_json_with_retries(
        model=model,
        system=WHOLE_SYSTEM,
        base_prompt=base_user,
        parse_fn=_parse_array,
        tries=tries,
        strict_suffix=STRICT_WHOLE_SUFFIX,
        attempt_hint="\n\nReturn ONLY a JSON array. No prose. Keys: title, description, start_s, end_s."
    )

    raw_info = {
        "mode": "whole_timestamped",
        "attempts": attempts_raw
    }
    return topics, raw_info

# -------------------- Main processing --------------------
def process_video(json_path: str, out_dir: str, model: str,
                  max_duration_s: int, max_chars: int,
                  force: bool=False,
                  whole_video_outline: bool=False,
                  whole_max_chars: int=80000,
                  outline_min: int=6, outline_max: int=12,
                  outline_simple: bool=False,
                  coverage_text_max_chars: int=8000,
                  raw_max_chars: int=50000,
                  llm_retries: int=3) -> str:
    with open(json_path, "r", encoding="utf-8") as f:
        doc = json.load(f)
    vid = doc.get("video_id") or splitext(basename(json_path))[0]
    out_path = join(out_dir, f"{vid}.topics.json")
    if isfile(out_path) and not force:
        return f"[SKIP] {vid} (topics exist)"

    segments = doc.get("segments", [])
    if not segments:
        return f"[ERR ] {vid} has no segments"

    # --- Whole-video(s) first with progressive fallback ---
    whole_ok = False
    outlined: List[Dict[str, Any]] = []
    raw_info: Dict[str, Any] = {}

    if whole_video_outline:
        # 1) Try WHOLE SIMPLE first (if requested)
        if outline_simple:
            print(f"[INFO] {vid}: trying WHOLE-SIMPLE first...")
            try:
                outlined, raw_info = outline_entire_video_simple(
                    model=model,
                    segments=segments,
                    kmin=outline_min, kmax=outline_max,
                    max_chars=whole_max_chars,
                    tries=llm_retries,
                    vid=vid,
                    debug_dir=join(out_dir, "_raw"),
                    coverage_text_max_chars=coverage_text_max_chars,
                )
                if outlined:
                    whole_ok = True
                    print(f"[INFO] {vid}: WHOLE-SIMPLE succeeded with {len(outlined)} topics.")
                    # write and return
                    created_at = dt.datetime.now(dt.timezone.utc).isoformat()
                    prompt_fingerprint = digest_prompt(WHOLE_SIMPLE_SYSTEM + "\n" + WHOLE_SIMPLE_USER_TMPL)
                    for t in outlined:
                        t.setdefault("start_s", None);
                        t.setdefault("end_s", None)
                    payload = {
                        "video_id": vid,
                        "title": doc.get("title") or vid,
                        "url": doc.get("url") or "",
                        "created_at": created_at,
                        "model": model,
                        "prompt_version": f"{PROMPT_VERSION}:{prompt_fingerprint}",
                        "chunking": {"max_duration_s": 0, "max_chars": whole_max_chars},
                        "topics_mode": "title_description_with_coverage",
                        "topics": outlined,
                        "llm_raw": {
                            "mode": raw_info.get("mode", "whole_simple"),
                            "attempts": [_truncate(a, raw_max_chars) for a in raw_info.get("attempts", [])],
                        },
                    }
                    os.makedirs(out_dir, exist_ok=True)
                    with open(out_path, "w", encoding="utf-8") as wf:
                        json.dump(payload, wf, ensure_ascii=False, indent=2)
                    return f"[OK  ] {vid} topics (whole-video simple): {len(outlined)}"
            except Exception as e:
                print(
                    f"[WHOLE] {vid}: WHOLE-SIMPLE failed after {llm_retries} retries ({e}); will try WHOLE-TIMESTAMPED.")

        # 2) If simple didn't succeed, try WHOLE TIMESTAMPED
        if not whole_ok:
            print(f"[INFO] {vid}: trying WHOLE-TIMESTAMPED...")
            try:
                outlined, raw_info = outline_entire_video(
                    model=model,
                    segments=segments,
                    kmin=outline_min, kmax=outline_max,
                    max_chars=whole_max_chars,
                    tries=llm_retries,
                    vid=vid,
                    debug_dir=join(out_dir, "_raw"),
                )
                if outlined:
                    print(f"[INFO] {vid}: WHOLE-TIMESTAMPED succeeded with {len(outlined)} topics.")
                    created_at = dt.datetime.now(dt.timezone.utc).isoformat()
                    prompt_fingerprint = digest_prompt(WHOLE_SYSTEM + "\n" + WHOLE_USER_TMPL)
                    payload = {
                        "video_id": vid,
                        "title": doc.get("title") or vid,
                        "url": doc.get("url") or "",
                        "created_at": created_at,
                        "model": model,
                        "prompt_version": f"{PROMPT_VERSION}:{prompt_fingerprint}",
                        "chunking": {"max_duration_s": 0, "max_chars": whole_max_chars},
                        "topics_mode": "timestamped_outline",
                        "topics": outlined,
                        "llm_raw": {
                            "mode": raw_info.get("mode", "whole_timestamped"),
                            "attempts": [_truncate(a, raw_max_chars) for a in raw_info.get("attempts", [])],
                        },
                    }
                    os.makedirs(out_dir, exist_ok=True)
                    with open(out_path, "w", encoding="utf-8") as wf:
                        json.dump(payload, wf, ensure_ascii=False, indent=2)
                    return f"[OK  ] {vid} topics (whole-video): {len(outlined)}"
            except Exception as e:
                print(
                    f"[WHOLE] {vid}: WHOLE-TIMESTAMPED failed after {llm_retries} retries ({e}); will fall back to CHUNKED.")

    # 3) If neither whole variant succeeded (or whole_video_outline=False), do CHUNKED
    print(f"[INFO] {vid}: proceeding with CHUNKED extraction.")

    # --- Chunked fallback (with retries per chunk) ---
    chunks = list(chunk_segments(segments, max_duration_s=max_duration_s, max_chars=max_chars))
    STRICT_SUFFIX = (
        "\n\nFORMAT RULES (strict):\n"
        "- Return ONLY a JSON array, no prose, no code fences.\n"
        "- Each item must have keys exactly: title, description, evidence_quotes, time_range.\n"
        "- time_range must be 'HH:MM:SS.mmm–HH:MM:SS.mmm'.\n"
    )

    all_abs_topics: List[Dict[str, Any]] = []
    raw_chunks: List[Dict[str, Any]] = []

    for idx, ch in enumerate(chunks):
        base_user = EXTRACTION_USER_INSTRUCTIONS + "\nTRANSCRIPT SEGMENT (with local timestamps):\n\n" + ch["text"]

        def _parse(text: str) -> list[dict]:
            local_topics = parse_topics_json(text)
            return shift_to_absolute(local_topics, ch["start"])

        attempts_raw: List[str] = []
        try:
            abs_topics, attempts_raw = ask_json_with_retries(
                model=model,
                system=EXTRACTION_SYSTEM_PROMPT,
                base_prompt=base_user,
                parse_fn=_parse,
                tries=llm_retries,
                strict_suffix=STRICT_SUFFIX,
                attempt_hint="\n\nReturn ONLY a JSON array. No prose. Keys: title, description, evidence_quotes, time_range."
            )
            all_abs_topics.extend(abs_topics)
        except Exception as e:
            # record attempts for debugging and bail on this video (resume-safe)
            raw_chunks.append({
                "chunk_index": idx,
                "attempts": [_truncate(a, raw_max_chars) for a in attempts_raw]
            })
            debug_dir = join(out_dir, "_raw")
            os.makedirs(debug_dir, exist_ok=True)
            safe_vid = re.sub(r"[^A-Za-z0-9_-]+", "_", str(vid))
            with open(join(debug_dir, f"{safe_vid}_chunk{idx + 1:03d}.txt"), "w", encoding="utf-8") as df:
                df.write(attempts_raw[-1] if attempts_raw else "<empty>")
            return f"[ERR ] {vid} chunk {idx + 1}/{len(chunks)} failed after {llm_retries} retries: {e}"

        raw_chunks.append({
            "chunk_index": idx,
            "attempts": [_truncate(a, raw_max_chars) for a in attempts_raw]
        })

    final_topics = consolidate_topics(all_abs_topics)
    created_at = dt.datetime.now(dt.timezone.utc).isoformat()
    prompt_fingerprint = digest_prompt(EXTRACTION_SYSTEM_PROMPT + "\n" + EXTRACTION_USER_INSTRUCTIONS)
    payload = {
        "video_id": vid,
        "title": doc.get("title") or vid,
        "url": doc.get("url") or "",
        "created_at": created_at,
        "model": model,
        "prompt_version": f"{PROMPT_VERSION}:{prompt_fingerprint}",
        "chunking": {"max_duration_s": max_duration_s, "max_chars": max_chars},
        "topics": final_topics,
        "llm_raw": {"mode": "chunked", "chunks": raw_chunks},
    }
    os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as wf:
        json.dump(payload, wf, ensure_ascii=False, indent=2)

    return f"[OK  ] {vid} topics: {len(final_topics)}"

# -------------------- CLI --------------------
def main():
    ap = argparse.ArgumentParser(description="Extract topics from Whisper transcripts using an LLM (resumable)")
    ap.add_argument("--src_dir", default="../transcription/transcripts", help="Directory containing <video_id>.json transcripts")
    ap.add_argument("--out_dir", default="topics", help="Where to write <video_id>.topics.json")
    ap.add_argument("--model", default=None, help="LLM model name for topic extraction")
    ap.add_argument("--language", default=None, help="Reserved; not used here but kept for parity")
    ap.add_argument("--max_duration_s", type=int, default=900, help="Max seconds per chunk (default: 900 = 15min)")
    ap.add_argument("--max_chars", type=int, default=12000, help="Max characters per chunk")
    ap.add_argument("--start", type=int, default=0, help="Start index over files")
    ap.add_argument("--end", type=int, default=-1, help="End index (exclusive), -1 for all")
    ap.add_argument("--force", action="store_true", help="Recompute even if output exists")

    ap.add_argument("--outline_simple", action="store_true",
                    help="Whole-video outline WITHOUT timestamps; returns title/description + coverage_* fields.")
    ap.add_argument("--whole_video_outline", action="store_true",
                    help="Try a single-pass outline from the FULL transcript before chunking.")
    ap.add_argument("--whole_max_chars", type=int, default=80000,
                    help="Max characters to send for whole-video outline (uniform thinning to fit).")
    ap.add_argument("--outline_min", type=int, default=6,
                    help="Min # of top-level topics for whole-video outline")
    ap.add_argument("--outline_max", type=int, default=12,
                    help="Max # of top-level topics for whole-video outline")

    ap.add_argument("--coverage_text_max_chars", type=int, default=8000,
                    help="Max chars of concatenated transcript kept in coverage_text per topic.")
    ap.add_argument("--raw_max_chars", type=int, default=50000,
                    help="Max chars of each raw LLM attempt stored in JSON.")
    ap.add_argument("--llm_retries", type=int, default=3,
                    help="How many times to re-ask the model before falling back (min 1).")

    args = ap.parse_args()

    files = [f for f in sorted(os.listdir(args.src_dir)) if f.endswith('.json') and isfile(join(args.src_dir, f))]
    subset = files[args.start:] if args.end == -1 else files[args.start: max(0, min(len(files), args.end))]

    print(f"[INFO] Found {len(files)} transcripts; processing {len(subset)}")
    os.makedirs(args.out_dir, exist_ok=True)

    model = args.model or Config.ollama_models[0]

    for fname in subset:
        path = join(args.src_dir, fname)
        msg = process_video(
            path, args.out_dir, model,
            args.max_duration_s, args.max_chars,
            force=args.force,
            whole_video_outline=args.whole_video_outline,
            whole_max_chars=args.whole_max_chars,
            outline_min=args.outline_min,
            outline_max=args.outline_max,
            outline_simple=args.outline_simple,
            coverage_text_max_chars=args.coverage_text_max_chars,
            raw_max_chars=args.raw_max_chars,
            llm_retries=max(1, args.llm_retries),
        )
        print(msg)

if __name__ == "__main__":
    main()
