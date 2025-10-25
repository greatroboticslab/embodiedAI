# transcribe_local_folder.py
import os, json, argparse
from os.path import join, isfile, splitext, basename
from os import listdir
import re
import whisper

def fmt_ts_srt(t):
    ms = int(round((t - int(t)) * 1000))
    s = int(t) % 60; m = (int(t)//60) % 60; h = int(t)//3600
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

def try_get_fps(video_path):
    try:
        import cv2
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        return float(fps) if fps and fps > 0 else None
    except Exception:
        return None

def looks_like_youtube_id(name):
    return re.fullmatch(r"[A-Za-z0-9_-]{11}", name) is not None

def infer_meta(src_dir, mp4_file):
    vid = splitext(basename(mp4_file))[0]
    info_path = join(src_dir, vid + ".info.json")
    url, title, category = None, None, None
    if isfile(info_path):
        try:
            with open(info_path, "r", encoding="utf-8") as f:
                info = json.load(f)
            url = info.get("webpage_url") or info.get("original_url")
            title = info.get("title")
            category = info.get("categories", [None])[0] or info.get("category")
        except Exception:
            pass
    if not url and looks_like_youtube_id(vid):
        url = f"https://www.youtube.com/watch?v={vid}"
    return {
        "video_id": vid,
        "url": url or "",
        "title": title or vid,
        "category": category or "Unknown Category"
    }

def write_plain_txt(path, meta, full_text):
    with open(path, "w", encoding="utf-8") as f:
        f.write(meta["title"].strip()+"\n")
        f.write(meta["url"].strip()+"\n")
        f.write(meta["category"].strip()+"\n")
        f.write((full_text or "").strip())

def write_timestamped_txt(path, meta, segments):
    with open(path, "w", encoding="utf-8") as f:
        f.write(meta["title"].strip()+"\n")
        f.write(meta["url"].strip()+"\n")
        f.write(meta["category"].strip()+"\n\n")
        for s in segments:
            start = fmt_ts_srt(float(s.get("start",0.0))).replace(",", ".")
            end   = fmt_ts_srt(float(s.get("end",0.0))).replace(",", ".")
            text  = (s.get("text") or "").strip()
            f.write(f"[{start} → {end}] {text}\n")

def write_srt(path, segments):
    with open(path, "w", encoding="utf-8") as f:
        for i, s in enumerate(segments, 1):
            start = fmt_ts_srt(float(s.get("start",0.0)))
            end   = fmt_ts_srt(float(s.get("end",0.0)))
            text  = (s.get("text") or "").strip()
            f.write(f"{i}\n{start} --> {end}\n{text}\n\n")

def write_json(path, meta, fps, segments):
    with open(path, "w", encoding="utf-8") as jf:
        json.dump({
            "video_id": meta["video_id"],
            "url": meta["url"],
            "title": meta["title"],
            "category": meta["category"],
            "fps": fps,
            "segments": [
                {"start": float(s.get("start",0.0)),
                 "end": float(s.get("end",0.0)),
                 "text": (s.get("text") or "").strip()}
                for s in segments
            ]
        }, jf, ensure_ascii=False)

def main():
    ap = argparse.ArgumentParser(description="Transcribe local .mp4s; auto-uses .info.json if present.")
    ap.add_argument("--src_dir", default="../rawvideos/downloaded_videos")
    ap.add_argument("--out_dir", default="transcripts")
    ap.add_argument("--model", default="turbo")   # tiny/base/small/medium/large-v3
    ap.add_argument("--device", default=None)        # cuda or cpu
    ap.add_argument("--language", default=None)      # e.g., en
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=-1)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    files = [f for f in sorted(listdir(args.src_dir)) if f.lower().endswith(".mp4") and isfile(join(args.src_dir,f))]
    start = max(0, args.start)
    end = None if args.end == -1 else min(args.end, len(files))
    subset = files[start:] if end is None else files[start:end]
    print(f"[INFO] Found {len(files)} mp4s; processing {len(subset)}")

    model = whisper.load_model(args.model, device=args.device) if args.device else whisper.load_model(args.model)

    for fname in subset:
        src_path = join(args.src_dir, fname)
        meta = infer_meta(args.src_dir, fname)
        vid = meta["video_id"]
        out_plain = join(args.out_dir, f"{vid}.txt")
        out_ts = join(args.out_dir, f"{vid}_timestamped.txt")
        out_srt = join(args.out_dir, f"{vid}.srt")
        out_json = join(args.out_dir, f"{vid}.json")

        if all(os.path.exists(p) for p in (out_plain, out_ts, out_srt, out_json)):
            print(f"[SKIP] {vid} (all outputs exist)")
            continue

        print(f"[RUN ] {fname}")
        try:
            t = model.transcribe(src_path, language=args.language)
            text = (t.get("text") or "").strip()
            segments = [
                {"start": float(s.get("start",0.0)),
                 "end": float(s.get("end",0.0)),
                 "text": (s.get("text") or "").strip()}
                for s in t.get("segments", [])
            ]
        except Exception as e:
            print(f"[ERR ] {vid}: {e}")
            continue

        fps = try_get_fps(src_path)

        if not os.path.exists(out_plain): write_plain_txt(out_plain, meta, text)
        if not os.path.exists(out_ts):    write_timestamped_txt(out_ts, meta, segments)
        if not os.path.exists(out_srt):   write_srt(out_srt, segments)
        if not os.path.exists(out_json):  write_json(out_json, meta, fps, segments)
        print(f"[OK  ] {vid}")

if __name__ == "__main__":
    main()
