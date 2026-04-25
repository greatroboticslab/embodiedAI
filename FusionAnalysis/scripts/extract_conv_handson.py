"""
Extract two targeted screenshots for Paper 2 revision:
1. Conventional video hands-on demo (for Figure 3)
2. Hands-not-visible segment without a face (for Figure 6 replacement)
"""

import json
import os
import subprocess

DATA_DIR = '/ocean/projects/cis240145p/byler/ben/embodiedAI/FusionAnalysis/results/video_classification'
VIDEO_BASE_CONV = '/ocean/projects/cis240145p/byler/ben/embodiedAI/VideoAnalysis/rawvideos/conventional_videos'
VIDEO_BASE_EMB = '/ocean/projects/cis240145p/byler/ben/embodiedAI/VideoAnalysis/rawvideos/embodied_videos'
OUTPUT_DIR = '/ocean/projects/cis240145p/byler/ben/embodiedAI/Journal_of_Scientific_Reports_Ben_Second_paper/figures'


def load_classifications(folder):
    results = []
    dirpath = os.path.join(DATA_DIR, folder)
    for fname in sorted(os.listdir(dirpath)):
        if not fname.endswith('.json'):
            continue
        vid_id = fname.replace('.json', '')
        with open(os.path.join(dirpath, fname)) as f:
            data = json.load(f)
        data['video_id'] = vid_id
        data['folder'] = folder
        results.append(data)
    return results


def get_video_path(video_id, folder):
    if folder == 'conventional':
        path = os.path.join(VIDEO_BASE_CONV, f'{video_id}.mp4')
    else:
        path = os.path.join(VIDEO_BASE_EMB, f'{video_id}.mp4')
    return path if os.path.exists(path) else None


def extract_frame(video_path, timestamp_sec, output_path):
    cmd = [
        'ffmpeg', '-y', '-ss', str(timestamp_sec),
        '-i', video_path, '-frames:v', '1',
        '-q:v', '2', output_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ERROR: {result.stderr[:200]}")
        return False
    return True


def find_conv_handson(conv_data):
    """Find best conventional video hands-on demonstration segment."""
    candidates = []
    for vid in conv_data:
        vid_id = vid['video_id']
        segs = vid.get('segments', [])
        n_segs = len(segs)
        for seg in segs:
            if seg.get('content_type') != 'hands_on_demonstration':
                continue
            if not seg.get('hands_visible', False):
                continue

            score = seg.get('instructional_density', 1) * 10
            score += seg.get('alignment_score', 0) / 10
            # Prefer middle segments
            rel_pos = seg['segment_index'] / max(n_segs, 1)
            if 0.2 < rel_pos < 0.8:
                score += 15
            # Prefer videos with more segments
            if n_segs >= 20:
                score += 5

            candidates.append((score, vid, seg))

    candidates.sort(key=lambda x: -x[0])
    return candidates


def find_no_face_no_hands(all_data):
    """Find a segment with hands NOT visible, content type that doesn't show a face."""
    # Content types that show content, not the instructor's face
    safe_types = {'diagram_or_whiteboard', 'slide_or_powerpoint', 'screen_or_software',
                  'animation_or_graphic'}
    candidates = []
    for vid in all_data:
        vid_id = vid['video_id']
        segs = vid.get('segments', [])
        n_segs = len(segs)
        for seg in segs:
            if seg.get('hands_visible', True):
                continue
            if seg.get('content_type') not in safe_types:
                continue

            score = seg.get('instructional_density', 1) * 10
            score += seg.get('alignment_score', 0) / 10
            # Prefer diagram/whiteboard (most relevant for "verbal instruction" contrast)
            if seg.get('content_type') == 'diagram_or_whiteboard':
                score += 20
            elif seg.get('content_type') == 'slide_or_powerpoint':
                score += 10
            # Prefer narration present (to contrast with hands-visible which is physical)
            if seg.get('narration', False):
                score += 10
            # Prefer middle segments
            rel_pos = seg['segment_index'] / max(n_segs, 1)
            if 0.2 < rel_pos < 0.8:
                score += 15

            candidates.append((score, vid, seg))

    candidates.sort(key=lambda x: -x[0])
    return candidates


def main():
    conv_data = load_classifications('conventional')
    emb_data = load_classifications('embodied')
    all_data = conv_data + emb_data
    print(f"Loaded: {len(conv_data)} conventional, {len(emb_data)} embodied videos\n")

    # === 1. Conventional hands-on screenshot for Figure 3 ===
    print("=" * 60)
    print("Figure 3: Conventional hands-on demonstration")
    print("=" * 60)

    cands = find_conv_handson(conv_data)
    print(f"Found {len(cands)} candidate segments")

    for i, (score, vid, seg) in enumerate(cands[:5]):
        vid_id = vid['video_id']
        path = get_video_path(vid_id, vid['folder'])
        ts = (seg['start'] + seg['end']) / 2
        print(f"  #{i+1} score={score:.1f} video={vid_id} seg={seg['segment_index']} "
              f"t={ts:.1f}s type={seg['content_type']} "
              f"density={seg.get('instructional_density')} "
              f"hands={seg.get('hands_visible')} narr={seg.get('narration')} "
              f"path_exists={path is not None}")

    # Extract the best one
    for score, vid, seg in cands:
        vid_id = vid['video_id']
        path = get_video_path(vid_id, vid['folder'])
        if not path:
            continue
        ts = (seg['start'] + seg['end']) / 2
        out = os.path.join(OUTPUT_DIR, 'fig1_conv_handson.jpg')
        print(f"\n  Extracting: {vid_id} seg {seg['segment_index']} at t={ts:.1f}s")
        print(f"    Content: {seg['content_type']}, density={seg.get('instructional_density')}")
        if extract_frame(path, ts, out):
            size = os.path.getsize(out) / 1024
            print(f"    Saved: {out} ({size:.0f}KB)")
        break

    # === 2. Hands-not-visible without face for Figure 6 ===
    print("\n" + "=" * 60)
    print("Figure 6: Hands NOT visible, no face (privacy-safe)")
    print("=" * 60)

    cands = find_no_face_no_hands(all_data)
    print(f"Found {len(cands)} candidate segments")

    for i, (score, vid, seg) in enumerate(cands[:5]):
        vid_id = vid['video_id']
        path = get_video_path(vid_id, vid['folder'])
        ts = (seg['start'] + seg['end']) / 2
        print(f"  #{i+1} score={score:.1f} video={vid_id} seg={seg['segment_index']} "
              f"t={ts:.1f}s type={seg['content_type']} "
              f"density={seg.get('instructional_density')} "
              f"hands={seg.get('hands_visible')} narr={seg.get('narration')} "
              f"path_exists={path is not None}")

    # Extract the best one
    for score, vid, seg in cands:
        vid_id = vid['video_id']
        path = get_video_path(vid_id, vid['folder'])
        if not path:
            continue
        ts = (seg['start'] + seg['end']) / 2
        out = os.path.join(OUTPUT_DIR, 'fig5_hands_not_visible.jpg')
        print(f"\n  Extracting: {vid_id} seg {seg['segment_index']} at t={ts:.1f}s")
        print(f"    Content: {seg['content_type']}, density={seg.get('instructional_density')}")
        if extract_frame(path, ts, out):
            size = os.path.getsize(out) / 1024
            print(f"    Saved: {out} ({size:.0f}KB)")
        break


if __name__ == '__main__':
    main()
