"""
Extract representative screenshots from videos for Paper 2 figures.

Uses Gemini classification data to find prototypical segments,
then extracts frames using ffmpeg.

Screenshot needs:
1. verb_emb_handson + vis_emb_handson — side-by-side comparison
2. handson_demo — clear hands-on demonstration
3. hands_visible + hands_not_visible — hand visibility contrast
4. narration_present + narration_absent — narration contrast
5. visual_heavy + transcript_heavy — channel contrast
6. dataset_examples — one per subgroup (conv, verb_emb, vis_emb)
"""

import json
import os
import subprocess
import sys

DATA_DIR = '/ocean/projects/cis240145p/byler/ben/embodiedAI/FusionAnalysis/results/video_classification'
VIDEO_BASE_CONV = '/ocean/projects/cis240145p/byler/ben/embodiedAI/VideoAnalysis/rawvideos/conventional_videos'
VIDEO_BASE_EMB = '/ocean/projects/cis240145p/byler/ben/embodiedAI/VideoAnalysis/rawvideos/embodied_videos'
OUTPUT_DIR = '/ocean/projects/cis240145p/byler/ben/embodiedAI/Journal_of_Scientific_Reports_Ben_Second_paper/figures'

# Subgroup assignments from master_analysis.csv
import csv
MASTER = '/ocean/projects/cis240145p/byler/ben/embodiedAI/TranscriptAnalysis/results/gemini_analysis/master_analysis.csv'


def load_subgroups():
    """Load video_id -> subgroup mapping."""
    subgroups = {}
    with open(MASTER) as f:
        reader = csv.DictReader(f)
        for row in reader:
            subgroups[row['video_id']] = row['subgroup']
    return subgroups


def load_classifications(subgroup_filter=None):
    """Load all classification JSONs, optionally filtered by subgroup."""
    subgroups = load_subgroups()
    results = []
    for folder, label in [('conventional', 'conventional'), ('embodied', 'embodied')]:
        dirpath = os.path.join(DATA_DIR, folder)
        if not os.path.exists(dirpath):
            continue
        for fname in os.listdir(dirpath):
            if not fname.endswith('.json'):
                continue
            vid_id = fname.replace('.json', '')
            sg = subgroups.get(vid_id, label)
            if subgroup_filter and sg != subgroup_filter:
                continue
            with open(os.path.join(dirpath, fname)) as f:
                data = json.load(f)
            data['subgroup'] = sg
            data['label'] = label
            results.append(data)
    return results


def get_video_path(video_id, label):
    """Get the path to a video file."""
    if label == 'conventional':
        path = os.path.join(VIDEO_BASE_CONV, f'{video_id}.mp4')
    else:
        path = os.path.join(VIDEO_BASE_EMB, f'{video_id}.mp4')
    if os.path.exists(path):
        return path
    return None


def extract_frame(video_path, timestamp_sec, output_path):
    """Extract a single frame at the given timestamp."""
    cmd = [
        'ffmpeg', '-y', '-ss', str(timestamp_sec),
        '-i', video_path, '-frames:v', '1',
        '-q:v', '2', output_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ERROR extracting frame: {result.stderr[:200]}")
        return False
    return True


def find_segments(classifications, content_type=None, hands_visible=None,
                  narration=None, min_density=None, prefer_middle=True):
    """Find segments matching criteria. Returns list of (video_data, segment)."""
    matches = []
    for vid in classifications:
        if not vid.get('segments'):
            continue
        n_segs = len(vid['segments'])
        for seg in vid['segments']:
            if content_type and seg.get('content_type') != content_type:
                continue
            if hands_visible is not None and seg.get('hands_visible') != hands_visible:
                continue
            if narration is not None and seg.get('narration') != narration:
                continue
            if min_density and seg.get('instructional_density', 0) < min_density:
                continue

            # Score: prefer middle segments, high density, high alignment
            score = seg.get('instructional_density', 1) * 10
            score += seg.get('alignment_score', 0) / 10
            if prefer_middle and n_segs > 5:
                rel_pos = seg['segment_index'] / n_segs
                if 0.2 < rel_pos < 0.8:
                    score += 20
            matches.append((score, vid, seg))

    matches.sort(key=lambda x: -x[0])
    return matches


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    subgroups = load_subgroups()
    all_data = load_classifications()

    # Split by subgroup
    conv_data = [d for d in all_data if d.get('subgroup') == 'conventional']
    verb_data = [d for d in all_data if d.get('subgroup') == 'verbally_embodied']
    vis_data = [d for d in all_data if d.get('subgroup') == 'visually_embodied']

    print(f"Loaded: {len(conv_data)} conv, {len(verb_data)} verb_emb, {len(vis_data)} vis_emb")

    screenshots = {}

    # 1. Verbally embodied hands-on (with narration)
    print("\n--- 1. Verbally embodied hands-on ---")
    matches = find_segments(verb_data, content_type='hands_on_demonstration',
                           hands_visible=True, narration=True, min_density=3)
    if matches:
        _, vid, seg = matches[0]
        vid_id = vid['video_id']
        ts = (seg['start'] + seg['end']) / 2
        path = get_video_path(vid_id, vid['label'])
        out = os.path.join(OUTPUT_DIR, 'verb_emb_handson.jpg')
        print(f"  Selected: {vid_id} @ {ts}s (density={seg['instructional_density']}, "
              f"alignment={seg['alignment_score']})")
        if path:
            extract_frame(path, ts, out)
            screenshots['verb_emb_handson'] = out
        else:
            print(f"  Video file not found for {vid_id}")

    # 2. Visually embodied hands-on (no narration)
    print("\n--- 2. Visually embodied hands-on ---")
    matches = find_segments(vis_data, content_type='hands_on_demonstration',
                           hands_visible=True, narration=False)
    if not matches:
        # Fall back to any vis_emb hands-on
        matches = find_segments(vis_data, content_type='hands_on_demonstration',
                               hands_visible=True)
    if matches:
        _, vid, seg = matches[0]
        vid_id = vid['video_id']
        ts = (seg['start'] + seg['end']) / 2
        path = get_video_path(vid_id, vid['label'])
        out = os.path.join(OUTPUT_DIR, 'vis_emb_handson.jpg')
        print(f"  Selected: {vid_id} @ {ts}s (narration={seg['narration']}, "
              f"density={seg['instructional_density']})")
        if path:
            extract_frame(path, ts, out)
            screenshots['vis_emb_handson'] = out
        else:
            print(f"  Video file not found for {vid_id}")

    # 3. Hands visible (any subgroup, hands-on content)
    print("\n--- 3. Hands visible ---")
    matches = find_segments(all_data, content_type='hands_on_demonstration',
                           hands_visible=True, min_density=4)
    if matches:
        # Pick one that is different from #1 and #2
        used_vids = {screenshots.get('verb_emb_handson', ''), screenshots.get('vis_emb_handson', '')}
        for _, vid, seg in matches:
            vid_id = vid['video_id']
            path = get_video_path(vid_id, vid['label'])
            if path and path not in used_vids:
                ts = (seg['start'] + seg['end']) / 2
                out = os.path.join(OUTPUT_DIR, 'hands_visible.jpg')
                print(f"  Selected: {vid_id} @ {ts}s (subgroup={vid['subgroup']})")
                extract_frame(path, ts, out)
                screenshots['hands_visible'] = out
                break

    # 4. Hands NOT visible (presenter talking or slides)
    print("\n--- 4. Hands not visible ---")
    matches = find_segments(all_data, content_type='presenter_talking',
                           hands_visible=False, narration=True, min_density=3)
    if not matches:
        matches = find_segments(all_data, content_type='slide_or_powerpoint',
                               hands_visible=False, narration=True)
    if matches:
        _, vid, seg = matches[0]
        vid_id = vid['video_id']
        ts = (seg['start'] + seg['end']) / 2
        path = get_video_path(vid_id, vid['label'])
        out = os.path.join(OUTPUT_DIR, 'hands_not_visible.jpg')
        print(f"  Selected: {vid_id} @ {ts}s (type={seg['content_type']}, "
              f"subgroup={vid['subgroup']})")
        if path:
            extract_frame(path, ts, out)
            screenshots['hands_not_visible'] = out

    # 5. Narration present (hands-on with narration)
    print("\n--- 5. Narration present ---")
    # Reuse verb_emb_handson if available
    if 'verb_emb_handson' in screenshots:
        print(f"  Reusing verb_emb_handson")
        screenshots['narration_present'] = screenshots['verb_emb_handson']
    else:
        matches = find_segments(all_data, narration=True, min_density=4)
        if matches:
            _, vid, seg = matches[0]
            vid_id = vid['video_id']
            ts = (seg['start'] + seg['end']) / 2
            path = get_video_path(vid_id, vid['label'])
            out = os.path.join(OUTPUT_DIR, 'narration_present.jpg')
            if path:
                extract_frame(path, ts, out)
                screenshots['narration_present'] = out

    # 6. Narration absent (visually embodied, hands-on, no narration)
    print("\n--- 6. Narration absent ---")
    if 'vis_emb_handson' in screenshots:
        print(f"  Reusing vis_emb_handson")
        screenshots['narration_absent'] = screenshots['vis_emb_handson']
    else:
        matches = find_segments(vis_data, narration=False, hands_visible=True)
        if matches:
            _, vid, seg = matches[0]
            vid_id = vid['video_id']
            ts = (seg['start'] + seg['end']) / 2
            path = get_video_path(vid_id, vid['label'])
            out = os.path.join(OUTPUT_DIR, 'narration_absent.jpg')
            if path:
                extract_frame(path, ts, out)
                screenshots['narration_absent'] = out

    # 7. Visual-heavy content (hands-on, high engagement)
    print("\n--- 7. Visual-heavy content ---")
    # Already covered by hands-on screenshots above
    if 'hands_visible' in screenshots:
        print(f"  Reusing hands_visible")
        screenshots['visual_heavy'] = screenshots['hands_visible']

    # 8. Transcript-heavy content (slides or diagram with narration)
    print("\n--- 8. Transcript-heavy content ---")
    matches = find_segments(conv_data, content_type='slide_or_powerpoint',
                           narration=True, min_density=4)
    if not matches:
        matches = find_segments(conv_data, content_type='diagram_or_whiteboard',
                               narration=True, min_density=3)
    if matches:
        _, vid, seg = matches[0]
        vid_id = vid['video_id']
        ts = (seg['start'] + seg['end']) / 2
        path = get_video_path(vid_id, vid['label'])
        out = os.path.join(OUTPUT_DIR, 'transcript_heavy.jpg')
        print(f"  Selected: {vid_id} @ {ts}s (type={seg['content_type']})")
        if path:
            extract_frame(path, ts, out)
            screenshots['transcript_heavy'] = out

    # 9. Dataset examples — one per subgroup
    print("\n--- 9. Dataset examples ---")
    for sg_name, sg_data, sg_label in [('conv', conv_data, 'conventional'),
                                        ('verb_emb', verb_data, 'embodied'),
                                        ('vis_emb', vis_data, 'embodied')]:
        # For conv: diagram or presenter
        if sg_name == 'conv':
            matches = find_segments(sg_data, content_type='diagram_or_whiteboard',
                                   narration=True, min_density=3)
            if not matches:
                matches = find_segments(sg_data, content_type='presenter_talking',
                                       narration=True)
        elif sg_name == 'verb_emb':
            matches = find_segments(sg_data, content_type='hands_on_demonstration',
                                   hands_visible=True, narration=True, min_density=3)
        else:  # vis_emb
            matches = find_segments(sg_data, content_type='hands_on_demonstration',
                                   hands_visible=True)

        if matches:
            # Try to pick a different video from ones already used
            used_paths = set(screenshots.values())
            selected = None
            for _, vid, seg in matches[:10]:
                vid_id = vid['video_id']
                label = 'conventional' if sg_name == 'conv' else 'embodied'
                path = get_video_path(vid_id, label)
                out = os.path.join(OUTPUT_DIR, f'dataset_{sg_name}.jpg')
                if path and out not in used_paths:
                    selected = (vid, seg, path, out)
                    break
            if not selected and matches:
                _, vid, seg = matches[0]
                vid_id = vid['video_id']
                label = 'conventional' if sg_name == 'conv' else 'embodied'
                path = get_video_path(vid_id, label)
                out = os.path.join(OUTPUT_DIR, f'dataset_{sg_name}.jpg')
                selected = (vid, seg, path, out)

            if selected:
                vid, seg, path, out = selected
                ts = (seg['start'] + seg['end']) / 2
                print(f"  {sg_name}: {vid['video_id']} @ {ts}s (type={seg['content_type']})")
                extract_frame(path, ts, out)
                screenshots[f'dataset_{sg_name}'] = out

    print("\n--- Summary ---")
    for name, path in sorted(screenshots.items()):
        exists = os.path.exists(path) if path else False
        print(f"  {name}: {path} (exists={exists})")


if __name__ == '__main__':
    main()
