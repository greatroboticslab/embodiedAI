"""
Extract figure-specific, non-repeating screenshots for Paper 2.

Each figure gets unique screenshots selected to illustrate its specific point.
No screenshot is reused across figures.

Figure inventory (13 total screenshots):
  fig:content_dist     — 2: verb_emb hands-on vs vis_emb hands-on
  fig:video_scatter    — 1: verb_emb video showing hands-on + narration
  fig:hands_effect     — 2: hands visible vs hands not visible
  tab:narration        — 2: narrated instruction vs silent demonstration
  fig:temporal         — 3: early/middle/late from SAME video
  fig:dataset_examples — 3: one per subgroup (conv, verb_emb, vis_emb)
"""

import csv
import json
import os
import subprocess
import sys

DATA_DIR = '/ocean/projects/cis240145p/byler/ben/embodiedAI/FusionAnalysis/results/video_classification'
VIDEO_BASE_CONV = '/ocean/projects/cis240145p/byler/ben/embodiedAI/VideoAnalysis/rawvideos/conventional_videos'
VIDEO_BASE_EMB = '/ocean/projects/cis240145p/byler/ben/embodiedAI/VideoAnalysis/rawvideos/embodied_videos'
OUTPUT_DIR = '/ocean/projects/cis240145p/byler/ben/embodiedAI/Journal_of_Scientific_Reports_Ben_Second_paper/figures'
MASTER = '/ocean/projects/cis240145p/byler/ben/embodiedAI/TranscriptAnalysis/results/gemini_analysis/master_analysis.csv'

# Track all used (video_id, segment_index) pairs globally
used_segments = set()
# Track all used video_ids per figure to avoid same-video in same figure
used_videos_per_figure = {}


def load_subgroups():
    subgroups = {}
    with open(MASTER) as f:
        for row in csv.DictReader(f):
            subgroups[row['video_id']] = row['subgroup']
    return subgroups


def load_classifications():
    subgroups = load_subgroups()
    results = []
    for folder, label in [('conventional', 'conventional'), ('embodied', 'embodied')]:
        dirpath = os.path.join(DATA_DIR, folder)
        if not os.path.exists(dirpath):
            continue
        for fname in sorted(os.listdir(dirpath)):
            if not fname.endswith('.json'):
                continue
            vid_id = fname.replace('.json', '')
            sg = subgroups.get(vid_id, label)
            with open(os.path.join(dirpath, fname)) as f:
                data = json.load(f)
            data['subgroup'] = sg
            data['label'] = label
            results.append(data)
    return results


def get_video_path(video_id, label):
    if label == 'conventional':
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


def find_best_segment(videos, figure_name, content_type=None, hands_visible=None,
                      narration=None, min_density=None, prefer_position=None,
                      subgroup_filter=None):
    """
    Find the best unused segment matching criteria.
    prefer_position: 'early' (0-33%), 'middle' (33-67%), 'late' (67-100%), or None
    Returns (video_data, segment, score) or None.
    """
    candidates = []
    for vid in videos:
        if subgroup_filter and vid.get('subgroup') != subgroup_filter:
            continue
        segs = vid.get('segments', [])
        if not segs:
            continue
        n_segs = len(segs)
        vid_id = vid['video_id']

        for seg in segs:
            key = (vid_id, seg['segment_index'])
            if key in used_segments:
                continue

            if content_type and seg.get('content_type') != content_type:
                continue
            if hands_visible is not None and seg.get('hands_visible') != hands_visible:
                continue
            if narration is not None and seg.get('narration') != narration:
                continue
            if min_density and seg.get('instructional_density', 0) < min_density:
                continue

            # Score based on instructional quality
            score = seg.get('instructional_density', 1) * 10
            score += seg.get('alignment_score', 0) / 10

            # Position preference
            rel_pos = seg['segment_index'] / max(n_segs, 1)
            if prefer_position == 'early' and rel_pos < 0.33:
                score += 30
            elif prefer_position == 'middle' and 0.33 <= rel_pos <= 0.67:
                score += 30
            elif prefer_position == 'late' and rel_pos > 0.67:
                score += 30
            elif prefer_position is None and 0.2 < rel_pos < 0.8:
                score += 15  # mild middle preference when no position specified

            # Prefer videos with enough segments for variety
            if n_segs >= 20:
                score += 5

            candidates.append((score, vid, seg))

    candidates.sort(key=lambda x: -x[0])
    return candidates


def pick_and_extract(candidates, figure_name, output_filename, description):
    """Pick the best candidate, mark it used, and extract the frame."""
    fig_used = used_videos_per_figure.setdefault(figure_name, set())

    for score, vid, seg in candidates:
        vid_id = vid['video_id']
        path = get_video_path(vid_id, vid['label'])
        if not path:
            continue
        # Avoid reusing same video within a figure (unless temporal)
        if vid_id in fig_used and figure_name != 'temporal':
            continue

        key = (vid_id, seg['segment_index'])
        ts = (seg['start'] + seg['end']) / 2
        out = os.path.join(OUTPUT_DIR, output_filename)

        print(f"  {description}")
        print(f"    Video: {vid_id} (subgroup={vid['subgroup']})")
        print(f"    Segment {seg['segment_index']}: {seg['start']:.0f}-{seg['end']:.0f}s "
              f"(t={ts:.1f}s, type={seg['content_type']}, "
              f"density={seg['instructional_density']}, hands={seg['hands_visible']}, "
              f"narr={seg['narration']})")

        if extract_frame(path, ts, out):
            used_segments.add(key)
            fig_used.add(vid_id)
            return (vid, seg, out)

    print(f"  WARNING: No suitable candidate found for {output_filename}")
    return None


def extract_temporal_from_same_video(videos):
    """
    For fig:temporal, find a single video with diverse content across
    early/middle/late portions and extract one frame from each third.
    Prefer a verbally embodied video with many segments.
    """
    subgroups = load_subgroups()
    best_video = None
    best_score = -1

    for vid in videos:
        segs = vid.get('segments', [])
        n = len(segs)
        if n < 30:
            continue  # need enough segments for meaningful thirds

        vid_id = vid['video_id']
        sg = vid.get('subgroup', '')

        # Prefer verbally embodied (has both visual and transcript engagement)
        sg_bonus = 20 if sg == 'verbally_embodied' else 0

        # Check that each third has usable segments
        early = [s for s in segs if s['segment_index'] / n < 0.33]
        middle = [s for s in segs if 0.33 <= s['segment_index'] / n <= 0.67]
        late = [s for s in segs if s['segment_index'] / n > 0.67]

        # Want high-density segments available in each third
        early_dens = [s for s in early if s.get('instructional_density', 0) >= 3]
        mid_dens = [s for s in middle if s.get('instructional_density', 0) >= 3]
        late_dens = [s for s in late if s.get('instructional_density', 0) >= 3]

        if not (early_dens and mid_dens and late_dens):
            continue

        # Score: prefer diverse content types across the three thirds
        types_early = set(s['content_type'] for s in early_dens)
        types_mid = set(s['content_type'] for s in mid_dens)
        types_late = set(s['content_type'] for s in late_dens)
        diversity = len(types_early | types_mid | types_late)

        score = sg_bonus + diversity * 5 + n / 10
        if score > best_score:
            path = get_video_path(vid_id, vid['label'])
            if path:
                best_score = score
                best_video = vid

    return best_video


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_data = load_classifications()

    conv_data = [d for d in all_data if d.get('subgroup') == 'conventional']
    verb_data = [d for d in all_data if d.get('subgroup') == 'verbally_embodied']
    vis_data = [d for d in all_data if d.get('subgroup') == 'visually_embodied']

    print(f"Loaded: {len(conv_data)} conv, {len(verb_data)} verb_emb, {len(vis_data)} vis_emb")
    print(f"Total videos: {len(all_data)}\n")

    # ================================================================
    # 1. fig:content_dist — verb_emb vs vis_emb hands-on comparison
    # ================================================================
    print("=" * 60)
    print("FIGURE: content_dist (hands-on demo proportion by subgroup)")
    print("=" * 60)

    # Verbally embodied: hands-on with narration (their characteristic mode)
    cands = find_best_segment(verb_data, 'content_dist',
                              content_type='hands_on_demonstration',
                              hands_visible=True, narration=True, min_density=3)
    pick_and_extract(cands, 'content_dist', 'fig1_verb_emb_handson.jpg',
                     'Verbally embodied hands-on demo (narrated)')

    # Visually embodied: hands-on without narration (their characteristic mode)
    cands = find_best_segment(vis_data, 'content_dist',
                              content_type='hands_on_demonstration',
                              hands_visible=True, narration=False)
    if not cands:
        cands = find_best_segment(vis_data, 'content_dist',
                                  content_type='hands_on_demonstration',
                                  hands_visible=True)
    pick_and_extract(cands, 'content_dist', 'fig1_vis_emb_handson.jpg',
                     'Visually embodied hands-on demo (silent)')

    # ================================================================
    # 2. fig:video_scatter — verbally embodied video-level relationship
    # ================================================================
    print("\n" + "=" * 60)
    print("FIGURE: video_scatter (video-level hands-on vs visual engagement)")
    print("=" * 60)

    # A different verb_emb hands-on video than fig:content_dist
    cands = find_best_segment(verb_data, 'video_scatter',
                              content_type='hands_on_demonstration',
                              hands_visible=True, narration=True, min_density=4)
    pick_and_extract(cands, 'video_scatter', 'fig4_verb_emb_scatter.jpg',
                     'Verbally embodied instructor demonstrating (high-engagement example)')

    # ================================================================
    # 3. fig:hands_effect — visible hands vs not visible
    # ================================================================
    print("\n" + "=" * 60)
    print("FIGURE: hands_effect (hand visibility dual-channel tradeoff)")
    print("=" * 60)

    # Hands visible: clear hands-on work
    cands = find_best_segment(all_data, 'hands_effect',
                              content_type='hands_on_demonstration',
                              hands_visible=True, min_density=3)
    pick_and_extract(cands, 'hands_effect', 'fig5_hands_visible.jpg',
                     'Segment with visible hands (physical manipulation)')

    # Hands NOT visible: presenter/slides/diagram
    cands = find_best_segment(all_data, 'hands_effect',
                              content_type='presenter_talking',
                              hands_visible=False, narration=True, min_density=3)
    if not cands:
        cands = find_best_segment(all_data, 'hands_effect',
                                  content_type='diagram_or_whiteboard',
                                  hands_visible=False, narration=True)
    pick_and_extract(cands, 'hands_effect', 'fig5_hands_not_visible.jpg',
                     'Segment without visible hands (verbal instruction)')

    # ================================================================
    # 4. tab:narration — narration present vs absent
    # ================================================================
    print("\n" + "=" * 60)
    print("TABLE: narration (narration present vs absent)")
    print("=" * 60)

    # Narration present: conventional or verb_emb instructor explaining
    cands = find_best_segment(all_data, 'narration',
                              narration=True, min_density=4,
                              content_type='diagram_or_whiteboard')
    if not cands:
        cands = find_best_segment(all_data, 'narration',
                                  narration=True, min_density=3)
    pick_and_extract(cands, 'narration', 'fig6_narration_present.jpg',
                     'Narrated instruction (verbal channel active)')

    # Narration absent: vis_emb silent demo
    cands = find_best_segment(vis_data, 'narration',
                              narration=False, hands_visible=True,
                              content_type='hands_on_demonstration')
    pick_and_extract(cands, 'narration', 'fig6_narration_absent.jpg',
                     'Silent demonstration (no narration)')

    # ================================================================
    # 5. fig:temporal — early/middle/late from SAME video
    # ================================================================
    print("\n" + "=" * 60)
    print("FIGURE: temporal (early/middle/late engagement dynamics)")
    print("=" * 60)

    temporal_vid = extract_temporal_from_same_video(all_data)
    if temporal_vid:
        vid_id = temporal_vid['video_id']
        segs = temporal_vid['segments']
        n = len(segs)
        path = get_video_path(vid_id, temporal_vid['label'])
        print(f"  Selected video: {vid_id} (subgroup={temporal_vid['subgroup']}, "
              f"{n} segments, {temporal_vid.get('duration',0):.0f}s)")

        for position, label, frac_lo, frac_hi in [
            ('early', 'Early segment (0-33%)', 0.0, 0.33),
            ('middle', 'Middle segment (33-67%)', 0.33, 0.67),
            ('late', 'Late segment (67-100%)', 0.67, 1.0),
        ]:
            pool = [s for s in segs
                    if frac_lo <= s['segment_index'] / n < frac_hi
                    and s.get('instructional_density', 0) >= 3
                    and (vid_id, s['segment_index']) not in used_segments]
            # Sort by density, then pick middle of the pool
            pool.sort(key=lambda s: -s.get('instructional_density', 0))
            if pool:
                seg = pool[0]
                ts = (seg['start'] + seg['end']) / 2
                out_name = f'fig7_temporal_{position}.jpg'
                out = os.path.join(OUTPUT_DIR, out_name)
                print(f"  {label}: seg {seg['segment_index']}, "
                      f"t={ts:.1f}s, type={seg['content_type']}, "
                      f"density={seg['instructional_density']}")
                extract_frame(path, ts, out)
                used_segments.add((vid_id, seg['segment_index']))
            else:
                print(f"  WARNING: no suitable {position} segment found")
    else:
        print("  WARNING: no suitable video found for temporal figure")

    # ================================================================
    # 6. fig:dataset_examples — one per subgroup (Methods section)
    # ================================================================
    print("\n" + "=" * 60)
    print("FIGURE: dataset_examples (representative frames, Methods)")
    print("=" * 60)

    # Conventional: diagram/whiteboard with narration
    cands = find_best_segment(conv_data, 'dataset_examples',
                              content_type='diagram_or_whiteboard',
                              narration=True, min_density=3)
    if not cands:
        cands = find_best_segment(conv_data, 'dataset_examples',
                                  content_type='presenter_talking',
                                  narration=True)
    pick_and_extract(cands, 'dataset_examples', 'fig9_dataset_conv.jpg',
                     'Conventional: instructor with whiteboard/diagram')

    # Verbally embodied: hands-on with narration
    cands = find_best_segment(verb_data, 'dataset_examples',
                              content_type='hands_on_demonstration',
                              hands_visible=True, narration=True, min_density=3)
    pick_and_extract(cands, 'dataset_examples', 'fig9_dataset_verb_emb.jpg',
                     'Verbally embodied: narrated hands-on demonstration')

    # Visually embodied: hands-on without narration
    cands = find_best_segment(vis_data, 'dataset_examples',
                              content_type='hands_on_demonstration',
                              hands_visible=True, narration=False)
    if not cands:
        cands = find_best_segment(vis_data, 'dataset_examples',
                                  content_type='hands_on_demonstration',
                                  hands_visible=True)
    pick_and_extract(cands, 'dataset_examples', 'fig9_dataset_vis_emb.jpg',
                     'Visually embodied: silent hands-on demonstration')

    # ================================================================
    # Summary
    # ================================================================
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total unique (video, segment) pairs used: {len(used_segments)}")
    print(f"Expected: 13")
    print()

    expected_files = [
        'fig1_verb_emb_handson.jpg', 'fig1_vis_emb_handson.jpg',
        'fig4_verb_emb_scatter.jpg',
        'fig5_hands_visible.jpg', 'fig5_hands_not_visible.jpg',
        'fig6_narration_present.jpg', 'fig6_narration_absent.jpg',
        'fig7_temporal_early.jpg', 'fig7_temporal_middle.jpg', 'fig7_temporal_late.jpg',
        'fig9_dataset_conv.jpg', 'fig9_dataset_verb_emb.jpg', 'fig9_dataset_vis_emb.jpg',
    ]

    for fname in expected_files:
        fpath = os.path.join(OUTPUT_DIR, fname)
        exists = os.path.exists(fpath)
        size = os.path.getsize(fpath) if exists else 0
        status = f"{size/1024:.0f}KB" if exists else "MISSING"
        print(f"  {fname}: {status}")


if __name__ == '__main__':
    main()
