import json

data = json.load(open('D:/Antigravity/embodiedAI/FusionAnalysis/results/segment_correlations_embodied/-qCO7NUpLc0_segment_correlation.json'))
segs = data['segments']

print(f'Total segments: {len(segs)}')
print(f'Comments: {data["stats"]["total_comments"]}')
print()

# Find segments with no visual correlations
no_visual = []
no_transcript = []

for seg in segs:
    vis_count = len(seg.get('visual_correlations', []))
    trans_count = len(seg.get('transcript_correlations', []))
    
    if vis_count == 0:
        no_visual.append(seg)
    if trans_count == 0:
        no_transcript.append(seg)

print(f'Segments with NO visual correlations: {len(no_visual)}')
if no_visual:
    for seg in no_visual:
        print(f'\n  Segment {seg["segment_index"]} ({seg["start_time"]}-{seg["end_time"]}s):')
        print(f'    Frames: {seg.get("frames", [])}')
        print(f'    Visual summary: {seg.get("visual_summary", "")[:100]}...')
        print(f'    Visual correlations: {len(seg.get("visual_correlations", []))}')

print(f'\nSegments with NO transcript correlations: {len(no_transcript)}')
if no_transcript:
    for seg in no_transcript[:3]:  # Show first 3
        print(f'\n  Segment {seg["segment_index"]} ({seg["start_time"]}-{seg["end_time"]}s):')
        print(f'    Transcript: "{seg.get("transcript", "")}"')
        print(f'    Transcript correlations: {len(seg.get("transcript_correlations", []))}')
