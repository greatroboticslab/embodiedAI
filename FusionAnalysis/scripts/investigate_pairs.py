import json

data = json.load(open('D:/Antigravity/embodiedAI/FusionAnalysis/results/segment_correlations_embodied/-qCO7NUpLc0_segment_correlation.json'))
segs = data['segments']
total_comments = data['stats']['total_comments']

print(f'Total segments: {len(segs)}')
print(f'Comments: {total_comments}')
print(f'Expected pairs per modality: {len(segs) * total_comments}')
print()

visual_total = 0
transcript_total = 0

for s in segs:
    vis_count = len(s.get('visual_correlations', []))
    trans_count = len(s.get('transcript_correlations', []))
    visual_total += vis_count
    transcript_total += trans_count
    
    if vis_count != total_comments or trans_count != total_comments:
        print(f'Seg {s["segment_index"]}: visual={vis_count}, transcript={trans_count} (expected {total_comments} each)')

print(f'\nTotal visual pairs: {visual_total}')
print(f'Total transcript pairs: {transcript_total}')
