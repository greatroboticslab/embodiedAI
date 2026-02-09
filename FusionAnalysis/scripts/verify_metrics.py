import pandas as pd

df = pd.read_excel('D:/Antigravity/embodiedAI/FusionAnalysis/results/fusion_metrics_embodied_test.xlsx')
print(f'Shape: {df.shape}')
print(f'\nColumns: {list(df.columns)}')
print(f'\nFirst row (video -qCO7NUpLc0):')
row = df[df['video_id'] == '-qCO7NUpLc0'].iloc[0]
print(f'Segments: {row["num_segments"]}')
print(f'Comments: {row["total_comments_used"]}')
print(f'Expected pairs: {row["num_segments"]} x {row["total_comments_used"]} = {row["num_segments"] * row["total_comments_used"]}')
print(f'\nVisual total pairs: {row["visual_total_pairs"]}')
print(f'Visual correlated: {row["visual_correlated_count"]} ({row["visual_correlated_pct"]:.2f}%)')
print(f'\nTranscript total pairs: {row["transcript_total_pairs"]}')
print(f'Transcript correlated: {row["transcript_correlated_count"]} ({row["transcript_correlated_pct"]:.2f}%)')
print(f'\nUnion total pairs: {row["union_total_pairs"]}')
print(f'Union correlated: {row["union_correlated_count"]} ({row["union_correlated_pct"]:.2f}%)')
