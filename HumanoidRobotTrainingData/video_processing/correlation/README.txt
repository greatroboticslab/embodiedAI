Usage:

python correlate_comments.py /path/to/frames \
  /path/to/all_video_comments.docx \
  --min_score 65 --top_k 3

python correlate_comments_3frames.py /path/to/frames \
  /path/to/all_video_comments.docx \
  --min_score 65 --top_k 3 --group_stride 3


python section_correlation.py \
  /path/to/frames_root \
  /path/to/comments.docx \
  --min_score 60 --top_k 5 \
  --seg_penalty 18 --seg_min_size 6 \
  --retrieval_topk 50 --rep_k 3
Want ~10 sections? use --seg_n_sections 10 (this overrides the penalty).
Correlation display/thresholds

--min_score 60
Cutoff (0–100) from the LLM scorer. Only comment–section pairs scoring ≥ 60 are shown/kept.
Raise it → higher precision, fewer matches. Lower it → more matches, more noise.
Typical: 55–70. If reasons look weak, bump to 65–75.

--top_k 5
How many kept matches per section to show in the DOCX. Doesn’t affect scoring, just the report length.
Use 3–10 depending on how dense your comments are.

Sectioning (change-point detection on caption embeddings)

--seg_penalty 18
Penalty for making a cut (PELT). Higher = fewer, larger sections. Lower = more, smaller sections.
Think of it as a granularity knob. Try ~12–30. If sections are too short/choppy, increase; if they merge different steps, decrease.

--seg_min_size 6
Minimum frames per section (hard floor). Prevents tiny sections even if the algorithm sees a change.
Raise it (e.g., 8–12) if you sample frames densely or want chunkier sections.

Retrieval (pre-filter comments before LLM scoring)

--retrieval_topk 50
For each section, take the Top-50 nearest comments by embedding similarity, then ask the LLM only for those.
Bigger = more recall but slower (more LLM calls). Smaller = faster but you might miss some.
Rule of thumb:

Few comments (<500 total): 100–200 or even 0 (score all).

Many comments (1k–10k): 30–100.
Set 0 to skip retrieval and score all comments (slow).

Report visuals

--rep_k 3
Number of representative thumbnails per section shown in the DOCX (picked evenly across the section).
3 is a good default; use 2–6 depending on space.

Quick tuning guide

Too many tiny sections → increase --seg_penalty, or increase --seg_min_size.

Different steps merged → decrease --seg_penalty.

Matches feel generic → raise --min_score; optionally lower --retrieval_topk to focus on closest comments.

Missing good matches → lower --min_score a bit and/or increase --retrieval_topk.

Report too long → lower --top_k and/or --rep_k.



python extract_topics.py \
  --src_dir ../transcription/transcripts \
  --out_dir topics \
  --model <your_model_name> \
  --whole_video_outline \  #Pass the whole video transcript at once instead of chunking it
  --outline_simple \  #Use segment ID instead of timestamps
  --whole_max_chars 80000 \
  --outline_min 6 --outline_max 12

  To use topic_comment_correlation.py
  # Full run
python topic_comment_correlation.py /path/to/videos_root comments.docx \
  --topics_root /path/to/topics --model llama3.1 --min_score 60 --top_k 5

# Rebuild reports without rescoring
python topic_comment_correlation.py /path/to/videos_root comments.docx \
  --topics_root /path/to/topics --rebuild


