# Transcript Analysis of Videos
This project generate transcripts of videos and correlation with comments.

### To generate transcripts

    cd TranscriptAnalysis/scripts
    conda activate whisper
    python transcribe_local_folder.py --src_dir path/to/videos/root

### To extract topics from transcripts

    cd TranscriptAnalysis/scripts
    conda activate videoanalysis
    python extract_topics.py \
    --src_dir path/to/transcripts \
    --outline_simple \
    --whole_video_outline \
    --force

Select --outline_simple to uses segments instead of timestamps in transcripts.
Select --whole_video_outline to pass entire transcript instead of chunked segments.
Select --force to recompute even if outputs exist.

### To generate docx files for topics and transcripts

    cd TranscriptAnalysis/scripts
    conda activate videoanalysis
    python build_topic_transcript_docx.py \
    --transcripts_root path/to/transcripts \
    --topics_root path/to/topics \
    --out_dir output/path

### To Correlate Comments with Topics

    cd TranscriptAnalysis/scripts
    conda activate videoanalysis
    python topic_comment_correlation.py 

### To Correlate YouTube Comments with Topics (JSON input)

This script is similar to `topic_comment_correlation.py` but accepts a folder of JSON/JSONL comment files (downloaded from YouTube) instead of a single DOCX file.

    cd TranscriptAnalysis/scripts
    conda activate videoanalysis
    python topic_comment_correlation_youtube.py \
      --comments_dir ../../VideoAnalysis/data/comments \
      --results_root ../results_youtube 

### To Generate Metrics

    cd TranscriptAnalysis/scripts
    # Ensure you are in the correct environment (e.g. videoanalysis)
    
    # 1. Run collection
    # Conventional
    python embodiment_project_collect_metrics.py \
      --videos_root ../../VideoAnalysis/rawvideos/conventional_videos \
      --frames_root ../../VideoAnalysis/data/frames/frames_conventional \
      --transcripts_root ../data/transcripts_conventional \
      --topics_root ../data/topics_conventional \
      --correlation_root ../results/correlation_conventional \
      --integrated_captions_root ../../CaptionAnalysis/data/integrated_caption/frames_conventional_captions_integrated \
      --out_dir ../results/metrics \
      --label Conventional

    # Embodied
    python embodiment_project_collect_metrics.py \
      --videos_root ../../VideoAnalysis/rawvideos/embodied_videos \
      --frames_root ../../VideoAnalysis/data/frames/frames_embodied \
      --transcripts_root ../data/transcripts_embodied \
      --topics_root ../data/topics_embodied \
      --correlation_root ../results/correlation_embodied \
      --integrated_captions_root ../../CaptionAnalysis/data/integrated_caption/frames_embodied_captions_integrated \
      --out_dir ../results/metrics \
      --label Embodied

    # 2. Export summary CSV for LaTeX
    python export_plot_data.py \
      --conventional ../results/metrics/Conventional_metrics.json \

### To Collect YouTube Metrics (Pipelined)

    cd TranscriptAnalysis/scripts
    conda activate videoanalysis

    # 1. Collect Detailed Metrics (JSON)
    # Conventional
    python collect_youtube_metrics_detailed.py \
      --comments_root ../../VideoAnalysis/data/comments/conventional \
      --correlation_root ../results_youtube/correlation_conventional \
      --metrics_root ../results/metrics \
      --out_dir ../results_youtube/metrics \
      --label Conventional

    # Embodied
    python collect_youtube_metrics_detailed.py \
      --comments_root ../../VideoAnalysis/data/comments/embodied \
      --correlation_root ../results_youtube/correlation_embodied \
      --metrics_root ../results/metrics \
      --out_dir ../results_youtube/metrics \
      --label Embodied

    # 2. Export Summary CSV
    python export_youtube_plot_data.py \
      --conventional ../results_youtube/metrics/Conventional_youtube_metrics.json \
      --embodied ../results_youtube/metrics/Embodied_youtube_metrics.json \
      --output ../results_youtube/metrics/transcript_youtube_plot_data.csv

