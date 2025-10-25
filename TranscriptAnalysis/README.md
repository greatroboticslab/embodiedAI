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
