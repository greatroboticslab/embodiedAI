# Automated Video Transcription, Topic Extraction, and Comment Correlation
This project automates video analysis by downloading, transcribing, extracting topics, and correlating them with viewer comments.

## 🚀 Quick Setup
### Whisper
You should create a conda environment called whisper. This is optional, but the SLURM batch jobs provided use this environment.

	conda create -n whisper python=3.12
	conda activate whisper

You will need yt-dlp:

	python3 -m pip install -U "yt-dlp[default]"

You will also need whisper, which will convert audio to transcripts:

	pip install -U openai-whisper

Finally, you will need ffmpeg:

	conda install conda-forge::ffmpeg

There is a text file called videos.txt in the video_processing folder. Paste YouTube links separated by a newline into this text folder for videos you wish to download.

### For Frame Extraction
### Depth Anything

You can also generate depth frames (pictures of 3d depth) from the relevant videos. Set up the environment:

	cd VideoAnalysis/frame_extraction/
	conda create -n depthanything
	conda activate depthanything
	pip install -r requirements.txt

### For Correlation
### Video Analysis

    cd VideoAnalysis/
    conda create -n videoanalysis
    conda activate videoanalysis
    pip install -r requirements.txt


## Usage 🏃‍♂️
### To Download Videos 
Paste YouTube links separated by a newline into VideoAnalysis/output/video_downloading/videos.txt for videos you wish to download and run
    
    cd VideoAnalysis/local_server_script
    conda activate whisper
    bash batchvideos.sh

Videos will be downloaded in VideoAnalysis/rawvideoss

### To Generate Transcripts

    cd VideoAnalysis/transcription
    conda activate whisper
    python transcribe_local_folder.py --src_dir path/to/videos/root

### To Extract Frames

    cd VideoAnalysis/frame_extraction
    conda activate depthanything
    python extract_frames.py --video_dir path/to/videos/root

### To Generate Topics from Transcripts

    cd VideoAnalysis/correlation
    conda activate videoanalysis
    python extract_topics.py \
    --src_dir path/to/transcripts \
    --outline_simple \
    --whole_video_outline \
    --force

Select --outline_simple to uses segments instead of timestamps in transcripts.
Select --whole_video_outline to pass entire transcript instead of chunked segments.
Select --force to recompute even if outputs exist.

### To Correlate Comments with Topics

    cd VideoAnalysis/correlation
    conda activate videoanalysis
    python topic_comment_correlation.py 


