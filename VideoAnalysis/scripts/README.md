# Video Analysis Scripts

This directory contains utility scripts for video analysis tasks.

## `download_comments.py`

This script downloads comments from a list of YouTube video URLs.

### Usage

**Default Usage:**
Run the script without arguments to use the default input and output paths:
```bash
python download_comments.py
```

**Custom Usage:**
Specify custom input file and output directory:
```bash
python download_comments.py --input_file path/to/urls.txt --output_dir path/to/output
```

### Arguments

- `--input_file`: Path to a text file containing YouTube video URLs (one per line).
  - **Default**: `../output/video_downloading/videos_sa_embodied.txt`
- `--output_dir`: Directory where the downloaded comments will be saved.
  - **Default**: `../data/comments/embodied`

### Output

For each video, the script generates two files in the output directory:
1. `{video_id}.json`: Contains the full comment data in JSONL format.
2. `{video_id}.txt`: Contains just the text of the comments.
