import cv2
import os
import re
import argparse
import subprocess
from tqdm import tqdm



def create_folder(folder_name):
    safe_name = re.sub(r'[\\/*?:"<>|]', '', folder_name)
    folder_path = os.path.join("../data/frames/frames/", safe_name, "raw_frames")
    os.makedirs(folder_path, exist_ok=True)
    return folder_path

def is_av1_encoded(video_path):
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=codec_name", "-of", "default=noprint_wrappers=1:nokey=1",
             video_path],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        return result.stdout.strip() == "av1"
    except Exception as e:
        print(f"Could not check codec for {video_path}: {e}")
        return False

def convert_to_h264(input_path, output_path):
    print(f"Converting AV1 to H.264: {os.path.basename(input_path)}")
    subprocess.run([
        "ffmpeg", "-y", "-i", input_path,
        "-c:v", "libx264", "-crf", "23", "-preset", "fast",
        "-c:a", "copy", output_path
    ])

def video_to_frames(video_path, output_dir, video_name, skip_frames=100):
    safe_video_name = re.sub(r'[\\/*?:"<>|]', '', video_name)
    full_path = os.path.join(video_path, safe_video_name)

    # Convert AV1 if needed
    if is_av1_encoded(full_path):
        converted_path = os.path.join("/tmp", f"converted_{safe_video_name}")
        convert_to_h264(full_path, converted_path)
        full_path = converted_path

    video_capture = cv2.VideoCapture(full_path)
    if not video_capture.isOpened():
        print(f"Could not open video file: {full_path}")
        return []

    frame_paths = []
    frame_idx = 0
    saved_idx = 0

    while True:
        success, frame = video_capture.read()
        if not success:
            break

        if frame_idx % skip_frames == 0:
            image_path = os.path.join(output_dir, f"frame_{saved_idx:04d}.jpg")
            cv2.imwrite(image_path, frame)
            frame_paths.append(image_path)
            saved_idx += 1

        frame_idx += 1

    video_capture.release()
    return frame_paths

def main():
    ap = argparse.ArgumentParser(description="Extract frames from videos")
    ap.add_argument('--start', type=int, default=0, help='Start index of video list')
    ap.add_argument('--end', type=int, default=-1, help='End index of video list (-1 = all)')
    ap.add_argument("--video_dir", default="../rawvideos/conventional_videos", help="Path to video root")
    args = ap.parse_args()

    video_dir = args.video_dir
    videos = [f for f in os.listdir(video_dir) if f.lower().endswith('.mp4')]
    videos = videos[args.start:args.end] if args.end != -1 else videos[args.start:]

    if not videos:
        print("No videos found to process.")
        return

    for video in tqdm(videos, desc="Extracting videos"):
        video_name = video[:-4]
        frames_dir = create_folder(video_name)
        print(f"Extracting: {video} → {frames_dir}")
        frame_paths = video_to_frames(video_dir, frames_dir, video, skip_frames=100)
        print(f"Extracted {len(frame_paths)} frames.")

    print("All done.")

if __name__ == "__main__":
    main()
