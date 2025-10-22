import os

root_dir = "/data/ben/HumanoidRobotTrainingData/video_processing/frames"

for subdir, _, files in os.walk(root_dir):
    for file in files:
        if file.endswith("_raw_frames_integrated.docx"):
            old_path = os.path.join(subdir, file)

            # Insert "_llava" before ".docx"
            new_file = file.replace("_raw_frames_integrated.docx", "_raw_frames_captions_integrated.docx")
            new_path = os.path.join(subdir, new_file)

            os.rename(old_path, new_path)
            print(f"Renamed: {old_path} -> {new_path}")
