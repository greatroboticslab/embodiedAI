import os
import glob

base_dir = r"C:\Users\Benjamin Li\Documents\Antigravity\embodiedAI\VideoAnalysis\data\comments"

def check_files(category):
    dir_path = os.path.join(base_dir, category)
    files = glob.glob(os.path.join(dir_path, "*.json"))
    print(f"--- {category} ({len(files)} files) ---")
    
    empty_files = []
    small_files = []
    
    for f in files:
        size = os.path.getsize(f)
        if size == 0:
            empty_files.append(os.path.basename(f))
        elif size < 100: # Arbitrary small threshold to check for error msgs
            small_files.append((os.path.basename(f), size))
            
    print(f"Empty files ({len(empty_files)}): {empty_files}")
    print(f"Small files (<100b): {small_files}")

check_files('conventional')
check_files('embodied')
