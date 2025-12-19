import os
import shutil

def cleanup_duplicates():
    results_root = r"C:\Users\Benjamin Li\Documents\Antigravity\embodiedAI\TranscriptAnalysis\results"
    embodied_root = os.path.join(results_root, "correlation_embodied")
    
    if not os.path.exists(embodied_root):
        print(f"Error: {embodied_root} does not exist. Cannot verify safe cleanup.")
        return

    print(f"Scanning {results_root} for duplicates present in {embodied_root}...")
    
    deleted_count = 0
    
    # List directories in the encoded subfolder
    safe_dirs = set(os.listdir(embodied_root))
    
    # List directories in the root
    for item in os.listdir(results_root):
        item_path = os.path.join(results_root, item)
        
        # Skip if not a directory or if it is the target folder itself
        if not os.path.isdir(item_path) or item == "correlation_embodied":
            continue
            
        # If this directory exists in the correlation_embodied folder, it is a duplicate we likely created
        if item in safe_dirs:
            print(f"Deleting duplicate in root: {item}")
            try:
                shutil.rmtree(item_path)
                deleted_count += 1
            except Exception as e:
                print(f"Failed to delete {item}: {e}")
                
    print(f"\nCleanup complete. Deleted {deleted_count} duplicate folders.")

if __name__ == "__main__":
    cleanup_duplicates()
