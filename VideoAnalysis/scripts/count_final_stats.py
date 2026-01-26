import os
import glob
import statistics

base_dir = r"C:\Users\Benjamin Li\Documents\Antigravity\embodiedAI\VideoAnalysis\data\comments"

def get_stats(counts):
    if not counts:
        return 0, 0, 0, 0, 0
    return min(counts), max(counts), statistics.mean(counts), len(counts), sum(counts)

def count_valid_comments(category):
    counts = []
    dir_path = os.path.join(base_dir, category)
    files = glob.glob(os.path.join(dir_path, "*.json"))
    
    for f in files:
        filename = os.path.basename(f)
        video_id = os.path.splitext(filename)[0]
        
        # Check for valid YouTube ID length (11 chars)
        if len(video_id) != 11:
            print(f"Skipping invalid ID in {category}: {filename}")
            continue
            
        # Count comments, treat empty file as 0
        if os.path.getsize(f) == 0:
            counts.append(0)
            continue
            
        try:
            with open(f, 'r', encoding='utf-8') as file:
                line_count = sum(1 for line in file if line.strip())
                counts.append(line_count)
        except:
            print(f"Error reading {filename}")
            counts.append(0)
            
    return counts

results = {}
for cat in ['conventional', 'embodied']:
    results[cat] = count_valid_comments(cat)

print("\nFinal Comment Statistics (Filtered):\n")
for cat in ['conventional', 'embodied']:
    print(f"--- {cat.capitalize()} ---")
    c_min, c_max, c_avg, c_count, c_sum = get_stats(results[cat])
    print(f"Total Number of Comments:")
    print(f"  Minimum: {c_min}")
    print(f"  Maximum: {c_max}")
    print(f"  Average: {c_avg:.2f}")
    print(f"  Total Count (Videos): {c_count}")
    print(f"  Sum (Total Comments): {c_sum}")
    print("")
