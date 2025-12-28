import argparse
import json
import os
import sys
from itertools import islice
from youtube_comment_downloader import YoutubeCommentDownloader, SORT_BY_POPULAR

def download_comments(url, output_dir):
    downloader = YoutubeCommentDownloader()
    try:
        # Extract Video ID (simple split, can be improved with regex if needed)
        if "v=" in url:
            video_id = url.split("v=")[1].split("&")[0]
        else:
            # Handle short URLs or other formats if necessary, but assuming standard watch URLs for now
            # or try to use the last part of the url
            video_id = url.split("/")[-1]
            if "?" in video_id:
                 video_id = video_id.split("?")[0]
    
        print(f"Downloading comments for Video ID: {video_id}")
        
        json_output_path = os.path.join(output_dir, f"{video_id}.json")
        txt_output_path = os.path.join(output_dir, f"{video_id}.txt")

        # Generator for comments
        comments = downloader.get_comments_from_url(url, sort_by=SORT_BY_POPULAR)
        
        count = 0
        with open(json_output_path, 'w', encoding='utf-8') as json_file, \
             open(txt_output_path, 'w', encoding='utf-8') as txt_file:
            
            for comment in comments:
                # Save to JSONL
                json.dump(comment, json_file, ensure_ascii=False)
                json_file.write('\n')
                
                # Save to Text
                text = comment.get('text', '')
                if text:
                    txt_file.write(text + '\n\n')
                
                count += 1
                if count % 100 == 0:
                    print(f"Downloaded {count} comments...", end='\r')
        
        print(f"Finished! Total comments: {count}. Saved to {json_output_path} and {txt_output_path}")

    except Exception as e:
        print(f"Error downloading comments for {url}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Download YouTube comments from a list of URLs.")
    parser.add_argument("--input_file", default="../output/video_downloading/videos_sa_conventional.txt", help="Path to text file containing YouTube URLs (one per line)")
    parser.add_argument("--output_dir", default="../data/comments/conventional", help="Directory to save output files")

    args = parser.parse_args()

    if not os.path.exists(args.input_file):
        print(f"Error: Input file not found: {args.input_file}")
        sys.exit(1)

    if not os.path.exists(args.output_dir):
        try:
            os.makedirs(args.output_dir)
            print(f"Created output directory: {args.output_dir}")
        except OSError as e:
            print(f"Error creating output directory: {e}")
            sys.exit(1)

    with open(args.input_file, 'r') as f:
        urls = [line.strip() for line in f if line.strip()]

    print(f"Found {len(urls)} URLs to process.")

    for url in urls:
        download_comments(url, args.output_dir)

if __name__ == "__main__":
    main()
