#!/bin/bash

# CONFIGURATION
ENV_NAME="whisper"
RUN_ID=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="downloaded_videos_$RUN_ID"
mkdir -p "$OUTPUT_DIR"

FILES=(../output/video_downloading/videos.txt)
BATCH_SIZE=15
SLEEP_BETWEEN_BATCHES=300 # Seconds between batches
MAX_DOWNLOADS=100 # Set your max video downloads here

set -x
cd ../rawvideos

echo "YT-DLP LOG" > ../output/ytdlp_output.txt
TOTAL_DOWNLOADED=0
TOTAL_SUCCESSFUL=0
TOTAL_FAILED=0

for URL_LIST in "${FILES[@]}"; do
    BASENAME=$(basename "$URL_LIST" .txt)
    LOG_FILE="${BASENAME}_downloaded.log"
    FAILED_LOG="${BASENAME}_failed.log"
    RETRY_LIST="${BASENAME}_retry.txt"

    echo ""
    echo "Processing: $URL_LIST"
    echo "Log file: $LOG_FILE"

    touch "$LOG_FILE"
    touch "$FAILED_LOG"

    # Exclude both successfully downloaded and previously failed URLs
    grep -vxFf "$LOG_FILE" "$URL_LIST" | grep -vxFf "$FAILED_LOG" > temp_remaining_"$BASENAME".txt

    echo "Remaining URLs to download:"
    cat temp_remaining_"$BASENAME".txt

    # First pass: Standard download method
    while [ -s temp_remaining_"$BASENAME".txt ]; do
        if [ "$TOTAL_DOWNLOADED" -ge "$MAX_DOWNLOADS" ]; then
            echo "🚫 Reached max download limit of $MAX_DOWNLOADS videos. Stopping."
            break 2
        fi

        echo "Starting new batch for $URL_LIST..."
        REMAINING=$((MAX_DOWNLOADS - TOTAL_DOWNLOADED))
        CURRENT_BATCH_SIZE=$((REMAINING < BATCH_SIZE ? REMAINING : BATCH_SIZE))

        head -n "$CURRENT_BATCH_SIZE" temp_remaining_"$BASENAME".txt > current_batch_"$BASENAME".txt

        # Standard download with best quality
        echo "📥 Attempting standard download method..."
        conda run -n "$ENV_NAME" yt-dlp \
            --format "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best" \
            --merge-output-format mp4 \
            --cookies "../cookies.txt" \
            --retries 3 \
            --fragment-retries 3 \
            --retry-sleep 5 \
            --no-continue \
            --ignore-errors \
            -o "${OUTPUT_DIR}/%(id)s.%(ext)s" \
            -a current_batch_"$BASENAME".txt \
            >> ../output/ytdlp_output.txt 2>&1

        # Check which videos were actually downloaded
        BATCH_SUCCESSFUL=0
        BATCH_FAILED=0

        > "${RETRY_LIST}.tmp"  # Clear temp retry list

        while IFS= read -r url; do
            # Extract video ID from URL
            video_id=$(echo "$url" | grep -oP '(?<=v=)[^&]+' || echo "$url" | grep -oP '(?<=be/)[^?]+')

            # Check if video file exists
            if ls "${OUTPUT_DIR}/${video_id}".* 1> /dev/null 2>&1; then
                echo "$url" >> "$LOG_FILE"
                ((BATCH_SUCCESSFUL++))
            else
                echo "$url" >> "${RETRY_LIST}.tmp"
                ((BATCH_FAILED++))
                echo "⚠️  Failed with standard method: $url (ID: $video_id)"
            fi
        done < current_batch_"$BASENAME".txt

        TOTAL_DOWNLOADED=$((TOTAL_DOWNLOADED + CURRENT_BATCH_SIZE))
        TOTAL_SUCCESSFUL=$((TOTAL_SUCCESSFUL + BATCH_SUCCESSFUL))

        sed -i "1,${CURRENT_BATCH_SIZE}d" temp_remaining_"$BASENAME".txt

        echo "Batch complete: $BATCH_SUCCESSFUL successful, $BATCH_FAILED failed ($TOTAL_SUCCESSFUL / $TOTAL_DOWNLOADED total attempted)"

        # Second pass: Retry failed videos with Android client
        if [ -s "${RETRY_LIST}.tmp" ]; then
            echo ""
            echo "🔄 Retrying $BATCH_FAILED failed videos with Android client method..."

            conda run -n "$ENV_NAME" yt-dlp \
                --format "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best" \
                --merge-output-format mp4 \
                --extractor-args "youtube:player_client=android" \
                --force-ipv4 \
                -N 4 \
                --cookies "../cookies.txt" \
                --retries 3 \
                --fragment-retries 3 \
                --retry-sleep 5 \
                --no-continue \
                --ignore-errors \
                -o "${OUTPUT_DIR}/%(id)s.%(ext)s" \
                -a "${RETRY_LIST}.tmp" \
                >> ../output/ytdlp_output.txt 2>&1

            # Check retry results
            RETRY_SUCCESSFUL=0
            while IFS= read -r url; do
                video_id=$(echo "$url" | grep -oP '(?<=v=)[^&]+' || echo "$url" | grep -oP '(?<=be/)[^?]+')

                if ls "${OUTPUT_DIR}/${video_id}".* 1> /dev/null 2>&1; then
                    echo "$url" >> "$LOG_FILE"
                    ((RETRY_SUCCESSFUL++))
                    ((TOTAL_SUCCESSFUL++))
                    echo "✅ Retry successful: $url"
                else
                    echo "$url" >> "$FAILED_LOG"
                    ((TOTAL_FAILED++))
                    echo "❌ Retry failed: $url"
                fi
            done < "${RETRY_LIST}.tmp"

            echo "Retry complete: $RETRY_SUCCESSFUL recovered, $((BATCH_FAILED - RETRY_SUCCESSFUL)) still failed"
            rm -f "${RETRY_LIST}.tmp"
        else
            TOTAL_FAILED=$((TOTAL_FAILED + BATCH_FAILED))
        fi

        echo ""
        sleep "$SLEEP_BETWEEN_BATCHES"
    done

    echo "✅ Finished processing $URL_LIST or hit limit."
    rm -f current_batch_"$BASENAME".txt temp_remaining_"$BASENAME".txt
done

echo ""
echo "🎉 Script complete!"
echo "   Total videos attempted: $TOTAL_DOWNLOADED"
echo "   Successfully downloaded: $TOTAL_SUCCESSFUL"
echo "   Failed downloads: $TOTAL_FAILED"
echo ""
if [ "$TOTAL_FAILED" -gt 0 ]; then
    echo "⚠️  Check ${BASENAME}_failed.log for URLs that failed to download"
fi
echo ""
echo "Beginning transcription..."

cd ..
#conda run -n whisper python identify_videos.py
#conda run -n whisper python masswhisper.py