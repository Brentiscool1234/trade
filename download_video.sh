#!/bin/bash

URL="${1:-}"

if [ -z "$URL" ]; then
  echo "Usage: $0 <url> [output_filename]"
  echo "Example: $0 'https://example.com/video.m3u8' video.mp4"
  exit 1
fi

OUTPUT="${2:-video.mp4}"

yt-dlp \
  --no-playlist \
  --concurrent-fragments 16 \
  -f "bv*+ba/b" \
  --merge-output-format mp4 \
  --postprocessor-args "ffmpeg:-movflags +faststart" \
  -o "$OUTPUT" \
  "$URL"
