#!/bin/bash
# Extract each planned window to 16 kHz mono and transcribe it with base.en.
set -u
cd "$(dirname "$0")"
python3 -c "import json;[print(w['id'],w['offset'],w['len'],w['book'],sep='\t') for w in json.load(open('plan.json'))]" |
while IFS=$'\t' read -r id off len book; do
  [ -s "tx/$id.jsonl" ] && continue
  ffmpeg -v error -y -ss "$off" -t "$len" -i "$book" -ac 1 -ar 16000 "wav/$id.wav" || { echo "ffmpeg failed $id"; continue; }
  ../../zig-out/bin/zwhisper transcribe ../../models/ggml-base.en.bin "wav/$id.wav" -t 8 --json --no-voices > "tx/$id.jsonl.part" 2>/dev/null \
    && mv "tx/$id.jsonl.part" "tx/$id.jsonl" || echo "transcribe failed $id"
  rm -f "wav/$id.wav"
  echo "$id done $(date +%T)"
done
echo ALL_DONE
