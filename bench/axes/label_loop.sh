#!/bin/bash
# Re-chunk and label as transcripts arrive, for each teacher in turn.
cd "$(dirname "$0")"
for model in qwen3:30b-a3b qwen3.6:35b-a3b; do
  while true; do
    python3 chunk.py >/dev/null
    python3 label.py "$model" 2>>label_$model.log
    grep -q ALL_DONE transcribe.log && { python3 chunk.py >/dev/null; python3 label.py "$model" 2>>label_$model.log; break; }
    sleep 60
  done
  echo "teacher $model done $(date +%T)"
done
echo LABELS_DONE
