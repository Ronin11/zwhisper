#!/bin/bash
# Train one model per span length and score each at chunk resolution.
# usage: train.sh <tag> [extra train flags]
set -u; cd "$(dirname "$0")"
tag=$1; shift
B=~/.cache/indecis/models/bekko-embedding-v1-a8m
mkdir -p models results
for L in 1 2 4 8; do
  len=$((L * 256))
  ~/Projects/indecis/bin/indecis train -backbone $B -schema schema.json -train data/L${L}_train.jsonl -test data/L${L}_test.jsonl \
      -max-len $len -epochs 3 -out models/${tag}_L$L "$@" > results/${tag}_L$L.train.log 2>&1
  echo "== L$L (max-len $len) $(grep -o 'training: .*' results/${tag}_L$L.train.log)"
  python3 score.py models/${tag}_L$L data/L${L}_test.jsonl $([ $L = 1 ] && echo --smooth) | tee results/${tag}_L$L.txt
done
