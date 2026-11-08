#!/usr/bin/env bash
# Runs every pre-registered experiment in DESIGN.md; finished runs are skipped.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
uv run python -m dd.split
for cat in pcb1 pcb2 pcb3 pcb4; do
  for s in 0 1 2; do uv run python -m dd.patchcore "$cat" "$s"; done
done
for k in 5 10 25 50; do
  for cat in pcb1 pcb2 pcb3 pcb4; do
    for s in 0 1 2; do uv run python -m dd.yolo "$cat" "$k" "$s"; done
  done
done
uv run python -m dd.evaluate
