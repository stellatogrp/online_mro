#!/bin/bash
# Sync code up to della and results back down.
#   bash slurm/sync_results.sh up     # push code (excludes venv/results)
#   bash slurm/sync_results.sh down   # pull results
set -euo pipefail

HOST=della-stellato
REMOTE=/scratch/gpfs/BSTELLATO/bs37/online_mro_paper
LOCAL="$(cd "$(dirname "$0")/.." && pwd)"

case "${1:-}" in
  up)
    ssh "$HOST" "mkdir -p $REMOTE"
    rsync -av --delete \
      --exclude '.venv' --exclude '__pycache__' --exclude 'results' \
      --exclude 'results_epsx' --exclude '*.so' --exclude '*.dylib' \
      --exclude 'slurm/logs' --exclude '.pytest_cache' \
      "$LOCAL/" "$HOST:$REMOTE/"
    ;;
  down)
    mkdir -p "$LOCAL/portfolio/results" "$LOCAL/svm/results"
    rsync -av "$HOST:$REMOTE/portfolio/results/" "$LOCAL/portfolio/results/"
    rsync -av "$HOST:$REMOTE/svm/results/" "$LOCAL/svm/results/"
    ;;
  *)
    echo "usage: $0 {up|down}" >&2; exit 1 ;;
esac
