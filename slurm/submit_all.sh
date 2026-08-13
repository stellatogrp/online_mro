#!/bin/bash
# Submit the full paper-experiment suite. Run on a della login node from the
# project dir. Optionally pass a filter, e.g.:
#   bash slurm/submit_all.sh              # everything
#   bash slurm/submit_all.sh svm          # only svm jobs
#   EXTRA="--r_start 10 --R 10" bash slurm/submit_all.sh port_mro_subgrad  # seed extension
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs

FILTER="${1:-}"

# Sizing calibrated 2026-08-08 from profiling/results (Mac sweep at
# T in {51,101,201,401}, quadratic fit to T=2001) x measured della/Mac
# single-core ratio, x1.3-1.5 margin; memory from measured peak worker RSS.
# Heavy portfolio subgrad paths are split into two seed-halves (30 tasks
# each = 5 seeds x 6 eps <= 32 cpus, single joblib batch per job).
#
# name | experiment | method | solver | cpus | mem/cpu | time | extra args
JOBS=$(cat <<'EOF'
port_mro_subgrad_s0 portfolio mro      subgrad 32 1G  03:00:00 --r_start,0,--R,5
port_mro_subgrad_s5 portfolio mro      subgrad 32 1G  03:00:00 --r_start,5,--R,5
port_dro_subgrad_s0 portfolio dro      subgrad 32 1G  03:00:00 --r_start,0,--R,5
port_dro_subgrad_s5 portfolio dro      subgrad 32 1G  03:00:00 --r_start,5,--R,5
port_mro_exact      portfolio mro      exact   32 3G  02:00:00 -
port_dro_exact_s0   portfolio dro      exact   14 12G 08:00:00 --r_start,0,--R,5
port_dro_exact_s5   portfolio dro      exact   14 12G 08:00:00 --r_start,5,--R,5
port_true_saa       portfolio true_saa exact    4 10G 12:00:00 -
svm_mro_subgrad     svm       mro      subgrad 32 1G  06:00:00 -
svm_mro_exact       svm       mro      exact   32 1G  02:00:00 -
svm_dro_subgrad     svm       dro      subgrad 32 1G  01:00:00 -
svm_dro_exact       svm       dro      exact   32 1G  02:00:00 -
svm_true_saa        svm       true_saa exact    4 8G  08:00:00 -
EOF
)

while read -r name experiment method solver cpus mem time extra; do
    [[ -z "$name" || "$name" == \#* ]] && continue
    if [[ -n "$FILTER" && "$name" != *"$FILTER"* ]]; then continue; fi
    args="${extra//,/ }"
    [[ "$args" == "-" ]] && args=""
    args="$args ${EXTRA:-}"
    sbatch --job-name="$name" \
        --cpus-per-task="$cpus" --mem-per-cpu="$mem" --time="$time" \
        --output="logs/${name}_%j.out" \
        --export="ALL,EXPERIMENT=$experiment,METHOD=$method,SOLVER=$solver,EXTRA_ARGS=$args" \
        job.slurm
done <<< "$JOBS"
