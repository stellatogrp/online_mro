#!/bin/bash
# Seed-array variant of job.slurm: one array element per seed, each running
# the full epsilon sweep on its own cores (joblib width = cpus-per-task).
# Submit with e.g.:
#   sbatch --array=0-29 --cpus-per-task=7 --mem-per-cpu=6G --time=16:00:00 \
#     --export="ALL,EXPERIMENT=portfolio,METHOD=dro,SOLVER=exact,EXTRA_ARGS=--T 5001 --fixed_time 5001" \
#     job_array_seed.slurm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --account=bstellato
#SBATCH --mail-type=FAIL,TIME_LIMIT

source "${SLURM_SUBMIT_DIR}/common.sh"

if [ "$METHOD" = "true_saa" ]; then
    RESULTS_DIR="$PROJECT_DIR/$EXPERIMENT/results/true_saa"
else
    RESULTS_DIR="$PROJECT_DIR/$EXPERIMENT/results/${METHOD}_${SOLVER}"
fi

"$PY" -m "$EXPERIMENT.run" \
    --method "$METHOD" --solver "$SOLVER" \
    --results_dir "$RESULTS_DIR" \
    --r_start "$SLURM_ARRAY_TASK_ID" --R 1 \
    ${EXTRA_ARGS:-}
