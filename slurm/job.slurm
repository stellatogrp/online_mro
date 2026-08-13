#!/bin/bash
# Generic job template. Submitted by submit_all.sh, which passes resources on
# the sbatch command line and the workload via --export:
#   EXPERIMENT   portfolio | svm
#   METHOD       mro | dro | true_saa
#   SOLVER       exact | subgrad
#   EXTRA_ARGS   optional extra CLI args (e.g. "--r_start 10 --R 10")
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --account=bstellato
#SBATCH --mail-type=END,FAIL,TIME_LIMIT

# Slurm copies this script to a spool dir; resolve common.sh from the submit dir.
source "${SLURM_SUBMIT_DIR}/common.sh"

if [ "$METHOD" = "true_saa" ]; then
    RESULTS_DIR="$PROJECT_DIR/$EXPERIMENT/results/true_saa"
else
    RESULTS_DIR="$PROJECT_DIR/$EXPERIMENT/results/${METHOD}_${SOLVER}"
fi

"$PY" -m "$EXPERIMENT.run" \
    --method "$METHOD" --solver "$SOLVER" \
    --results_dir "$RESULTS_DIR" \
    ${EXTRA_ARGS:-}
