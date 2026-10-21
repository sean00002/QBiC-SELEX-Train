#!/bin/bash
# Within-experiment validation as a Slurm array: one experiment per task.
#
#   python -c "import pandas as pd; pd.read_csv('pairs.csv').exp_id.drop_duplicates().to_csv('exps.txt', index=False, header=False)"
#   sbatch --array=1-$(wc -l < exps.txt)%50 workflow/validate.sbatch exps.txt results
#   python -m selection.select_models within --results results
#   python -m selection.cross_experiment --results results      # (or split by --model)
#   python -m selection.select_models cross --results results
#SBATCH --job-name=qbic_validate
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/validate_%A_%a.out
set -euo pipefail
EXPS=${1:?usage: sbatch --array=1-N workflow/validate.sbatch exps.txt [results]}
OUT=${2:-results}
cd "$SLURM_SUBMIT_DIR"
EXP=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$EXPS")
python -m selection.within_experiment --results "$OUT" --exp_id "$EXP"
