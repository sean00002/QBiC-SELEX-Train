#!/bin/bash
# Download the reads for a list of experiments as a Slurm array (one experiment per task).
#
#   python make_pairs.py --all --per_experiment -o experiments.csv
#   tail -n +2 experiments.csv | cut -d, -f1 > exps.txt
#   sbatch --array=1-$(wc -l < exps.txt)%30 workflow/download_reads.sbatch exps.txt
#SBATCH --job-name=qbic_download
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=12:00:00
#SBATCH --output=logs/download_%A_%a.out
set -euo pipefail
EXPS=${1:?usage: sbatch --array=1-N workflow/download_reads.sbatch exps.txt}
cd "$SLURM_SUBMIT_DIR"
python download_reads.py --exp_id "$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$EXPS")"
