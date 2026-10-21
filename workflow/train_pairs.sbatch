#!/bin/bash
# Train cycle pairs as a Slurm array: one row of a make_pairs.py CSV per task.
#
#   python make_pairs.py --source yin2017 -o pairs.csv
#   sbatch --array=1-$(($(wc -l < pairs.csv) - 1))%50 workflow/train_pairs.sbatch pairs.csv results
#
# Adjust the partition/GPU lines for your cluster.
# A 7-mer OLS fit on 450,000 reads needs ~30 GB of GPU memory; on smaller GPUs
# add "--ols_device cpu" (about 32 GB of RAM instead).
#SBATCH --job-name=qbic_train
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=logs/train_%A_%a.out
set -euo pipefail
PAIRS=${1:?usage: sbatch --array=1-N workflow/train_pairs.sbatch pairs.csv [out_dir]}
OUT=${2:-results}
cd "$SLURM_SUBMIT_DIR"
python qbic_selex_train.py --batch "$PAIRS" --row "$SLURM_ARRAY_TASK_ID" --out_dir "$OUT" --skip_existing
