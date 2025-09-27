#!/bin/bash
#SBATCH --job-name=download
#SBATCH --array=0-300%30
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --output=logs/download_%A_%a.out
#SBATCH --error=logs/download_%A_%a.err

# Define variables
INPUT_FILE="download_list.txt"  # Your input file
NUM_ROWS=$(wc -l < "$INPUT_FILE")  # Total number of rows
TASK_COUNT=300  # Number of tasks
CHUNK_SIZE=$((NUM_ROWS / TASK_COUNT + (NUM_ROWS % TASK_COUNT > 0)))  # Rows per task
START_ROW=$((SLURM_ARRAY_TASK_ID * CHUNK_SIZE + 1))  # Start line for the task
END_ROW=$((START_ROW + CHUNK_SIZE - 1))  # End line for the task

# Ensure the logs directory exists
mkdir -p logs

# Process the relevant lines for the current task
sed -n "${START_ROW},${END_ROW}p" "$INPUT_FILE" | while read -r variable; do
    # Skip if the line is empty
    if [ -z "$variable" ]; then
        continue
    fi
    python3 download.py -i "$variable"
done

