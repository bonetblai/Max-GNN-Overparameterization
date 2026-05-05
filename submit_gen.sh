#!/bin/bash
#SBATCH --job-name=gen-data
#SBATCH --partition=rleap_cpu
#SBATCH --array=0-79
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=logs/gen_%A_%a.out
#SBATCH --error=logs/gen_%A_%a.err

CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate gemini

TASKS=(cycle path k4 indset4)
NUM_TASKS=${#TASKS[@]}
SHARDS=10

# Calculate task, type (train/test), and shard from SLURM_ARRAY_TASK_ID
TYPE_INDEX=$((SLURM_ARRAY_TASK_ID / (NUM_TASKS * SHARDS)))
REMAINDER=$((SLURM_ARRAY_TASK_ID % (NUM_TASKS * SHARDS)))
TASK_INDEX=$((REMAINDER / SHARDS))
SHARD_INDEX=$((REMAINDER % SHARDS))

TASK=${TASKS[$TASK_INDEX]}

if [ $TYPE_INDEX -eq 0 ]; then
    TYPE="train"
    NODES=20
else
    TYPE="test"
    NODES=40
fi

OUTPUT="data/shards/${TASK}_${TYPE}_shard${SHARD_INDEX}.pt"

python generate_data.py --task $TASK --num_graphs 100 --num_nodes $NODES --output $OUTPUT
