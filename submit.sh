#!/bin/bash
#SBATCH --job-name=max-gnn-overparam
#SBATCH --partition=rleap_cpu
#SBATCH --array=0-749%50
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/job_%A_%a.out
#SBATCH --error=logs/job_%A_%a.err

# Load conda environment
# Use the full path to conda run if needed, or source the shell script
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate gemini

# Get the config for this task
CONFIG_LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" configs.jsonl)

# Extract parameters using python (simplest way to parse JSON in bash)
TASK=$(python3 -c "import json; print(json.loads('$CONFIG_LINE')['task'])")
L=$(python3 -c "import json; print(json.loads('$CONFIG_LINE')['depth'])")
W=$(python3 -c "import json; print(json.loads('$CONFIG_LINE')['width'])")
LR=$(python3 -c "import json; print(json.loads('$CONFIG_LINE')['lr'])")
L_TARGET=$(python3 -c "import json; print(json.loads('$CONFIG_LINE')['l_target'])")
EPOCHS=$(python3 -c "import json; print(json.loads('$CONFIG_LINE')['epochs'])")
OUTPUT=$(python3 -c "import json; print(json.loads('$CONFIG_LINE')['output'])")

echo "Running task index $SLURM_ARRAY_TASK_ID: Task=$TASK, L=$L, W=$W, LR=$LR"

python experiment.py \
    --task $TASK \
    --depth $L \
    --width $W \
    --lr $LR \
    --l_target $L_TARGET \
    --epochs $EPOCHS \
    --output $OUTPUT
