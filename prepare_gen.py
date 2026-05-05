import os
import subprocess

def create_gen_submit_script(tasks, shards, num_graphs_per_shard, train_nodes, test_nodes):
    script_content = f"""#!/bin/bash
#SBATCH --job-name=gen-data
#SBATCH --partition=rleap_cpu
#SBATCH --array=0-{len(tasks) * shards * 2 - 1}
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=logs/gen_%A_%a.out
#SBATCH --error=logs/gen_%A_%a.err

CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate gemini

TASKS=({" ".join(tasks)})
NUM_TASKS=${{#TASKS[@]}}
SHARDS={shards}

# Calculate task, type (train/test), and shard from SLURM_ARRAY_TASK_ID
TYPE_INDEX=$((SLURM_ARRAY_TASK_ID / (NUM_TASKS * SHARDS)))
REMAINDER=$((SLURM_ARRAY_TASK_ID % (NUM_TASKS * SHARDS)))
TASK_INDEX=$((REMAINDER / SHARDS))
SHARD_INDEX=$((REMAINDER % SHARDS))

TASK=${{TASKS[$TASK_INDEX]}}

if [ $TYPE_INDEX -eq 0 ]; then
    TYPE="train"
    NODES={train_nodes}
else
    TYPE="test"
    NODES={test_nodes}
fi

OUTPUT="data/shards/${{TASK}}_${{TYPE}}_shard${{SHARD_INDEX}}.pt"

python generate_data.py --task $TASK --num_graphs {num_graphs_per_shard} --num_nodes $NODES --output $OUTPUT
"""
    with open('submit_gen.sh', 'w') as f:
        f.write(script_content)

if __name__ == "__main__":
    tasks = ['cycle', 'path', 'k4', 'indset4']
    shards = 10
    num_graphs_per_shard = 100 # Total 1000 per task
    train_nodes = 20
    test_nodes = 40
    
    os.makedirs('logs', exist_ok=True)
    create_gen_submit_script(tasks, shards, num_graphs_per_shard, train_nodes, test_nodes)
    print("Created submit_gen.sh. Run 'sbatch submit_gen.sh' to generate shards.")
