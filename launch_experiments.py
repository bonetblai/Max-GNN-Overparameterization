import json
import os
import subprocess
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry_run', action='store_true', help='Do not submit jobs')
    parser.add_argument('--results_dir', type=str, default='results', help='Directory to save results')
    args = parser.parse_args()

    # Define parameter grid
    tasks = ['path', 'cycle', 'k4', 'indset4']
    depths = [1, 2, 3, 4, 5, 6, 7, 8]
    widths = [8, 16, 32, 64]
    lrs = [0.001, 0.005, 0.01]
    
    # Constants
    l_target = 3
    epochs = 200

    configs = []
    for task in tasks:
        for L in depths:
            for W in widths:
                for lr in lrs:
                    output_path = os.path.join(args.results_dir, f"{task}_L{L}_W{W}_lr{lr:.4f}.json")
                    configs.append({
                        'task': task,
                        'depth': L,
                        'width': W,
                        'lr': lr,
                        'l_target': l_target,
                        'epochs': epochs,
                        'output': output_path
                    })

    # Write configs to a jsonl file
    configs_file = 'configs.jsonl'
    with open(configs_file, 'w') as f:
        for config in configs:
            f.write(json.dumps(config) + '\n')

    print(f"Generated {len(configs)} configurations in {configs_file}")

    # Create the Slurm submission script
    submit_script = 'submit.sh'
    with open(submit_script, 'w') as f:
        f.write(f"""#!/bin/bash
#SBATCH --job-name=max-gnn-overparam
#SBATCH --partition=rleap_cpu
#SBATCH --array=0-{len(configs) - 1}%50
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
CONFIG_LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" {configs_file})

# Extract parameters using python (simplest way to parse JSON in bash)
TASK=$(python3 -c "import json; print(json.loads('$CONFIG_LINE')['task'])")
L=$(python3 -c "import json; print(json.loads('$CONFIG_LINE')['depth'])")
W=$(python3 -c "import json; print(json.loads('$CONFIG_LINE')['width'])")
LR=$(python3 -c "import json; print(json.loads('$CONFIG_LINE')['lr'])")
L_TARGET=$(python3 -c "import json; print(json.loads('$CONFIG_LINE')['l_target'])")
EPOCHS=$(python3 -c "import json; print(json.loads('$CONFIG_LINE')['epochs'])")
OUTPUT=$(python3 -c "import json; print(json.loads('$CONFIG_LINE')['output'])")

echo "Running task index $SLURM_ARRAY_TASK_ID: Task=$TASK, L=$L, W=$W, LR=$LR"

python experiment.py \\
    --task $TASK \\
    --depth $L \\
    --width $W \\
    --lr $LR \\
    --l_target $L_TARGET \\
    --epochs $EPOCHS \\
    --output $OUTPUT
""")

    print(f"Created submission script: {submit_script}")
    os.makedirs('logs', exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    if not args.dry_run:
        print("Submitting jobs to Slurm...")
        result = subprocess.run(['sbatch', submit_script], capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print(result.stderr)
    else:
        print("Dry run: To submit, run 'sbatch submit.sh'")

if __name__ == "__main__":
    main()
