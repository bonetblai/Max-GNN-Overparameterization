# Research Memory: GNN Depth Hypothesis

## Current Findings
- For `path` and `cycle` ($L_{target}=3$), over-parameterization in depth ($L=4$ to $L=7$) significantly simplifies optimization and reduces training loss compared to the minimal expressive depth $L=3$.
- Best losses achieved: Cycle (~0.000045 at L=6), Path (~0.000000 at L=7).
- Results summarized in `RESULTS_SUMMARY.md`.

## New Tasks & Complex Optimization
- Added `k4` (Monochromatic K4 detection) and `indset4` (Independent set of size 4).
- Testing on larger graphs (40 nodes) than training (20 nodes) to evaluate generalization while focusing on training risk.
- Implemented offline data generation to handle $O(n^4)$ labeling complexity and ensure reproducibility.

## Environment & Workflow
- Conda environment: `gemini` (experiments), `ai-tools` (analysis).
- Slurm partition: `rleap_cpu`.
- Data generation parallelized: `prepare_gen.py` -> `sbatch submit_gen.sh` -> `collect_data.py`.
- Experiment sweep: `launch_experiments.py` (384 runs).

## Key Files
- `model.py`: MaxGNN implementation.
- `dataset.py`: Graph generation and labeling logic.
- `generate_data.py`: CLI for task/shard generation.
- `collect_data.py`: Aggregation of data shards.
- `experiment.py`: Training/Eval logic (loads data from `data/`).
