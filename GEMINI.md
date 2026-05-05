# Max-GNN Over-parameterization Research

This project investigates the hypothesis that increasing GNN depth ($L$) beyond the minimal expressive depth ($L_{target}$) simplifies the optimization landscape and reduces training risk (loss), even if expressivity is theoretically sufficient.

## Architectural Mandates

- **Max Aggregation:** Always use max-aggregation (`aggr='max'`).
- **Message Transformation:** Neighbor embeddings MUST be transformed by a linear layer *before* aggregation to maximize expressivity.
- **Update Logic:** Concatenate the node's current features with the aggregated message, followed by a linear layer and ReLU.
- **Learning Rate:** Use parameter-aware LR scaling: $LR \propto 1/\sqrt{N_{params}}$.

## Experiment Workflow

- **Tasks:**
    - `path`: Monochromatic path detection.
    - `cycle`: Cycle detection.
    - `k4`: Monochromatic K4 (clique) detection.
    - `indset4`: Independent set of size 4 detection.
- **Data Workflow:**
    - Use `prepare_gen.py` and `submit_gen.sh` to generate data shards on Slurm.
    - Use `collect_data.py` to aggregate shards into `data/`.
    - `experiment.py` loads pre-generated `.pt` files for reproducibility.
    - Training: 20 nodes; Testing: 40 nodes.
- **Metrics:** Track loss at milestones (Epoch 5, 20, 50, 100) to observe convergence speed.
