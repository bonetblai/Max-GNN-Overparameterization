# Max-GNN Over-parameterization Research

This project investigates the hypothesis that increasing GNN depth ($L$) beyond the minimal expressive depth ($L_{target}$) simplifies the optimization landscape and reduces training risk (loss), even if expressivity is theoretically sufficient.

## Architectural Mandates

- **Max Aggregation:** Always use max-aggregation (`aggr='max'`).
- **Message Transformation:** Neighbor embeddings MUST be transformed by a linear layer *before* aggregation to maximize expressivity.
- **Update Logic:** Concatenate the node's current features with the aggregated message, followed by a linear layer and ReLU.
- **Learning Rate:** Use parameter-aware LR scaling: $LR \propto 1/\sqrt{N_{params}}$.

## Experiment Workflow

- **Tasks:**
    - `path`: Monochromatic path detection (requires $L \ge L_{target}$).
    - `cycle`: Cycle detection (structurally harder for GNNs).
- **Metrics:** Track loss at milestones (Epoch 5, 20, 50, 100) to observe convergence speed.
