# Research Memory: GNN Depth Hypothesis

## Current Findings
- For $L_{target}=5$, $W=32$, we observed that $L=6$ and $L=7$ reached lower final training losses (0.0001) than the minimal depth $L=5$ (0.0004).
- Convergence speed (Loss@100) was faster for $L=6$ than $L=5$.
- Evaluation accuracy tends to peak at $L \approx L_{target}$ and may slightly degrade with extreme over-parameterization ($L=8+$), suggesting a trade-off between risk minimization and generalization.

## Planned Experiments on Cluster
- Sweep $L$ from 1 to 20 to find the "point of diminishing returns" for risk minimization.
- Test if Width ($W$) compensates for lack of Depth ($L$) or if Depth has a unique effect on the loss landscape.
- Investigate "Cycle Detection" task with higher widths ($W=64, 128$) to see if it becomes learnable.
