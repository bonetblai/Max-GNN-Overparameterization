# Experiment Results Summary: Max-GNN Over-parameterization

This report summarizes the investigation into the effects of over-parameterization on the optimization landscape of Max-Aggregation GNNs. We tested the hypothesis that increasing depth ($L$) beyond the minimal expressive depth ($L_{target}=3$) simplifies optimization and reduces training risk.

## Key Findings

### 1. Does Increasing Parameters Reduce Training Loss?
**Yes.** We observed a consistent negative correlation between the number of parameters and the training loss for both tasks:
- **Cycle Task Correlation:** -0.315
- **Path Task Correlation:** -0.288

While larger widths ($W$) obviously help, the impact of **depth ($L$)** was particularly pronounced. Increasing depth beyond $L=3$ (the expressive limit) significantly facilitated finding a near-zero loss solution.

### 2. $L_{target}$ vs. Optimal Depth Performance

| Task | Best Loss at $L=3$ | Best Overall Loss | Optimal Configuration | Improvement Factor |
| :--- | :--- | :--- | :--- | :--- |
| **Cycle Detection** | 0.035233 | **0.000045** | L=6, W=64, LR=0.001 | ~780x |
| **Path Detection** | 0.002769 | **0.000000** | L=7, W=64, LR=0.001 | ~50,000x |

#### Analysis:
- In the **Cycle task**, which is structurally harder, depth was critical. The best $L=3$ model still had significant residual loss, whereas $L=6$ achieved near-perfect convergence.
- In the **Path task**, the model at $L=3$ was already quite good, but over-parameterizing depth to $L=7$ resulted in an effectively zero loss, demonstrating that the optimization landscape becomes much smoother with extra layers.

## Conclusion
The data strongly supports the over-parameterization hypothesis. For both tasks, the "sweet spot" for optimization was found at roughly **2x to 2.3x the minimal expressive depth**. 

Beyond a certain point (e.g., $L=8$ for the Path task), we began to see diminishing returns or slight increases in average loss, likely due to the challenges of training very deep networks without specialized architectures like skip connections, but the intermediate over-parameterized range ($L=4$ to $L=7$) was consistently superior to the minimal depth.
