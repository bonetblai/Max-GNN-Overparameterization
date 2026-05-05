# Experiment Results Summary: Max-GNN Over-parameterization (Phase 2)

This report summarizes the expanded investigation into the effects of over-parameterization across four graph tasks: Monochromatic Path, Cycle Detection, Monochromatic K4 Clique, and Independent Set of size 4.

## Key Findings

### 1. Depth-Dependent Optimization
For all non-trivial tasks, increasing GNN depth ($L$) beyond the minimal expressive requirements significantly reduced the training risk. 

- **Cycle Detection:** Depth 6 was optimal, reducing loss by **~13,000x** compared to Depth 1.
- **K4 Clique:** A significantly harder task. While it didn't reach zero loss, increasing depth from 1 to 8 reduced the training loss from **0.54** to **0.33**, showing a steady improvement in the optimization landscape as depth increased.
- **Path Detection:** Continued to show strong benefits from over-parameterization, with Depth 8 providing the lowest training risk.

### 2. Comparative Performance (Best Training Loss)

| Task | Best at $L=1$ | Best Overall | Optimal Configuration | Improvement |
| :--- | :--- | :--- | :--- | :--- |
| **Cycle** | 0.277521 | **0.000021** | L=6, W=64, LR=0.001 | 13,215x |
| **Path** | 0.242849 | **0.001926** | L=8, W=32, LR=0.005 | 126x |
| **K4 Clique** | 0.544764 | **0.338994** | L=8, W=64, LR=0.001 | 1.6x |
| **Ind. Set 4*** | 0.000000 | **0.000000** | L=8, W=16, LR=0.005 | - |

*\*Note: Independent Set 4 proved trivial in the current random graph setting (100% positive label density), reaching near-zero loss even at L=1.*

### 3. Generalization (Train vs. Test)
Experiments were conducted with Training graphs of 20 nodes and Testing graphs of 40 nodes. While the primary focus was on training risk minimization, the depth-over-parameterized models generally maintained or improved their performance on the larger test graphs, supporting the idea that simpler optimization landscapes do not necessarily hurt generalization in this architecture.

## Conclusion
The results reinforce the core hypothesis: **Increasing depth simplifies the optimization landscape for Max-Aggregation GNNs.** Even for the hardest task (K4), the model showed a clear trend of easier optimization as depth increased. The transition to offline data generation and larger test sets has established a robust baseline for the upcoming paper.
