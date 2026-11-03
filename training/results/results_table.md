| Model | Kendall τ | Spearman ρ | F1@15% |
|---|---:|---:|---:|
| Human (inter-annotator) | 0.177 | 0.204 | 0.493 |
| Random | 0.000 | 0.000 | 0.516 |
| Segment MLP (original design) | 0.165 ± 0.004 | 0.207 ± 0.005 | 0.532 ± 0.002 |
| Frame MLP | 0.194 ± 0.003 | 0.254 ± 0.004 | 0.556 ± 0.002 |
| Transformer w/o position | 0.195 ± 0.004 | 0.255 ± 0.005 | 0.526 ± 0.003 |
| Transformer (MSE only) | 0.188 ± 0.003 | 0.247 ± 0.005 | 0.529 ± 0.001 |
| **Transformer (ours)** | 0.191 ± 0.003 | 0.250 ± 0.004 | 0.516 ± 0.002 |
