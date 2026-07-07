# Ablation results

| variant | CE | ppl | load_balance | specialization | params(M) |
|---|---|---|---|---|---|
| baseline (top2, 8E, swiglu, rope) | 1.151 | 3.2 | 0.993 | 0.276 | 15.43 |
| top1_gating | 1.110 | 3.0 | 0.963 | 0.532 | 15.43 |
| 16_experts | 1.078 | 2.9 | 0.974 | 0.490 | 17.80 |
| gelu_expert | 1.081 | 2.9 | 0.991 | 0.349 | 15.43 |
| learnable_pe | 1.080 | 2.9 | 0.989 | 0.273 | 15.44 |
| no_load_balance | 1.106 | 3.0 | 0.950 | 0.538 | 15.43 |
