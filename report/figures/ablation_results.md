# Ablation results

| variant | CE | ppl | load_balance | specialization | params(M) |
|---|---|---|---|---|---|
| baseline (top2, 8E, swiglu, rope) | 1.109 | 3.0 | 0.988 | 0.253 | 15.43 |
| top1_gating | 1.145 | 3.1 | 0.944 | 0.450 | 15.43 |
| 16_experts | 1.076 | 2.9 | 0.975 | 0.462 | 17.80 |
| gelu_expert | 1.063 | 2.9 | 0.993 | 0.222 | 15.43 |
| learnable_pe | 1.187 | 3.3 | 0.984 | 0.351 | 15.44 |
| no_load_balance | 1.078 | 2.9 | 0.935 | 0.504 | 15.43 |
