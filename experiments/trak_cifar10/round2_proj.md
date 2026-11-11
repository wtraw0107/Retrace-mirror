| base (seeds 0-2, proj seed 0) vs | score corr | raw corr (no Q) | out_to_loss corr |
|---|---|---|---|
| flip only (base_proj0b: proj seed 0, re-featurized) | 0.484 | 0.605 | 0.747 |
| projection + flip (base_proj1: proj seed 1) | 0.107 | 0.160 | 0.720 |
| retrain (ctrl: seeds 3-5) | 0.045 | 0.110 | 0.500 |

Median per-test Pearson over first 1000 test images.
