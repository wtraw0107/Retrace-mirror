---
layout: home
title: Dashboard
---

*Everything AI learned has an owner.* This is the running log of Retrace's attribution experiments: what we ran, and what we found.

## Experiments

| date | experiment | status | key result | post |
|---|---|---|---|---|
| 2026-11-11 | TRAK on CIFAR-10: quickstart, top-5, duplicate test | done | Duplicate test inconclusive: retraining alone moves A from rank 1 to 29,512 | [part 1]({{ '/2026/11/11/trak-cifar10-quickstart-and-duplicate/' | relative_url }}) |
| 2026-11-11 | TRAK noise diagnosis: Q factor, projection seed, model count, checkpoint ensemble | done | Agreement 0.045 → 0.22 with 3 → 15 members, not saturated | [part 2]({{ '/2026/11/11/trak-noise-diagnosis/' | relative_url }}) |
| next | LDS, small scale (20 × 50% subsets) | planned | Estimate: 15–17 min of GPU time with 1 model per subset | – |

## Headline numbers (TRAK, CIFAR-10, ResNet-9)

| setting | members per set | agreement between independent sets | same-label top-5 |
|---|---|---|---|
| final checkpoint, 3 models | 3 | 0.045 | 0.20 |
| final checkpoint, 10 models | 10 | 0.187 | 0.51 |
| 3 models × 5 checkpoints (quickstart) | 15 | **0.223** | **0.90** |

Agreement is the median, over 1,000 test images, of the Pearson correlation between two independently trained sets' scores across 50k training images.

![Agreement vs number of models]({{ '/assets/img/scaling_corr.png' | relative_url }})

