---
layout: post
title: "TRAK on CIFAR-10, part 1: quickstart and the duplicate test"
date: 2026-11-11 10:00:00 +0900
tags: [trak, cifar10, duplicate]
---

**Bottom line:** the TRAK pipeline runs end to end. With 3 models, though, the scores don't reproduce across retraining, so the duplicate test couldn't tell whether a copy splits the original's credit.

<!--more-->

## Why this experiment

The goal is royalties from training-data attribution, and music is the end target. This is the first hands-on TRAK run, on CIFAR-10 so that each run takes minutes. The question that matters for royalties: **if a creator's work appears twice in the training set, does TRAK split its credit between the two copies?**

## Setup

| item | value |
|---|---|
| model | ResNet-9 (official `traker` quickstart) |
| training | SGD lr 0.4, 24 epochs, triangular LR, batch 512, flip augmentation |
| models | 3 independent models, final checkpoint (epoch 23) each |
| TRAK | `traker` 0.3.2, `proj_dim=2048`, CUDA projector |
| test accuracy | 91.5–91.9% for every model |
| cost | ≈ 7 min per run (3 models + TRAK) on one RTX 3080 |

Two differences from the official quickstart: the notebook trains **1** model and uses its 5 intermediate checkpoints, and it uses `proj_dim=4096`.

## Top-5 attributions

![Top-5 TRAK training images for 5 test images]({{ '/assets/img/top5_trak.png' | relative_url }})

- Only **20%** of top-5 training images share the test image's label (chance is 10%).
- The same few training images show up for unrelated test images. For example, train #46371 is in the top 3 for both test #0 (cat) and test #3 (airplane).

## Duplicate test

Target: test #0 (cat). A is its top-1 training image, train #46371. We appended a copy of A, retrained the 3 models with the same seeds, and recomputed TRAK. As a control, we retrained **without** the copy using new seeds.

| quantity | score | rank |
|---|---|---|
| base: A | 23.67 | 1 / 50,000 |
| with copy: A | 0.034 | 8,403 / 50,001 |
| with copy: copy of A | −0.0046 | 36,751 / 50,001 |
| **control, no copy:** A | ≈ 0 | 29,512 / 50,000 |

Retraining alone drops A from rank 1 to 29,512, so the "with copy" numbers can't be separated from retraining noise. Across all 50k training images, scores from two independent 3-model runs correlate at a median of only **0.04**.

**Next:** find out why the scores are this unstable ([part 2]({{ '/2026/11/11/trak-noise-diagnosis/' | relative_url }})).

Code: `experiments/trak_cifar10/` · full numbers: `experiments/trak_cifar10/README.md`
