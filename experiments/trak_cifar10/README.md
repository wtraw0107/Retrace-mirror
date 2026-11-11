# TRAK on CIFAR-10 (ResNet-9)

First hands-on run of TRAK (`traker` 0.3.2, https://trak.readthedocs.io) on a fast setting, before moving to generative (music) models.

## Result in one line

The setup runs end to end. With 3 models, though, TRAK scores do not reproduce across retraining (median correlation 0.04 between two independent 3-model runs), so the duplicate experiment **cannot tell** whether A's attribution is split with its copy. The control run shows A loses its rank-1 score even without a copy.

## Setup

| item | value |
|---|---|
| model | ResNet-9 (`construct_rn9`, verbatim from the quickstart) |
| training | quickstart hyperparameters: SGD lr 0.4, momentum 0.9, wd 5e-4, 24 epochs, triangular LR (peak epoch 5), batch 512, flip augmentation, AMP |
| models | 3 independently trained, TRAK uses each model's final checkpoint (epoch 23) |
| TRAK | `task='image_classification'`, `proj_dim=2048`, CUDA projector (`fast_jl`), featurize batch 128, all 10,000 test images as targets |
| hardware | 1× RTX 3080 20 GB (shared). One run (3 models + TRAK) ≈ 7 min: training ≈ 1.5 min/model, TRAK ≈ 2.5 min |

Deviations from the official quickstart:
- The notebook trains **1** model and uses its 5 intermediate checkpoints (epochs 12/15/18/21/23) as 5 `model_id`s. Here: 3 models × final checkpoint, as requested. (The docs prose says "three models"; its code trains one.)
- `proj_dim=2048` (notebook: 4096; docs default: 2048).
- `torch.amp` instead of deprecated `torch.cuda.amp`.
- Kept from the quickstart: the featurization loader uses `augment=True` (random horizontal flip) on training images.

Environment note: `fast_jl` 0.1.3 does not build against CUDA 13 because its `setup.py` hard-codes `TORCH_CUDA_ARCH_LIST="7.0+PTX"` (CUDA 13 dropped `compute_70`). It was built from the sdist with the arch changed to `8.6` (`/workspace/shared/hyuntae/build_fastjl/`). Check: on vectors with known cosine (1, 0.9, 0.5, 0, −0.5), the CUDA projector gives (1.0, 0.897, 0.494, −0.042, −0.517) and `BasicProjector` gives (1.0, 0.901, 0.521, 0.0, −0.514).

## Test accuracy

| run | model 0 | model 1 | model 2 |
|---|---|---|---|
| base (seeds 0–2, 50,000 train) | 0.9174 | 0.9185 | 0.9166 |
| dup (seeds 0–2, 50,001 train = + copy of A) | 0.9163 | 0.9148 | 0.9176 |
| control (seeds 3–5, 50,000 train) | 0.9146 | 0.9154 | 0.9177 |

## 1. Top-5 figure

![top-5](top5_trak.png)

Test images #0–#4 (first five of the CIFAR-10 test set), base run.
- The fraction of top-5 training images that share the test image's label is 0.20 for the base run (first 1,000 test images); chance is 0.10.
- Train #46371 is top-3 for both test #0 (cat) and test #3 (airplane). Train #42866 is top-5 for both test #0 and test #4.

## 2. Duplicate experiment

Target: test #0 (cat). A = its top-1 training image in the base run, train #46371 (cat). The dup run appends a copy of A at index 50000, then retrains the 3 models (seeds 0–2) and recomputes TRAK.

| quantity | score | rank |
|---|---|---|
| base: A | 23.67 | 1 / 50,000 |
| dup: A (original) | 0.034 | 8,403 / 50,001 |
| dup: copy of A | −0.0046 | 36,751 / 50,001 |
| dup: A + copy | 0.029 | – |
| **control** (no copy, seeds 3–5): A | −0.00009 | 29,512 / 50,000 |

How stable the scores are across runs:

| comparison | Pearson corr. of scores over 50k training images |
|---|---|
| base vs dup, test #0 | 0.064 |
| base vs control, test #0 | 0.040 |
| base vs dup, first 200 test images (median) | 0.047 |
| base vs control, first 200 test images (median) | 0.044 |

The top-6 training images for test #0 share no images across base, dup and control (full lists are in `dup_results.md`).

Facts:
- Without a copy, retraining alone drops A from score 23.7 (rank 1) to ≈0 (rank 29,512).
- So the drop in the dup run (A + copy = 0.029) cannot be separated from retraining noise.
- Within the dup run, A and its copy are the same image but score differently (0.034 vs −0.0046). Two things in this setup can cause that: the random flip applied during featurization, and score noise.

Inference (mine, not tested):
- With 3 models, between-run noise dominates, so this design cannot measure "credit splitting". Two fixes:
  - (a) Many more models, so retraining noise averages out. The TRAK paper uses far more than 3; the exact number needed here is unknown.
  - (b) Isolate the attribution formula's effect: keep the base models fixed, add only the copy's features, and featurize with `augment=False` so A and the copy get identical features.

## Files

Code (here):
- `trak_cifar.py`: training and TRAK.
  - `train|trak --tag {base,dup,ctrl} [--dup-index 46371] [--seed-offset 3]`
- `analyze.py`: figure and duplicate table.
  - `pick`: choose the target test image and A.
  - `figure`: make the top-5 figure.
  - `dup --dup-index 46371`: build the duplicate table.
- `top5_trak.png`: the top-5 figure.
- `dup_results.md`: raw numbers for the duplicate table.

Not committed (`/workspace/shared/hyuntae/`):
- `data/`: CIFAR-10.
- `checkpoints/{base,dup,ctrl}/`: `sd_{i}_epoch_{12,15,18,21,23}.pt` and `test_acc.json`.
- `trak_results/{base,dup,ctrl}/`: TRAK features; scores are in `scores/quickstart.mmap` with shape (train, test).
- `logs/`: run logs.
- `.venv/`: Python 3.12, torch 2.14.1+cu130, traker 0.3.2, fast_jl 0.1.3 (built for sm_86).
- `build_fastjl/`: the patched `fast_jl` build.

Reproduce:
```bash
source /workspace/shared/hyuntae/.venv/bin/activate
python trak_cifar.py train --tag base && python trak_cifar.py trak --tag base
python analyze.py pick && python analyze.py figure
python trak_cifar.py train --tag dup --dup-index 46371 && python trak_cifar.py trak --tag dup --dup-index 46371
python trak_cifar.py train --tag ctrl --seed-offset 3 && python trak_cifar.py trak --tag ctrl
python analyze.py dup --dup-index 46371
```
Both `train` and `trak` exit if `nvidia-smi` shows another process on the GPU.
