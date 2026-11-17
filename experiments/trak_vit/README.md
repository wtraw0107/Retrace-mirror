# TRAK on CIFAR-10 (small ViT)

Same experiment as [`../trak_cifar10`](../trak_cifar10/README.md) (Hyuntae, ResNet-9), with the model swapped for a small Vision Transformer. Goal: compare TRAK attributions between a CNN and a ViT under the same setup.

Status: code written, not yet run on the GPU server. Checked on CPU only: the model builds (3,195,146 parameters), and TRAK featurize → score runs end to end on dummy data.

## What differs from `trak_cifar10/trak_cifar.py`

| item | trak_cifar10 (ResNet-9) | trak_vit (ViT) |
|---|---|---|
| ROOT | `/workspace/shared/hyuntae` | `/workspace/shared/minjun` |
| CIFAR-10 | `download=False` | `download=True` (first run downloads ~170 MB) |
| model | ResNet-9, ~6.5M params | ViT: patch 4, dim 256, 6 blocks, 4 heads, MLP ×2, ~3.2M params |
| optimizer | SGD lr 0.4, momentum 0.9, wd 5e-4 | AdamW lr 1e-3, wd 0.05 (none on bias / norm / pos-embed) |
| schedule | triangular, peak at epoch 5 | linear warmup 5 epochs → cosine to 0 |
| epochs | 24 | 50 (`--epochs-train`) |
| training aug | flip | RandomCrop(32, pad 4) + flip |
| saved checkpoints | epochs 12, 15, 18, 21, 23 | 5 epochs over the 2nd half (50 ep: 24, 30, 37, 43, 49) |
| TRAK default | final checkpoint only | all saved checkpoints (ensemble) |

Unchanged: normalization, flip-augmented featurization loader, `proj_dim=2048`, fp16 autocast, `check_gpu()`, `--dup-index`, CLI.

## Run

```bash
cd ~/Retrace/experiments/trak_vit
nvidia-smi                                    # GPU is shared: check nobody is using it

# 1) smoke test (a few minutes): 1 model, 2 epochs, TRAK on 100 test images
python3 trak_vit.py train --tag smoke --n-models 1 --epochs-train 2
python3 trak_vit.py trak  --tag smoke --n-models 1 --num-targets 100

# 2) real run: 3 models x 5 checkpoints
python3 trak_vit.py train --tag base
python3 trak_vit.py trak  --tag base --num-targets 1000 --del-grads
```

Needs `torch`, `torchvision`, `traker` (`pip install traker`). `traker` uses the CUDA projector from `fast_jl` if it is installed; see the environment note in `../trak_cifar10/README.md` (Hyuntae rebuilt it for CUDA 13).

## Not known yet

- Test accuracy and wall-clock time on the server.
- Whether TRAK's per-example gradients work on the GPU with fp16 for this model (checked only on CPU, fp32).
