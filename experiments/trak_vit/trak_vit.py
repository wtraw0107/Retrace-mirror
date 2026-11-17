"""TRAK on CIFAR-10 with a small Vision Transformer (ViT).

Starting point: experiments/trak_cifar10/trak_cifar.py (Hyuntae, ResNet-9).
Everything that is not about the model or its training is kept the same on purpose,
so ResNet-9 and ViT results can be compared directly.

What changed vs. trak_cifar.py
  1. ROOT -> /workspace/shared/minjun (own data / checkpoints / TRAK results).
     CIFAR-10 is downloaded there on first use (download=True).
  2. Model: construct_rn9() -> construct_vit(), a small ViT written by hand (see below).
  3. Training: SGD lr 0.4 / 24 epochs -> AdamW + linear warmup + cosine decay / 50 epochs,
     plus RandomCrop(32, padding=4) augmentation. ViTs train badly with the ResNet recipe.
  4. Checkpoints: saved at 5 epochs spread over the second half of training
     (for 50 epochs: 24, 30, 37, 43, 49; 0-indexed) and listed in ckpt_epochs.json.
     `trak` uses all of them by default (checkpoint ensemble), because Hyuntae's round 2
     showed final-checkpoint-only TRAK with 3 models is too noisy.
  5. --epochs-train / --num-targets make a quick smoke test possible (see Usage).

Kept from trak_cifar.py
  - CIFAR-10 normalization, featurization with flip augmentation, proj_dim=2048,
    fp16 autocast + GradScaler, check_gpu() (the GPU is shared), CLI layout.
  - The duplicate experiment (--dup-index) still works.

Data / checkpoints / TRAK outputs live under ROOT (not committed).

Usage (run from anywhere; paths are absolute)
  # 0) smoke test: 1 model, 2 epochs, TRAK on 100 test images. Should finish in a few minutes.
  python3 trak_vit.py train --tag smoke --n-models 1 --epochs-train 2
  python3 trak_vit.py trak  --tag smoke --n-models 1 --num-targets 100
  # 1) real run: 3 models x 5 checkpoints = 15 TRAK model_ids
  python3 trak_vit.py train --tag base
  python3 trak_vit.py trak  --tag base --num-targets 1000 --del-grads
"""
import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW, lr_scheduler

ROOT = Path("/workspace/shared/minjun")          # changed: was /workspace/shared/hyuntae
DATA = ROOT / "data"
CKPT = ROOT / "checkpoints"
TRAK_OUT = ROOT / "trak_results"
N_MODELS = 3
TRAIN_EPOCHS = 50
PROJ_DIM = 2048
MEAN, STD = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.201)


# ---------------- model: a small ViT for 32x32 images ----------------
# How a ViT sees an image:
#   1. Cut the 32x32 image into 4x4 patches -> 8x8 = 64 patches.
#   2. Turn each patch into a 256-dim vector ("token") with one linear layer
#      (a Conv2d with kernel = stride = patch size does exactly that).
#   3. Prepend a learnable [CLS] token and add learnable position embeddings,
#      because attention itself does not know where a patch was.
#   4. Run 6 Transformer blocks. Each block = self-attention (every token looks at every
#      other token) + an MLP, each wrapped in LayerNorm + residual connection.
#   5. Classify from the final [CLS] token.
# Size: ~3.2M parameters (ResNet-9 is ~6.5M), so the two models are in the same range.
#
# Why attention is written out by hand instead of nn.MultiheadAttention / timm:
# TRAK computes one gradient per training example with torch.func (vmap + grad).
# Plain matmul + softmax is guaranteed to work there; fused attention kernels may not.
# No dropout and no BatchNorm, so the model behaves the same per example and in a batch.

class Attention(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):                                   # x: (B, N, dim)
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)                # each: (B, heads, N, head_dim)
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = attn.softmax(dim=-1)                         # who looks at whom
        out = (attn @ v).transpose(1, 2).reshape(B, N, D)
        return self.proj(out)


class Block(nn.Module):
    def __init__(self, dim, n_heads, mlp_ratio):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, n_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, mlp_ratio * dim), nn.GELU(),
                                 nn.Linear(mlp_ratio * dim, dim))

    def forward(self, x):
        x = x + self.attn(self.norm1(x))    # pre-norm residual
        x = x + self.mlp(self.norm2(x))
        return x


class ViT(nn.Module):
    def __init__(self, num_classes=10, img_size=32, patch=4, dim=256, depth=6,
                 n_heads=4, mlp_ratio=2):
        super().__init__()
        n_patches = (img_size // patch) ** 2
        self.patch_embed = nn.Conv2d(3, dim, kernel_size=patch, stride=patch)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, dim))
        self.blocks = nn.Sequential(*[Block(dim, n_heads, mlp_ratio) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            nn.init.zeros_(m.bias)

    def forward(self, x):                                   # x: (B, 3, 32, 32)
        x = self.patch_embed(x).flatten(2).transpose(1, 2)  # (B, 64, dim)
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos_embed     # (B, 65, dim)
        x = self.blocks(x)
        return self.head(self.norm(x[:, 0]))                # logits from [CLS]


def construct_vit(num_classes=10):
    return ViT(num_classes=num_classes)


# ---------------- data (same as trak_cifar.py, + RandomCrop for training) ----------------
def get_dataset(split="train", augment=True, dup_index=None, crop=False):
    tfs = []
    if augment:
        if crop:  # training only: ViTs need more augmentation than ResNets
            tfs.append(torchvision.transforms.RandomCrop(32, padding=4))
        tfs += [torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.RandomAffine(0)]
    tfs += [torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(MEAN, STD)]
    ds = torchvision.datasets.CIFAR10(root=str(DATA), download=True,
                                      train=(split == "train"),
                                      transform=torchvision.transforms.Compose(tfs))
    if split == "train" and dup_index is not None:
        # copy of A is appended -> lives at index len(ds) (= 50000)
        ds = torch.utils.data.ConcatDataset([ds, torch.utils.data.Subset(ds, [dup_index])])
    return ds


def get_dataloader(batch_size=256, num_workers=8, split="train", shuffle=False,
                   augment=True, dup_index=None, crop=False):
    ds = get_dataset(split, augment, dup_index, crop)
    return torch.utils.data.DataLoader(ds, shuffle=shuffle, batch_size=batch_size,
                                       num_workers=num_workers)


# ---------------- training (changed: AdamW + warmup + cosine) ----------------
def ckpt_epochs_for(epochs, n=5):
    """5 checkpoint epochs spread over the second half of training (0-indexed)."""
    eps = np.linspace(epochs / 2, epochs, n).round().astype(int) - 1
    return sorted(set(int(e) for e in eps if e >= 0))


def train(model, loader, ckpt_dir, save_epochs, lr=1e-3, epochs=TRAIN_EPOCHS,
          weight_decay=0.05, warmup_epochs=5, label_smoothing=0.0, model_id=0):
    # No weight decay on biases, LayerNorm weights, cls token, position embeddings (standard for ViT).
    decay = [p for n, p in model.named_parameters() if p.ndim >= 2 and "pos_embed" not in n]
    no_decay = [p for n, p in model.named_parameters() if p.ndim < 2 or "pos_embed" in n]
    opt = AdamW([{"params": decay, "weight_decay": weight_decay},
                 {"params": no_decay, "weight_decay": 0.0}], lr=lr)
    iters_per_epoch = len(loader)
    total = epochs * iters_per_epoch
    warmup = min(warmup_epochs * iters_per_epoch, total // 2)

    def lr_lambda(step):  # linear warmup, then cosine down to 0
        if step < warmup:
            return (step + 1) / warmup
        return 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup)))

    scheduler = lr_scheduler.LambdaLR(opt, lr_lambda)
    scaler = torch.amp.GradScaler("cuda")
    loss_fn = CrossEntropyLoss(label_smoothing=label_smoothing)
    for ep in range(epochs):
        model.train()
        for ims, labs in loader:
            ims, labs = ims.cuda(), labs.cuda()
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda"):
                loss = loss_fn(model(ims), labs)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            scheduler.step()
        if ep in save_epochs:
            torch.save(model.state_dict(), ckpt_dir / f"sd_{model_id}_epoch_{ep}.pt")
        print(f"model {model_id} epoch {ep} loss {loss.item():.3f}", flush=True)
    return model


@torch.no_grad()
def test_acc(model):
    model.eval()
    loader = get_dataloader(batch_size=512, split="val", augment=False)
    correct = n = 0
    for ims, labs in loader:
        with torch.autocast("cuda"):
            out = model(ims.cuda())
        correct += (out.argmax(1).cpu() == labs).sum().item()
        n += len(labs)
    return correct / n


def gpu_busy_by_others():
    """True if any compute process not owned by us is on the GPU."""
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                          "--format=csv,noheader"], capture_output=True, text=True).stdout
    pids = [l.split(",")[0].strip() for l in out.strip().splitlines() if l.strip()]
    return [p for p in pids if int(p) != os.getpid()]


def check_gpu():
    others = gpu_busy_by_others()
    if others:
        sys.exit(f"GPU in use by other processes {others}; stopping. Check `nvidia-smi`.")


def cmd_train(args):
    ckpt_dir = CKPT / args.tag
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    save_epochs = ckpt_epochs_for(args.epochs_train)
    (ckpt_dir / "ckpt_epochs.json").write_text(json.dumps(save_epochs))
    loader = get_dataloader(batch_size=512, split="train", shuffle=True,
                            dup_index=args.dup_index, crop=True)
    print(f"train set size {len(loader.dataset)}, checkpoints at epochs {save_epochs}")
    acc_file = ckpt_dir / "test_acc.json"
    accs = json.loads(acc_file.read_text()) if acc_file.exists() else {}
    seed_file = ckpt_dir / "seeds.json"
    seeds = json.loads(seed_file.read_text()) if seed_file.exists() else {}
    for i in range(args.start_id, args.start_id + args.n_models):
        check_gpu()  # re-check before every model: the GPU is shared
        seed = args.seed_offset + i
        torch.manual_seed(seed)  # same seed per model_id across base & dup runs
        model = construct_vit().cuda()
        if i == args.start_id:
            print(f"ViT parameters: {sum(p.numel() for p in model.parameters()):,}")
        model = train(model, loader, ckpt_dir, save_epochs, epochs=args.epochs_train, model_id=i)
        accs[str(i)] = test_acc(model)
        seeds[str(i)] = seed
        print(f"model {i} (seed {seed}) test acc {accs[str(i)]:.4f}", flush=True)
        acc_file.write_text(json.dumps(accs, indent=1))
        seed_file.write_text(json.dumps(seeds, indent=1))


def cmd_trak(args):
    from trak import TRAKer
    check_gpu()
    ckpt_dir = CKPT / args.tag
    # default: every checkpoint saved by `train` (checkpoint ensemble)
    if args.epochs:
        epochs = [int(e) for e in args.epochs.split(",")]
    else:
        epochs = json.loads((ckpt_dir / "ckpt_epochs.json").read_text())
    print(f"TRAK on {args.n_models} model(s) x epochs {epochs}")
    # one TRAK model_id per (model, epoch) checkpoint; model_id = model * len(epochs) + epoch index
    ids, ckpts = [], []
    for i in range(args.n_models):
        for j, ep in enumerate(epochs):
            ids.append(i * len(epochs) + j)
            ckpts.append(torch.load(ckpt_dir / f"sd_{i}_epoch_{ep}.pt", map_location="cpu"))
    # same as trak_cifar.py: featurization loader uses flip augmentation, shuffle=False
    loader_train = get_dataloader(batch_size=128, split="train", dup_index=args.dup_index)
    model = construct_vit().cuda().eval()
    traker = TRAKer(model=model, task="image_classification", proj_dim=PROJ_DIM,
                    train_set_size=len(loader_train.dataset),
                    save_dir=str(TRAK_OUT / (args.out_tag or args.tag)),
                    projector_seed=args.proj_seed)
    for model_id, ckpt in zip(ids, ckpts):
        check_gpu()
        traker.load_checkpoint(ckpt, model_id=model_id)
        for batch in loader_train:
            batch = [x.cuda() for x in batch]
            traker.featurize(batch=batch, num_samples=batch[0].shape[0])
        print(f"featurized model {model_id}", flush=True)
    traker.finalize_features(del_grads=args.del_grads)
    targets = get_dataset(split="val", augment=False)
    if args.num_targets:
        targets = torch.utils.data.Subset(targets, range(args.num_targets))
    loader_targets = torch.utils.data.DataLoader(targets, batch_size=128, num_workers=8)
    for model_id, ckpt in zip(ids, ckpts):
        traker.start_scoring_checkpoint(exp_name="quickstart", checkpoint=ckpt,
                                        model_id=model_id, num_targets=len(targets))
        for batch in loader_targets:
            batch = [x.cuda() for x in batch]
            traker.score(batch=batch, num_samples=batch[0].shape[0])
    scores = traker.finalize_scores(exp_name="quickstart")
    print("scores shape (train, test):", scores.shape)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["train", "trak"])
    p.add_argument("--tag", required=True)
    p.add_argument("--dup-index", type=int, default=None)
    p.add_argument("--seed-offset", type=int, default=0,
                   help="model i uses seed offset+i")
    p.add_argument("--start-id", type=int, default=0, help="train: first model_id")
    p.add_argument("--n-models", type=int, default=N_MODELS,
                   help="train: how many models from start-id; trak: use model_ids 0..n-1")
    p.add_argument("--epochs-train", type=int, default=TRAIN_EPOCHS,
                   help="train: number of epochs (smoke test: 2)")
    p.add_argument("--proj-seed", type=int, default=0, help="trak: projector_seed")
    p.add_argument("--out-tag", default=None, help="trak: output dir name (default = tag)")
    p.add_argument("--num-targets", type=int, default=None, help="trak: first N test images")
    p.add_argument("--del-grads", action="store_true", help="trak: drop raw projected grads")
    p.add_argument("--epochs", default=None,
                   help="trak: comma-separated checkpoint epochs (default: all in ckpt_epochs.json)")
    a = p.parse_args()
    {"train": cmd_train, "trak": cmd_trak}[a.cmd](a)
