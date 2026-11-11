"""TRAK on CIFAR-10 ResNet-9, following the official traker quickstart
(https://trak.readthedocs.io/en/latest/quickstart.html, examples/cifar_quickstart.ipynb).

Deviations from the quickstart (all deliberate, see README):
  - 3 independently trained models; TRAK uses each model's final checkpoint (epoch 23).
    (The notebook trains 1 model and uses its 5 intermediate checkpoints.)
  - proj_dim=2048 (notebook uses 4096; docs default is 2048).
  - torch.amp instead of the deprecated torch.cuda.amp.
  - Optional duplicate: `--dup-index A` appends a copy of training image A at index 50000.

Data / checkpoints / TRAK outputs live under ROOT (not committed).

Usage:
  python trak_cifar.py train --tag base
  python trak_cifar.py trak  --tag base
  python trak_cifar.py train --tag dup --dup-index A
  python trak_cifar.py trak  --tag dup --dup-index A
  python trak_cifar.py train --tag ctrl --seed-offset 3   # no-dup control, new seeds
  python trak_cifar.py trak  --tag ctrl
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision
from torch.nn import CrossEntropyLoss
from torch.optim import SGD, lr_scheduler

ROOT = Path("/workspace/shared/hyuntae")
DATA = ROOT / "data"
CKPT = ROOT / "checkpoints"
TRAK_OUT = ROOT / "trak_results"
N_MODELS = 3
FINAL_EPOCH = 23
PROJ_DIM = 2048
MEAN, STD = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.201)


# ---------------- model (verbatim from quickstart) ----------------
class Mul(torch.nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.weight = weight

    def forward(self, x):
        return x * self.weight


class Flatten(torch.nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Residual(torch.nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x):
        return x + self.module(x)


def construct_rn9(num_classes=10):
    def conv_bn(channels_in, channels_out, kernel_size=3, stride=1, padding=1, groups=1):
        return torch.nn.Sequential(
            torch.nn.Conv2d(channels_in, channels_out, kernel_size=kernel_size,
                            stride=stride, padding=padding, groups=groups, bias=False),
            torch.nn.BatchNorm2d(channels_out),
            torch.nn.ReLU(inplace=True),
        )
    return torch.nn.Sequential(
        conv_bn(3, 64, kernel_size=3, stride=1, padding=1),
        conv_bn(64, 128, kernel_size=5, stride=2, padding=2),
        Residual(torch.nn.Sequential(conv_bn(128, 128), conv_bn(128, 128))),
        conv_bn(128, 256, kernel_size=3, stride=1, padding=1),
        torch.nn.MaxPool2d(2),
        Residual(torch.nn.Sequential(conv_bn(256, 256), conv_bn(256, 256))),
        conv_bn(256, 128, kernel_size=3, stride=1, padding=0),
        torch.nn.AdaptiveMaxPool2d((1, 1)),
        Flatten(),
        torch.nn.Linear(128, num_classes, bias=False),
        Mul(0.2),
    )


# ---------------- data (quickstart transforms) ----------------
def get_dataset(split="train", augment=True, dup_index=None):
    if augment:
        tf = torchvision.transforms.Compose([
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomAffine(0),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(MEAN, STD)])
    else:
        tf = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(MEAN, STD)])
    ds = torchvision.datasets.CIFAR10(root=str(DATA), download=False,
                                      train=(split == "train"), transform=tf)
    if split == "train" and dup_index is not None:
        # copy of A is appended -> lives at index len(ds) (= 50000)
        ds = torch.utils.data.ConcatDataset([ds, torch.utils.data.Subset(ds, [dup_index])])
    return ds


def get_dataloader(batch_size=256, num_workers=8, split="train", shuffle=False,
                   augment=True, dup_index=None):
    ds = get_dataset(split, augment, dup_index)
    return torch.utils.data.DataLoader(ds, shuffle=shuffle, batch_size=batch_size,
                                       num_workers=num_workers)


# ---------------- training (quickstart hyperparameters) ----------------
def train(model, loader, ckpt_dir, lr=0.4, epochs=24, momentum=0.9,
          weight_decay=5e-4, lr_peak_epoch=5, label_smoothing=0.0, model_id=0):
    opt = SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    iters_per_epoch = len(loader)
    lr_schedule = np.interp(np.arange((epochs + 1) * iters_per_epoch),
                            [0, lr_peak_epoch * iters_per_epoch, epochs * iters_per_epoch],
                            [0, 1, 0])
    scheduler = lr_scheduler.LambdaLR(opt, lr_schedule.__getitem__)
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
        if ep in [12, 15, 18, 21, 23]:
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


def cmd_train(args):
    others = gpu_busy_by_others()
    if others:
        sys.exit(f"GPU in use by other processes {others}; stopping.")
    ckpt_dir = CKPT / args.tag
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    loader = get_dataloader(batch_size=512, split="train", shuffle=True, dup_index=args.dup_index)
    print(f"train set size {len(loader.dataset)}")
    accs = {}
    for i in range(N_MODELS):
        torch.manual_seed(args.seed_offset + i)  # same seed per model_id across base & dup runs
        model = construct_rn9().to(memory_format=torch.channels_last).cuda()
        model = train(model, loader, ckpt_dir, model_id=i)
        accs[i] = test_acc(model)
        print(f"model {i} test acc {accs[i]:.4f}", flush=True)
    (ckpt_dir / "test_acc.json").write_text(json.dumps(accs, indent=1))


def cmd_trak(args):
    from trak import TRAKer
    others = gpu_busy_by_others()
    if others:
        sys.exit(f"GPU in use by other processes {others}; stopping.")
    ckpt_dir = CKPT / args.tag
    ckpt_files = [ckpt_dir / f"sd_{i}_epoch_{FINAL_EPOCH}.pt" for i in range(N_MODELS)]
    ckpts = [torch.load(f, map_location="cpu") for f in ckpt_files]
    # quickstart: featurization loader uses augment=True default, shuffle=False
    loader_train = get_dataloader(batch_size=128, split="train", dup_index=args.dup_index)
    model = construct_rn9().to(memory_format=torch.channels_last).cuda().eval()
    traker = TRAKer(model=model, task="image_classification", proj_dim=PROJ_DIM,
                    train_set_size=len(loader_train.dataset),
                    save_dir=str(TRAK_OUT / args.tag))
    for model_id, ckpt in enumerate(ckpts):
        traker.load_checkpoint(ckpt, model_id=model_id)
        for batch in loader_train:
            batch = [x.cuda() for x in batch]
            traker.featurize(batch=batch, num_samples=batch[0].shape[0])
    traker.finalize_features()
    loader_targets = get_dataloader(batch_size=128, split="val", augment=False)
    for model_id, ckpt in enumerate(ckpts):
        traker.start_scoring_checkpoint(exp_name="quickstart", checkpoint=ckpt,
                                        model_id=model_id,
                                        num_targets=len(loader_targets.dataset))
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
                   help="model i uses seed offset+i (control run: 3)")
    a = p.parse_args()
    {"train": cmd_train, "trak": cmd_trak}[a.cmd](a)
