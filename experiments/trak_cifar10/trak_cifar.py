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
  # projection-seed noise: same 3 base models, new projector seed (and a seed-0 re-featurization)
  python trak_cifar.py trak --tag base --out-tag base_proj1 --proj-seed 1 --num-targets 1000
  python trak_cifar.py trak --tag base --out-tag base_proj0b --proj-seed 0 --num-targets 1000
  # model-count scaling: grow base and ctrl to 10 models each
  python trak_cifar.py train --tag base --start-id 3 --n-models 7 --seed-offset 100
  python trak_cifar.py train --tag ctrl --start-id 3 --n-models 7 --seed-offset 200
  python trak_cifar.py trak --tag base --out-tag scale_base --n-models 10 --num-targets 1000 --del-grads
  python trak_cifar.py trak --tag ctrl --out-tag scale_ctrl --n-models 10 --num-targets 1000 --del-grads
  # quickstart-style checkpoint ensemble: 3 models x epochs 12,15,18,21,23 = 15 model_ids per set
  python trak_cifar.py trak --tag base --out-tag ckpt5_base --epochs 12,15,18,21,23 --num-targets 1000 --del-grads
  python trak_cifar.py trak --tag ctrl --out-tag ckpt5_ctrl --epochs 12,15,18,21,23 --num-targets 1000 --del-grads
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


def check_gpu():
    others = gpu_busy_by_others()
    if others:
        sys.exit(f"GPU in use by other processes {others}; stopping.")


def cmd_train(args):
    ckpt_dir = CKPT / args.tag
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    loader = get_dataloader(batch_size=512, split="train", shuffle=True, dup_index=args.dup_index)
    print(f"train set size {len(loader.dataset)}")
    acc_file = ckpt_dir / "test_acc.json"
    accs = json.loads(acc_file.read_text()) if acc_file.exists() else {}
    seed_file = ckpt_dir / "seeds.json"
    seeds = json.loads(seed_file.read_text()) if seed_file.exists() else {}
    for i in range(args.start_id, args.start_id + args.n_models):
        check_gpu()  # re-check before every model: the GPU is shared
        seed = args.seed_offset + i
        torch.manual_seed(seed)  # same seed per model_id across base & dup runs
        model = construct_rn9().to(memory_format=torch.channels_last).cuda()
        model = train(model, loader, ckpt_dir, model_id=i)
        accs[str(i)] = test_acc(model)
        seeds[str(i)] = seed
        print(f"model {i} (seed {seed}) test acc {accs[str(i)]:.4f}", flush=True)
        acc_file.write_text(json.dumps(accs, indent=1))
        seed_file.write_text(json.dumps(seeds, indent=1))


def cmd_trak(args):
    from trak import TRAKer
    check_gpu()
    ckpt_dir = CKPT / args.tag
    # one TRAK model_id per (model, epoch) checkpoint; model_id = model * len(epochs) + epoch index
    epochs = [int(e) for e in args.epochs.split(",")]
    ids, ckpts = [], []
    for i in range(args.n_models):
        for j, ep in enumerate(epochs):
            ids.append(i * len(epochs) + j)
            ckpts.append(torch.load(ckpt_dir / f"sd_{i}_epoch_{ep}.pt", map_location="cpu"))
    # quickstart: featurization loader uses augment=True default, shuffle=False
    loader_train = get_dataloader(batch_size=128, split="train", dup_index=args.dup_index)
    model = construct_rn9().to(memory_format=torch.channels_last).cuda().eval()
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
                   help="model i uses seed offset+i (control run: 3)")
    p.add_argument("--start-id", type=int, default=0, help="train: first model_id")
    p.add_argument("--n-models", type=int, default=N_MODELS,
                   help="train: how many models from start-id; trak: use model_ids 0..n-1")
    p.add_argument("--proj-seed", type=int, default=0, help="trak: projector_seed")
    p.add_argument("--out-tag", default=None, help="trak: output dir name (default = tag)")
    p.add_argument("--num-targets", type=int, default=None, help="trak: first N test images")
    p.add_argument("--del-grads", action="store_true", help="trak: drop raw projected grads")
    p.add_argument("--epochs", default=str(FINAL_EPOCH),
                   help="trak: comma-separated checkpoint epochs per model (quickstart: 12,15,18,21,23)")
    a = p.parse_args()
    {"train": cmd_train, "trak": cmd_trak}[a.cmd](a)
