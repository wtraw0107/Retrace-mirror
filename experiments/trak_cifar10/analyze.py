"""Figure (top-5 TRAK training images for 5 test images) and duplicate-experiment table.

  python analyze.py pick              # print target test image and its top-1 train image A
  python analyze.py figure            # top-5 figure from the base run
  python analyze.py dup --dup-index A # table: base score of A vs dup-run scores of [A, copy]
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torchvision
from numpy.lib.format import open_memmap

ROOT = Path("/workspace/shared/hyuntae")
TRAK_OUT = ROOT / "trak_results"
HERE = Path(__file__).parent
TEST_IDS = [0, 1, 2, 3, 4]   # 5 test images for the figure (first five of the test set)
TARGET = TEST_IDS[0]         # test image used for the duplicate experiment
N_TRAIN = 50000


def load_scores(tag):
    return open_memmap(TRAK_OUT / tag / "scores" / "quickstart.mmap", mode="r")  # (train, test)


def cmd_pick(_):
    s = np.asarray(load_scores("base")[:, TARGET])
    order = np.argsort(-s)
    print(json.dumps({"target_test": TARGET, "A": int(order[0]),
                      "top5": order[:5].tolist(), "top5_scores": s[order[:5]].tolist()}))


def cmd_figure(_):
    tr = torchvision.datasets.CIFAR10(str(ROOT / "data"), train=True)
    te = torchvision.datasets.CIFAR10(str(ROOT / "data"), train=False)
    cls = tr.classes
    S = load_scores("base")
    fig, axes = plt.subplots(len(TEST_IDS), 6, figsize=(12, 2.3 * len(TEST_IDS)))
    for r, t in enumerate(TEST_IDS):
        s = np.asarray(S[:, t])
        top = np.argsort(-s)[:5]
        im, lab = te[t]
        axes[r, 0].imshow(im)
        axes[r, 0].set_title(f"TEST #{t}\n{cls[lab]}", fontsize=9, color="tab:blue")
        for c, j in enumerate(top, start=1):
            im, lab = tr[int(j)]
            axes[r, c].imshow(im)
            axes[r, c].set_title(f"train #{j} {cls[lab]}\nscore {s[j]:.3g}", fontsize=8)
    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("TRAK top-5 training images (3x ResNet-9, proj_dim=2048)")
    fig.tight_layout()
    out = HERE / "top5_trak.png"
    fig.savefig(out, dpi=120)
    print("saved", out)


def cmd_dup(args):
    A = args.dup_index
    base = np.asarray(load_scores("base")[:, TARGET])
    dup = np.asarray(load_scores("dup")[:, TARGET])
    assert dup.shape[0] == N_TRAIN + 1
    rank = lambda v, i: int((v > v[i]).sum()) + 1
    rows = [
        ("base: A", base[A], rank(base, A)),
        ("dup: A (original)", dup[A], rank(dup, A)),
        ("dup: A copy (idx 50000)", dup[N_TRAIN], rank(dup, N_TRAIN)),
        ("dup: A + copy", dup[A] + dup[N_TRAIN], None),
    ]
    # context: score scale of the rest of the top
    base_top = np.argsort(-base)[:6]
    dup_top = np.argsort(-dup)[:6]
    lines = ["| quantity | score | rank (of 50000 / 50001) |", "|---|---|---|"]
    for name, v, rk in rows:
        lines.append(f"| {name} | {v:.4g} | {rk if rk else '-'} |")
    lines.append(f"| ratio (A + copy) / base A | {(dup[A] + dup[N_TRAIN]) / base[A]:.3f} | - |")
    lines.append(f"| ratio dup A / base A | {dup[A] / base[A]:.3f} | - |")
    ctrl_path = TRAK_OUT / "ctrl" / "scores" / "quickstart.mmap"
    extra = {}
    if ctrl_path.exists():
        ctrl = np.asarray(load_scores("ctrl")[:, TARGET])
        lines.append(f"| control (no copy, seeds 3-5): A | {ctrl[A]:.4g} | {rank(ctrl, A)} |")
        extra = {"ctrl_A": float(ctrl[A]),
                 "ctrl_top6": [[int(i), float(ctrl[i])] for i in np.argsort(-ctrl)[:6]],
                 "corr_base_vs_ctrl": float(np.corrcoef(base, ctrl)[0, 1])}
    out = {
        "target_test": TARGET, "A": A,
        "base_A": float(base[A]), "dup_A": float(dup[A]), "dup_copy": float(dup[N_TRAIN]),
        "base_top6": [[int(i), float(base[i])] for i in base_top],
        "dup_top6": [[int(i), float(dup[i])] for i in dup_top],
        "corr_base_vs_dup_first50000": float(np.corrcoef(base, dup[:N_TRAIN])[0, 1]),
        **extra,
    }
    print("\n".join(lines))
    print(json.dumps(out, indent=1))
    (HERE / "dup_results.md").write_text("\n".join(lines) + "\n\n```json\n" + json.dumps(out, indent=1) + "\n```\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["pick", "figure", "dup"])
    p.add_argument("--dup-index", type=int)
    a = p.parse_args()
    {"pick": cmd_pick, "figure": cmd_figure, "dup": cmd_dup}[a.cmd](a)
