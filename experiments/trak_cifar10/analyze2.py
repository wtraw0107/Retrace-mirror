"""Round 2: score-scale diagnosis, projection-seed noise, model-count scaling.

  python analyze2.py scale_diag   # decompose A's score into raw dot product x out_to_loss
  python analyze2.py proj         # projection / flip / retrain noise, same 1000 test images
  python analyze2.py scaling      # cross-set correlation vs number of models + plot

traker 0.3.2 score (traker.py finalize_scores):
  score = mean_m(features_m @ target_grads_m) * mean_m(out_to_loss_m)
  features_m = X_m (X_m^T X_m)^-1 (projected train grads), out_to_loss = 1 - p(true class).
"raw" below = the first factor alone (no out_to_loss multiplier).
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
TR = ROOT / "trak_results"
HERE = Path(__file__).parent
N = 50000
A, TARGET = 46371, 0
NT = 1000  # first 1000 test images


def mm(path, sl=None):
    x = open_memmap(path, mode="r")
    return np.asarray(x if sl is None else x[sl]).astype(np.float32)


def per_model(tag, m, nt=NT):
    F = mm(TR / tag / str(m) / "features.mmap", slice(0, N))
    G = mm(TR / tag / str(m) / "quickstart_grads.mmap", slice(0, nt))
    Q = mm(TR / tag / str(m) / "out_to_loss.mmap", slice(0, N))
    return F @ G.T, Q  # (N, nt), (N, 1)


def ensemble(tag, ids, nt=NT):
    raw = 0.0
    q = 0.0
    for m in ids:
        r, qq = per_model(tag, m, nt)
        raw = raw + r
        q = q + qq
    raw, q = raw / len(ids), q / len(ids)
    return raw, q, raw * q


def colcorr(X, Y):
    """Pearson correlation per test column (over the 50k training images)."""
    X = X - X.mean(0)
    Y = Y - Y.mean(0)
    return (X * Y).sum(0) / np.sqrt((X * X).sum(0) * (Y * Y).sum(0))


def labels():
    ytr = np.array(torchvision.datasets.CIFAR10(str(ROOT / "data"), train=True).targets)
    yte = np.array(torchvision.datasets.CIFAR10(str(ROOT / "data"), train=False).targets)
    return ytr, yte


def same_label_top5(S, ytr, yte):
    top = np.argsort(-S, axis=0)[:5]
    return float((ytr[top] == yte[None, : S.shape[1]]).mean())


def cmd_scale_diag(_):
    out = {}
    for tag in ["base", "dup", "ctrl"]:
        rows = []
        for m in range(3):
            r, q = per_model(tag, m, nt=1)
            r, q = r[:, 0], q[:, 0]
            rows.append({"model": m, "raw_A": float(r[A]), "raw_rank_A": int((r > r[A]).sum()) + 1,
                         "Q_A": float(q[A]), "Q_median": float(np.median(q))})
        raw, q, s = ensemble(tag, range(3), nt=1)
        raw, q, s = raw[:, 0], q[:, 0], s[:, 0]
        out[tag] = {"per_model": rows, "raw_A": float(raw[A]), "raw_rank_A": int((raw > raw[A]).sum()) + 1,
                    "Qbar_A": float(q[A]), "Qbar_rank_A": int((q > q[A]).sum()) + 1,
                    "score_A": float(s[A]), "score_std": float(s.std()), "score_max": float(s.max())}
    # check reconstruction matches traker's saved scores
    saved = mm(TR / "base" / "scores" / "quickstart.mmap", (slice(0, N), TARGET))
    out["reconstruction_max_abs_err_base_test0"] = float(np.abs(saved - ensemble("base", range(3), nt=1)[2][:, 0]).max())
    # Q dominance on 1000 test images (base)
    ytr, yte = labels()
    raw, q, s = ensemble("base", range(3))
    thr = np.quantile(q, 0.99)
    out["base_top1_in_top1pct_Q"] = float((q[s.argmax(0), 0] >= thr).mean())
    out["base_distinct_top1"] = int(len(set(s.argmax(0).tolist())))
    out["base_top5_same_label_score"] = same_label_top5(s, ytr, yte)
    out["base_top5_same_label_raw"] = same_label_top5(raw, ytr, yte)
    out["Q_quantiles_base"] = {k: float(np.quantile(q, v)) for k, v in
                               [("p50", .5), ("p90", .9), ("p99", .99), ("max", 1.0)]}
    print(json.dumps(out, indent=1))
    (HERE / "round2_scale_diag.json").write_text(json.dumps(out, indent=1))


def cmd_proj(_):
    ref_raw, ref_q, ref_s = ensemble("base", range(3))
    res = {}
    for name, tag in [("flip only (base_proj0b: proj seed 0, re-featurized)", "base_proj0b"),
                      ("projection + flip (base_proj1: proj seed 1)", "base_proj1"),
                      ("retrain (ctrl: seeds 3-5)", "ctrl")]:
        raw, q, s = ensemble(tag, range(3))
        res[name] = {"score": float(np.median(colcorr(ref_s, s))),
                     "raw": float(np.median(colcorr(ref_raw, raw))),
                     "Qbar_corr": float(np.corrcoef(ref_q[:, 0], q[:, 0])[0, 1])}
    lines = ["| base (seeds 0-2, proj seed 0) vs | score corr | raw corr (no Q) | out_to_loss corr |",
             "|---|---|---|---|"]
    for k, v in res.items():
        lines.append(f"| {k} | {v['score']:.3f} | {v['raw']:.3f} | {v['Qbar_corr']:.3f} |")
    txt = "\n".join(lines)
    print(txt)
    (HERE / "round2_proj.md").write_text(txt + "\n\nMedian per-test Pearson over first 1000 test images.\n")


def cmd_scaling(_):
    ytr, yte = labels()
    ks = list(range(1, 11))
    sums = {}
    res = {k: {} for k in ks}
    for tag in ["scale_base", "scale_ctrl"]:
        raw_sum, q_sum = 0.0, 0.0
        for m in range(10):
            r, q = per_model(tag, m)
            raw_sum, q_sum = raw_sum + r, q_sum + q
            k = m + 1
            sums[(tag, k)] = (raw_sum / k, q_sum / k)
        # cross-check with traker's own 10-model scores
        saved = mm(TR / tag / "scores" / "quickstart.mmap", slice(0, N))
        rk, qk = sums[(tag, 10)]
        res.setdefault("recon_err", {})[tag] = float(np.abs(saved - rk * qk).max())
    for k in ks:
        (rb, qb), (rc, qc) = sums[("scale_base", k)], sums[("scale_ctrl", k)]
        cs, cr = colcorr(rb * qb, rc * qc), colcorr(rb, rc)
        res[k] = {"score_median": float(np.median(cs)), "score_q25": float(np.quantile(cs, .25)),
                  "score_q75": float(np.quantile(cs, .75)),
                  "raw_median": float(np.median(cr)), "raw_q25": float(np.quantile(cr, .25)),
                  "raw_q75": float(np.quantile(cr, .75)),
                  "Qbar_corr": float(np.corrcoef(qb[:, 0], qc[:, 0])[0, 1]),
                  "same_label_top5_score_base": same_label_top5(rb * qb, ytr, yte),
                  "same_label_top5_raw_base": same_label_top5(rb, ytr, yte)}
        if k in (3, 6, 10):
            print(k, json.dumps(res[k]))
    (HERE / "round2_scaling.json").write_text(json.dumps({str(k): v for k, v in res.items()}, indent=1))

    # plot: two series, palette slots 1-2 (blue, orange), light surface
    BLUE, ORANGE, INK, MUTED, SURF = "#2a78d6", "#eb6834", "#1f1f1e", "#6b6a64", "#fcfcfb"
    fig, ax = plt.subplots(figsize=(7, 4.2), facecolor=SURF)
    ax.set_facecolor(SURF)
    for key, color, label in [("raw", BLUE, "raw (without out_to_loss)"),
                              ("score", ORANGE, "TRAK score (traker default)")]:
        med = [res[k][f"{key}_median"] for k in ks]
        lo = [res[k][f"{key}_q25"] for k in ks]
        hi = [res[k][f"{key}_q75"] for k in ks]
        ax.fill_between(ks, lo, hi, color=color, alpha=0.15, linewidth=0)
        ax.plot(ks, med, color=color, linewidth=2, marker="o", markersize=5, label=label)
        ax.annotate(f"{med[-1]:.2f}", (ks[-1], med[-1]), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=9, color=INK)
        for k in (3, 6):
            ax.annotate(f"{med[k-1]:.2f}", (k, med[k-1]), xytext=(0, 8), textcoords="offset points",
                        ha="center", fontsize=8, color=MUTED)
    ax.set_xticks(ks)
    ax.set_xlim(0.6, 10.8)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("models per set (each set trained independently)", color=INK)
    ax.set_ylabel("median Pearson corr. between sets", color=INK)
    ax.set_title("TRAK agreement between two independent model sets\n"
                 "(1000 CIFAR-10 test images; band = interquartile range)", fontsize=10, color=INK)
    ax.grid(axis="y", color="#e6e5e0", linewidth=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c9c8c1")
    ax.tick_params(colors=MUTED)
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(HERE / "scaling_corr.png", dpi=130, facecolor=SURF)
    print("saved scaling_corr.png", json.dumps(res["recon_err"]))


def stats_pair(tb, ib, tc, ic, ytr, yte):
    """Q concentration, label agreement, and cross-set agreement for two ensembles."""
    rb, qb, sb = ensemble(tb, ib)
    rc, qc, sc = ensemble(tc, ic)
    conc = lambda q, s: float((q[s.argmax(0), 0] >= np.quantile(q, 0.99)).mean())
    return {
        "Q_conc_top1_in_top1pct (set1)": conc(qb, sb),
        "Q_conc_top1_in_top1pct (set2)": conc(qc, sc),
        "distinct_top1 (set1, of 1000)": int(len(set(sb.argmax(0).tolist()))),
        "Qbar_p50/p99/max (set1)": [float(np.quantile(qb, v)) for v in (.5, .99, 1.0)],
        "same_label_top5 score (set1)": same_label_top5(sb, ytr, yte),
        "same_label_top5 raw (set1)": same_label_top5(rb, ytr, yte),
        "corr score (set1 vs set2)": float(np.median(colcorr(sb, sc))),
        "corr raw (set1 vs set2)": float(np.median(colcorr(rb, rc))),
        "score_A_test0 (set1)": float(sb[A, TARGET]),
        "A_rank_test0 (set1)": int((sb[:, TARGET] > sb[A, TARGET]).sum()) + 1,
    }


def cmd_ckpt5(_):
    ytr, yte = labels()
    E = [12, 15, 18, 21, 23]
    res = {
        "final only, 3 models (base vs ctrl)": stats_pair("base", range(3), "ctrl", range(3), ytr, yte),
        "final only, 10 models (scale_base vs scale_ctrl)":
            stats_pair("scale_base", range(10), "scale_ctrl", range(10), ytr, yte),
        "quickstart: 3 models x 5 ckpts (ckpt5_base vs ckpt5_ctrl)":
            stats_pair("ckpt5_base", range(15), "ckpt5_ctrl", range(15), ytr, yte),
    }
    for j, ep in enumerate(E):  # single epoch, 3 models: ids model*5 + j
        ids = [m * 5 + j for m in range(3)]
        res[f"epoch {ep} only, 3 models (ckpt5)"] = stats_pair("ckpt5_base", ids, "ckpt5_ctrl", ids, ytr, yte)
    keys = ["Q_conc_top1_in_top1pct (set1)", "Q_conc_top1_in_top1pct (set2)", "distinct_top1 (set1, of 1000)",
            "same_label_top5 score (set1)", "same_label_top5 raw (set1)",
            "corr score (set1 vs set2)", "corr raw (set1 vs set2)"]
    short = ["Q conc. set1", "Q conc. set2", "distinct top-1", "same-label top-5 (score)",
             "same-label top-5 (raw)", "corr score", "corr raw"]
    lines = ["| ensemble per set | " + " | ".join(short) + " |", "|---" * (len(short) + 1) + "|"]
    for name, v in res.items():
        lines.append(f"| {name} | " + " | ".join(
            f"{v[k]:.3f}" if isinstance(v[k], float) else str(v[k]) for k in keys) + " |")
    txt = "\n".join(lines)
    print(txt)
    print(json.dumps(res, indent=1))
    (HERE / "round2_ckpt5.md").write_text(
        txt + "\n\nset1 = seeds 0-2 (+100s), set2 = independently trained seeds 3-5 (+200s). "
        "First 1000 test images; corr = median per-test Pearson over 50k train.\n\n```json\n"
        + json.dumps(res, indent=1) + "\n```\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["scale_diag", "proj", "scaling", "ckpt5"])
    a = p.parse_args()
    {"scale_diag": cmd_scale_diag, "proj": cmd_proj, "scaling": cmd_scaling, "ckpt5": cmd_ckpt5}[a.cmd](a)
