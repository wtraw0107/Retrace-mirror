"""Slide figures for Section 5.5.1, built from the score matrices that attribute.py saves.

    python toy/figures.py heatmap --metric top10     # 5x5: query topic x document topic
    python toy/figures.py heatmap --metric share
    python toy/figures.py query --query 24          # one answer's credit over documents and authors
    python toy/figures.py seeds                     # top-1 / MRR over 5 seeds with 95% intervals

Figures and the numbers behind them (JSON) go to figures/.
"""

import argparse
import json
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from attribute import Run  # noqa: E402
from train import answer_logprobs, encode_queries, load_inputs  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCORES = os.path.join(ROOT, "runs", "attribution")
OUT = os.path.join(ROOT, "figures")

AREAS = [("shakespeare", "Shakespeare"), ("kpop", "K-pop"), ("olympic_games", "Olympics"),
         ("nobel_physics", "Nobel Physics"), ("dinosaurs", "Dinosaurs")]
METHODS = {  # name: (score file, panel title)
    "naive": ("naive_all_margin.pt", "Naive (TracIn-style)\n1 model, full gradients"),
    "trak": ("trak_all_margin_k512.pt", "TRAK\n5 models, k = 512"),
    "trak8192": ("trak_all_margin_k8192.pt", "TRAK\n5 models, k = 8192"),
    "loo": ("loo_all.pt", "Exact leave-one-out\n1 model, one retrain per document"),
}

# Light slide surface, text ink, and a one-hue blue ramp (light = near zero).
SURFACE, INK, INK_2 = "#fcfcfb", "#0b0b0b", "#52514e"
SEQ = LinearSegmentedColormap.from_list("blue", [SURFACE, "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
plt.rcParams.update({"font.size": 13, "axes.edgecolor": INK_2, "text.color": INK,
                     "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK})


def load_data():
    run = Run(os.path.join(ROOT, "runs", "toy", "seed0"))
    tok, docs, queries = load_inputs(run.train_args())
    return tok, docs, queries


def recall_hits(tok, queries):
    """{seed: [bool per query]}: whether that model answers each query correctly (greedy decoding)."""
    qb = encode_queries(tok, queries)
    runs = sorted(r for r in os.listdir(os.path.join(ROOT, "runs", "toy")) if r.startswith("seed"))
    return {int(r[4:]): answer_logprobs(Run(os.path.join(ROOT, "runs", "toy", r)).load("cpu"), qb, "cpu")[1].tolist()
            for r in runs}


def recalled_by_all(tok, queries):
    """Queries every trained seed answers correctly."""
    hits = recall_hits(tok, queries)
    return [all(h[i] for h in hits.values()) for i in range(len(queries))]


def area_matrix(S, qids, dids, docs, picked, metric):
    """M[r, c]: share of credit that queries from topic r give to documents from topic c,
    averaged over that topic's queries. top10: share of the 10 highest-scored documents.
    share: share of all positive score mass. Also returns the share left outside the 5 topics."""
    area = torch.tensor([[a for a, _ in AREAS].index(docs[d]["area_id"]) if docs[d]["area_id"] in dict(AREAS) else -1
                         for d in dids])
    M = torch.zeros(len(AREAS), len(AREAS))
    for r, (a, _) in enumerate(AREAS):
        for qid in picked[a]:
            s = S[qids.index(qid)]
            if metric == "top10":
                w = torch.zeros_like(s).index_fill(0, s.topk(10).indices, 0.1)
            else:
                w = s.clamp_min(0) / s.clamp_min(0).sum()
            for c in range(len(AREAS)):
                M[r, c] += w[area == c].sum() / len(picked[a])
    return M, 1 - M.sum(1)


def heatmap(args):
    tok, docs, queries = load_data()
    ok = recalled_by_all(tok, queries)
    # The first 3 queries per topic (by id) that every model answers correctly: a fixed rule, not hand-picked.
    picked = {a: [q["query_id"] for q, h in zip(queries, ok) if h and q["area_id"] == a][:3] for a, _ in AREAS}
    methods = [m for m in args.methods if os.path.exists(os.path.join(SCORES, METHODS[m][0]))]
    skipped = set(args.methods) - set(methods)
    if skipped:
        print(f"skipping (no scores yet): {', '.join(sorted(skipped))}")

    mats = {}
    for m in methods:
        R = torch.load(os.path.join(SCORES, METHODS[m][0]))
        mats[m] = area_matrix(R["scores"], R["query_ids"], R["doc_ids"], docs, picked, args.metric)
    vmax = max(M.max().item() for M, _ in mats.values())

    fig, axes = plt.subplots(1, len(methods), figsize=(5.6 * len(methods) + 1.2, 5.2), squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    names = [n for _, n in AREAS]
    for ax, m in zip(axes[0], methods):
        M, outside = mats[m]
        ax.set_facecolor(SURFACE)
        mesh = ax.pcolormesh(M.numpy(), cmap=SEQ, vmin=0, vmax=vmax, edgecolors=SURFACE, linewidth=2)
        ax.invert_yaxis()
        ax.set_aspect("equal")
        for r in range(len(AREAS)):
            for c in range(len(AREAS)):
                v = M[r, c].item()
                ax.text(c + 0.5, r + 0.5, f"{v:.0%}", ha="center", va="center", fontsize=12,
                        color="white" if v > 0.55 * vmax else INK)
            ax.text(len(AREAS) + 0.12, r + 0.5, f"{outside[r].item():.0%}", ha="left", va="center",
                    fontsize=11, color=INK_2)
        ax.text(len(AREAS) + 0.12, -0.15, "other\ntopics", ha="left", va="bottom", fontsize=10, color=INK_2)
        ax.set_xticks([i + 0.5 for i in range(len(AREAS))], names, rotation=35, ha="right")
        ax.set_yticks([i + 0.5 for i in range(len(AREAS))], names if ax is axes[0][0] else [""] * len(AREAS))
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(METHODS[m][1], fontsize=13, pad=10)
        ax.set_xlabel("Topic of the credited training documents", fontsize=11, color=INK_2)
    axes[0][0].set_ylabel("Topic of the query", fontsize=11, color=INK_2)
    cbar = fig.colorbar(mesh, ax=axes[0].tolist(), shrink=0.7, pad=0.06)
    cbar.outline.set_visible(False)
    cbar.ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    what = "of the 10 top-credited documents" if args.metric == "top10" else "of all positive credit"
    cbar.set_label(f"Share {what}", color=INK_2)
    fig.suptitle("Does the credit go back to the right topic?", fontsize=16, x=0.02, ha="left", y=1.075, va="bottom")
    fig.text(0.02, 1.0, "3 queries per topic (the first 3 every model answers correctly), scored against all 1,528 "
             "training documents in 30 topics.\nRight of each row: the share that went to the other 25 topics.",
             fontsize=10, color=INK_2, va="bottom")

    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"heatmap_{args.metric}.png")
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    with open(path.replace(".png", ".json"), "w") as f:
        json.dump({"metric": args.metric, "topics": names, "queries": picked,
                   "matrices": {m: {"M": mats[m][0].tolist(), "other_topics": mats[m][1].tolist()} for m in methods}},
                  f, indent=1)
    print(f"saved {path} (+ .json with the numbers)")


def query_split(args):
    """One query: how a method's positive credit splits over documents, and over their authors."""
    _, docs, queries = load_data()
    with open(os.path.join(ROOT, "data", "authors.json")) as f:
        authors = {a["author_id"]: a for a in json.load(f)}
    q = next(x for x in queries if x["query_id"] == args.query)
    R = torch.load(os.path.join(SCORES, METHODS[args.method][0]))
    s, dids = R["scores"][R["query_ids"].index(args.query)], R["doc_ids"]
    share = s.clamp_min(0) / s.clamp_min(0).sum()  # royalty view: only positive credit is paid out
    order = share.argsort(descending=True).tolist()
    src = set(q["source_doc_ids"])
    src_authors = {docs[d]["author_id"] for d in src}

    by_author = {}
    for j, d in enumerate(dids):
        by_author[docs[d]["author_id"]] = by_author.get(docs[d]["author_id"], 0.0) + share[j].item()
    top_authors = sorted(by_author.items(), key=lambda x: -x[1])[:args.top_authors]
    rest = 1 - sum(v for _, v in top_authors)

    BLUE, GRAY = "#2a78d6", "#8c8b85"
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.2), gridspec_kw={"width_ratios": [1.7, 1]})
    fig.subplots_adjust(left=0.3, right=0.97, wspace=0.75)  # room for the document labels inside the figure
    fig.patch.set_facecolor(SURFACE)

    top = order[:args.top]
    vals = [share[j].item() for j in top]
    ax1.barh(range(len(top)), vals, height=0.72, color=[BLUE if dids[j] in src else GRAY for j in top])
    ax1.set_yticks(range(len(top)), [f"#{dids[j]}  {docs[dids[j]]['text'][:58]}{'...' if len(docs[dids[j]]['text']) > 58 else ''}"
                                     for j in top], fontsize=10)
    ax1.set_title(f"Top {len(top)} of {len(dids):,} documents", fontsize=13, loc="left")

    names = [f"{authors[a]['name']} ({authors[a]['area_id'].replace('_', ' ')})" for a, _ in top_authors] + \
            [f"other {len(by_author) - len(top_authors)} authors"]
    avals = [v for _, v in top_authors] + [rest]
    ax2.barh(range(len(names)), avals, height=0.72,
             color=[BLUE if a in src_authors else GRAY for a, _ in top_authors] + [GRAY])
    ax2.set_yticks(range(len(names)), names, fontsize=10)
    ax2.set_title("Royalty split by author", fontsize=13, loc="left")

    for ax, vs in ((ax1, vals), (ax2, avals)):
        ax.set_facecolor(SURFACE)
        ax.invert_yaxis()
        for i, v in enumerate(vs):
            ax.text(v + max(vs) * 0.015, i, f"{v:.1%}", va="center", fontsize=10, color=INK)
        ax.set_xlim(0, max(vs) * 1.18)
        ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
        ax.tick_params(length=0)
        ax.grid(axis="x", color="#e4e3df", linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.set_xlabel("Share of positive credit", fontsize=11, color=INK_2)
    handles = [matplotlib.patches.Patch(color=BLUE, label="ground-truth source / its author"),
               matplotlib.patches.Patch(color=GRAY, label="everything else")]
    ax1.legend(handles=handles, loc="lower right", frameon=False, fontsize=10)

    rank = {dids[j]: r + 1 for r, j in enumerate(order)}
    sh = share[order]
    eff = (-(share[share > 0] * share[share > 0].log()).sum()).exp().item()
    method = METHODS[args.method][1].replace("\n", ", ")
    fig.suptitle("Who gets credit for this answer?", fontsize=16, x=0.02, ha="left", y=1.12, va="bottom")
    fig.text(0.02, 1.0, f"Method: {method}\nQuery {q['query_id']}: \"{q['prompt']}\" → \"{q['answer']}\"\n" +
             "; ".join(f"source doc #{d} ranks #{rank[d]} with {share[dids.index(d)].item():.1%}" for d in sorted(src)) +
             f"; top 10 docs hold {sh[:10].sum():.0%}, top 100 hold {sh[:100].sum():.0%}; "
             f"credit is spread over an effective {eff:.0f} documents",
             fontsize=11, color=INK_2, va="bottom")

    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"query{q['query_id']}_{args.method}.png")
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    with open(path.replace(".png", ".json"), "w") as f:  # everything plotted, so the figure can be rebuilt
        json.dump({
            "method": args.method, "method_label": method, "scores_file": METHODS[args.method][0],
            "query": {k: q[k] for k in ("query_id", "form", "prompt", "answer", "source_doc_ids", "area_id")},
            "share_definition": "max(score, 0) / sum over all documents of max(score, 0)",
            "summary": {"source_rank": {d: rank[d] for d in sorted(src)},
                        "source_share": {d: share[dids.index(d)].item() for d in sorted(src)},
                        "top10_share": sh[:10].sum().item(), "top100_share": sh[:100].sum().item(),
                        "effective_n_docs": eff, "n_docs": len(dids),
                        "n_positive": int((s > 0).sum()), "n_negative": int((s < 0).sum())},
            "top_docs": [{"rank": r + 1, "doc_id": dids[j], "share": share[j].item(), "score": s[j].item(),
                          "is_source": dids[j] in src, "author_id": docs[dids[j]]["author_id"],
                          "author": authors[docs[dids[j]]["author_id"]]["name"], "text": docs[dids[j]]["text"]}
                         for r, j in enumerate(top)],
            "authors": [{"author_id": a, "author": authors[a]["name"], "area_id": authors[a]["area_id"],
                         "share": v, "is_source_author": a in src_authors} for a, v in top_authors]
                       + [{"author": f"other {len(by_author) - len(top_authors)} authors", "share": rest}],
        }, f, indent=1, ensure_ascii=False)
    print(f"saved {path} (+ .json with the numbers)")


SEED_METHODS = [  # label, per-seed score file, color (validated categorical slots 1-3), marker
    ("Naive, raw dot product", "naive_all_margin_seed{s}.pt", "#2a78d6", "o"),
    ("Naive, cosine", "naive_all_margin_cosine_seed{s}.pt", "#eb6834", "s"),
    ("TRAK, k = 512", "trak_all_margin_k512_seed{s}.pt", "#1baf7a", "D"),
]
T975 = {3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262}  # t(0.975, n-1), keyed by n


def source_ranks(R, by_id, qids):
    """Rank (1 = top) of the best-ranked ground-truth source document, for each query id."""
    col = {d: j for j, d in enumerate(R["doc_ids"])}
    out = []
    for qid in qids:
        ranks = R["scores"][R["query_ids"].index(qid)].argsort(descending=True).argsort()
        out.append(int(min(ranks[col[d]] for d in by_id[qid]["source_doc_ids"])) + 1)
    return torch.tensor(out, dtype=torch.float)


def seeds(args):
    """Top-1 and MRR of each method on each of 5 independently trained models, with 95% t-intervals."""
    tok, docs, queries = load_data()
    hits = recall_hits(tok, queries)
    by_id = {q["query_id"]: q for q in queries}
    forms = ["cloze", "question"]

    def qset_for(seed):
        """per-seed: queries this model answers correctly. all-seeds: queries all models answer correctly
        (identical for every seed, but only 5 question queries qualify)."""
        if args.query_set == "all-seeds":
            ok = [all(hits[s][i] for s in args.seeds) for i in range(len(queries))]
        else:
            ok = hits[seed]
        return {f: [q["query_id"] for q, h in zip(queries, ok) if h and q["form"] == f] for f in forms}

    qsets = {s: qset_for(s) for s in args.seeds}
    metrics = {"top1": lambda r: (r == 1).float().mean().item(), "mrr": lambda r: (1 / r).mean().item()}

    def evaluate(R, qset):
        return {f: {m: fn(source_ranks(R, by_id, qset[f])) for m, fn in metrics.items()} for f in forms}

    per_seed = {label: [evaluate(torch.load(os.path.join(SCORES, pat.format(s=s))), qsets[s]) for s in args.seeds]
                for label, pat, _, _ in SEED_METHODS}
    loo_path = os.path.join(SCORES, "loo_all.pt")
    loo = evaluate(torch.load(loo_path), qset_for(0)) if os.path.exists(loo_path) else None  # LOO exists for seed 0

    n = len(args.seeds)
    summary = {}
    for label, runs in per_seed.items():
        summary[label] = {}
        for f in forms:
            summary[label][f] = {}
            for m in metrics:
                v = torch.tensor([r[f][m] for r in runs])
                half = T975[n] * v.std(unbiased=True).item() / math.sqrt(n)
                summary[label][f][m] = {"seeds": v.tolist(), "mean": v.mean().item(), "ci95_half_width": half}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6))
    fig.patch.set_facecolor(SURFACE)
    offsets = [-0.3, 0.0, 0.3]
    for ax, (m, title) in zip(axes, [("top1", "Top-1 accuracy"), ("mrr", "Mean reciprocal rank")]):
        ax.set_facecolor(SURFACE)
        for g, f in enumerate(forms):
            for (label, _, color, marker), dx in zip(SEED_METHODS, offsets):
                st = summary[label][f][m]
                x = g + dx
                ax.scatter([x - 0.075 + 0.015 * j for j in range(n)], st["seeds"], s=16, color=color, alpha=0.5,
                           linewidths=0, zorder=2)
                ax.errorbar(x + 0.03, st["mean"], yerr=st["ci95_half_width"], fmt=marker, color=color, markersize=8,
                            capsize=5, elinewidth=2, capthick=2, markeredgecolor=SURFACE, markeredgewidth=1.2, zorder=3)
                ax.text(x + 0.08, st["mean"], f"{st['mean']:.2f}", va="center", fontsize=10, color=INK)
            if loo:
                ax.hlines(loo[f][m], g - 0.45, g + 0.45, colors=INK_2, linestyles="--", linewidth=1.2, zorder=1)
        ns = {f: sorted({len(qsets[s][f]) for s in args.seeds}) for f in forms}
        ax.set_xticks([0, 1], [f"{f.capitalize()} queries\n(n = {ns[f][0]}" + (f"-{ns[f][-1]} per seed)" if len(ns[f]) > 1 else ")")
                               for f in forms])
        ax.set_xlim(-0.55, 1.55)
        ax.set_ylim(0, 1.05)
        ax.set_title(title, fontsize=13, loc="left")
        ax.tick_params(length=0)
        ax.grid(axis="y", color="#e4e3df", linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
    handles = [matplotlib.lines.Line2D([], [], color=c, marker=mk, linestyle="", markersize=8, label=lb)
               for lb, _, c, mk in SEED_METHODS]
    handles.append(matplotlib.lines.Line2D([], [], color=INK_2, marker="o", linestyle="", markersize=4, alpha=0.5,
                                           label="individual seeds"))
    if loo:
        handles.append(matplotlib.lines.Line2D([], [], color=INK_2, linestyle="--", label="exact LOO (seed 0 only)"))
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=len(handles), frameon=False, fontsize=10)
    fig.suptitle("How often is the true source document ranked first?", fontsize=16, x=0.02, ha="left", y=1.08, va="bottom")
    fig.text(0.02, 1.0, f"Each method run separately on {n} independently trained models (seeds "
             f"{', '.join(map(str, args.seeds))}); marker = mean, bar = 95% t-interval (t = {T975[n]:.3f} x SD / sqrt({n})).\n"
             + ("Queries: each model is scored on the queries it answers correctly." if args.query_set == "per-seed" else
              "Queries: those all 5 models answer correctly, so every seed and method is scored on the same set."),
             fontsize=10, color=INK_2, va="bottom")

    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"seeds_ci_{args.query_set}.png")
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    with open(path.replace(".png", ".json"), "w") as f:
        json.dump({"seeds": args.seeds, "ci": f"95% t-interval, t = {T975[n]}, n = {n}", "query_set": args.query_set,
                   "query_ids_per_seed": qsets,
                   "methods": {lb: pat for lb, pat, _, _ in SEED_METHODS}, "summary": summary,
                   "loo_seed0": loo}, f, indent=1)
    print(f"saved {path} (+ .json with the numbers)")
    for label in summary:
        print(label + ": " + "; ".join(f"{f} top-1 {summary[label][f]['top1']['mean']:.2f} +/- {summary[label][f]['top1']['ci95_half_width']:.2f}, "
                                       f"MRR {summary[label][f]['mrr']['mean']:.2f} +/- {summary[label][f]['mrr']['ci95_half_width']:.2f}"
                                       for f in forms))
    if loo:
        print("LOO (seed 0): " + "; ".join(f"{f} top-1 {loo[f]['top1']:.2f}, MRR {loo[f]['mrr']:.2f}" for f in forms))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    h = sub.add_parser("heatmap")
    h.add_argument("--metric", choices=["top10", "share"], default="top10")
    h.add_argument("--methods", nargs="+", default=["naive", "trak", "loo"], choices=list(METHODS))
    qs = sub.add_parser("query")
    qs.add_argument("--query", type=int, required=True, help="query_id")
    qs.add_argument("--method", default="naive", choices=list(METHODS))
    qs.add_argument("--top", type=int, default=12, help="documents to show")
    qs.add_argument("--top-authors", type=int, default=6)
    sd = sub.add_parser("seeds")
    sd.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    sd.add_argument("--query-set", choices=["per-seed", "all-seeds"], default="per-seed")
    args = ap.parse_args()
    {"heatmap": heatmap, "query": query_split, "seeds": seeds}[args.cmd](args)


if __name__ == "__main__":
    main()
