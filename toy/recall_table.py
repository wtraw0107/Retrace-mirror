"""Answer accuracy on every fact in the training data, asked as a cloze prompt and as a question.

    python toy/recall_table.py                                        # v2: runs/toy, data/, epoch 50
    python toy/recall_table.py --runs runs/toy_v3 --data data_v3 --by-copies

Prints markdown tables: mean over the seeds in --runs (every seed*/ run) with a 95% t-interval.
A fact counts as answered when greedy decoding reproduces its whole answer (teacher-forced argmax,
as in train.py). Train entities have their question in the training data; test entities never do.
"""

import argparse
import glob
import json
import math
import os
import statistics
import sys
from collections import Counter, defaultdict

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "dataset"))
from attribute import Run  # noqa: E402
from common import cloze_prompt  # noqa: E402
from train import answer_logprobs, encode_queries, load_inputs  # noqa: E402

T975 = {3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262}  # t(0.975, n-1), keyed by n
ROWS = [("question", "train", "**Question, exact wording seen in training**"),
        ("cloze", "train", "Cloze, train entities"),
        ("cloze", "test", "Cloze, test entities"),
        ("question", "test", "**Question, wording never seen**")]


def hits(run, batch, device, chunk=512):
    """Whether the model answers each item of an encode_queries batch."""
    model = run.load(device)
    return torch.cat([answer_logprobs(model, tuple(t[i:i + chunk] for t in batch), device)[1].cpu()
                      for i in range(0, len(batch[3]), chunk)]).tolist()


def mean_ci(per_seed):
    n = len(per_seed)
    half = T975[n] * statistics.stdev(per_seed) / math.sqrt(n) if n in T975 else float("nan")
    return f"{statistics.mean(per_seed):.1%} ± {half * 100:.1f}"


def baselines(facts):
    """Answer-only baselines on test entities, per question type (area, relation): always the most
    common answer of that type, or an answer drawn at random from that type's facts."""
    by_type = defaultdict(Counter)
    for f in facts:
        by_type[f["area_id"], f["relation_id"]][f["answer"]] += 1
    test = [(f, by_type[f["area_id"], f["relation_id"]]) for f in facts if f["qa_split"] == "test"]
    common = statistics.mean(f["answer"] == c.most_common(1)[0][0] for f, c in test)
    rand = statistics.mean(c[f["answer"]] / sum(c.values()) for f, c in test)
    return common, rand


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default=os.path.join(ROOT, "runs", "toy"), help="directory holding the seed*/ runs")
    ap.add_argument("--data", default=os.path.join(ROOT, "data"), help="dataset directory the runs were trained on")
    ap.add_argument("--epochs", type=int, nargs="+", default=[50])
    ap.add_argument("--by-copies", action="store_true", help="also split test entities by k, the number of copies")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(os.path.join(args.data, "facts.jsonl")) as fh:
        facts = [json.loads(line) for line in fh]
    run_dirs = sorted(glob.glob(os.path.join(args.runs, "seed*")))
    tok, _, _ = load_inputs(Run(run_dirs[0]).train_args())
    # One cloze and one question per fact, in the same formats as queries.jsonl.
    items = [(form, f) for f in facts for form in ("cloze", "question")]
    batch = encode_queries(tok, [{"prompt": cloze_prompt(f) if form == "cloze" else f"Question: {f['question']} Answer:",
                                  "answer": f["answer"]} for form, f in items])

    for epoch in args.epochs:
        seed_hits = [hits(Run(d, epoch), batch, device) for d in run_dirs]

        def acc(keep):
            idx = [i for i, (form, f) in enumerate(items) if keep(form, f)]
            return len(idx), mean_ci([sum(h[i] for i in idx) / len(idx) for h in seed_hits])

        print(f"\n{args.runs}, ckpt_epoch{epoch:04d}, {len(run_dirs)} seeds\n")
        print("| Prompt | Accuracy |\n|---|---|")
        for form, split, label in ROWS:
            n, a = acc(lambda fm, f: fm == form and f["qa_split"] == split)
            bold = "**" if label.startswith("**") else ""
            print(f"| {label} ({n} facts) | {bold}{a}{bold} |")
        common, rand = baselines(facts)
        print(f"| Baseline: always answer the most common answer for that question type | {common:.1%} |")
        print(f"| Baseline: random answer of the same question type | {rand:.1%} |")

        if args.by_copies:
            print("\n| k (copies) | Test facts | Cloze, test entities | Question, wording never seen |\n|---|---|---|---|")
            for k in sorted({f.get("n_copies", 1) for f in facts}):
                cells = [acc(lambda fm, f: fm == form and f["qa_split"] == "test" and f.get("n_copies", 1) == k)
                         for form in ("cloze", "question")]
                print(f"| {k} | {cells[0][0]} | {cells[0][1]} | {cells[1][1]} |")


if __name__ == "__main__":
    main()
