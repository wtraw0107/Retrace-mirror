"""Attribute a model's answer to its training documents (Section 5.3).

Every method gives a score s(q, i) for query q and training document i; positive means
document i supports the answer.

Output functions:
    query     f(q) = log(p / (1 - p)), p = P(answer | prompt)   (TRAK's classification margin)
    document  --doc-output margin (default): f_i = sum over its tokens of the margin
                  logit(token) - logsumexp(other logits), and Q_ii = 1
              --doc-output logprob (Section 5.3 as written): f_i = log(p_i / (1 - p_i)),
                  p_i = P(document), Q_ii = 1 - p_i; then (1 - p_i) grad f_i = -grad L_i

Why margin is the default: the model memorizes every token after a document's first word
(p = 1.000), and a token's loss gradient is (1 - p) times its margin gradient. With logprob,
~99% of a document's gradient is "guess its first word" and the facts vanish from it; on
this model naive and TRAK then rank the source document near chance. Margins keep every
token's contribution. The price: Q is dropped, so TRAK is no longer the one-step Newton
estimate of leave-one-out. Exact LOO is the referee for whether the ranking still holds.

    naive  s = grad f(q) . g_i                         full gradients (4.4M dims), TracIn-style,
                                                      g_i = grad f_i * Q_ii
    trak   s = phi(q)^T (Phi^T Phi + lam I)^-1 phi_i * Q_ii
                                                      phi = P grad f, P = random projection to k
                                                      dims. Averaged over models, with Q averaged
                                                      separately (as the traker library does).
    loo    s = f(q; theta) - f(q; theta without doc i)  exact: retrain once per document

Naive is TRAK with P = identity and without (Phi^T Phi + lam I)^-1.

Modes:   pair   one query, one document     --query Q --doc D
         query  one query, every document   --query Q
         all    every query x every document
Models:  --runs (training run dirs) at --epoch (default: each run's final checkpoint).
         Default runs: trak averages every runs/toy/seed*; naive and loo use runs/toy/seed0.

Examples:
    python toy/attribute.py --method trak --mode query --query 0
    python toy/attribute.py --method naive --mode all
    python toy/attribute.py --method loo --mode pair --query 0 --doc 22
"""

import argparse
import glob
import hashlib
import json
import math
import os
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # deterministic cuBLAS, needed for exact LOO

import torch
import torch.multiprocessing as mp
from transformers import GPTNeoXForCausalLM

from train import answer_logprobs, encode_queries, load_inputs, train

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EPS = 1e-7  # floor for 1 - p: an fp32 log-prob can't resolve p any closer to 1


def logit_from_logprob(lp):
    """f = log(p / (1 - p)) and 1 - p, from log p."""
    one_minus_p = (-torch.expm1(lp)).clamp_min(EPS)
    return lp - one_minus_p.log(), one_minus_p


class Run:
    """One trained model: a training run directory at one checkpoint epoch."""

    def __init__(self, run_dir, epoch=None):
        self.dir = os.path.abspath(run_dir)
        with open(os.path.join(self.dir, "args.json")) as f:
            self.args = json.load(f)
        self.epoch = epoch or self.args["epochs"]
        self.ckpt = os.path.join(self.dir, f"ckpt_epoch{self.epoch:04d}")
        assert os.path.isdir(self.ckpt), f"no checkpoint {self.ckpt}"
        self.name = "_".join(self.dir.split(os.sep)[-2:]) + f"_ep{self.epoch}"
        # Changes whenever the run is retrained, so stale caches are detected.
        self.stamp = os.path.getmtime(os.path.join(self.ckpt, "model.safetensors"))

    def load(self, device):
        # Eager, like training: from_pretrained defaults to SDPA, whose rounding differs enough
        # that a retrained model no longer matches the checkpoint exactly.
        return GPTNeoXForCausalLM.from_pretrained(self.ckpt, attn_implementation="eager").to(device).eval()

    def train_args(self, exclude_doc_id=None):
        args = argparse.Namespace(**{k: v for k, v in self.args.items() if k != "n_params"})
        args.exclude_doc_ids = "" if exclude_doc_id is None else str(exclude_doc_id)
        return args


def encode_examples(tok, queries, docs):
    """Each example is (ids, n_ctx): ids[n_ctx:] is the segment whose probability p we score,
    given ids[:n_ctx]. Queries: [BOS] prompt | answer. Documents: [BOS] | text [EOS]."""
    q = []
    for x in queries:
        prompt = [tok.bos_token_id] + tok.encode(x["prompt"])
        q.append((torch.tensor(prompt + tok.encode(x["answer"])), len(prompt)))
    d = [(torch.tensor([tok.bos_token_id] + tok.encode(x["text"]) + [tok.eos_token_id]), 1) for x in docs]
    return q, d


def value_and_grad(model, params, ids, n_ctx, kind):
    """log p of the segment ids[n_ctx:] given ids[:n_ctx], and the gradient (one flat vector) of
    either log p (kind="logprob") or the sum of its per-token margins (kind="margin")."""
    z = model(input_ids=ids[None]).logits[0, n_ctx - 1:-1]  # position t predicts token t+1
    tgt = ids[n_ctx:, None]
    lp = z.log_softmax(-1).gather(1, tgt).sum()
    if kind == "logprob":
        value = lp
    else:
        value = (z.gather(1, tgt).squeeze(1) - z.scatter(1, tgt, float("-inf")).logsumexp(1)).sum()
    grads = torch.autograd.grad(value, params)
    return lp.detach(), torch.cat([g.flatten() for g in grads])


def grad_blocks(model, examples, device, kind="logprob", block=128):
    """Yield (log p, gradient) for consecutive blocks of examples, shapes (B,) and (B, n_params).

    One backward pass per example. This plain loop is the simple baseline; batching it with
    torch.func.vmap is possible (it needs equal-length, unmasked sequences in this model).
    """
    params = list(model.parameters())
    for start in range(0, len(examples), block):
        out = [value_and_grad(model, params, ids.to(device), n, kind) for ids, n in examples[start:start + block]]
        yield torch.stack([lp for lp, _ in out]), torch.stack([g for _, g in out])


def query_grads(model, examples, device):
    """grad f(q) = grad log p / (1 - p), for f = log(p / (1 - p))."""
    for lp, G in grad_blocks(model, examples, device):
        yield G / logit_from_logprob(lp)[1][:, None]


def doc_grads(model, examples, device, doc_output):
    """(grad f_i, Q_ii) per block of documents; see the module docstring for the two options."""
    for lp, G in grad_blocks(model, examples, device, kind=doc_output):
        if doc_output == "margin":
            yield G, torch.ones_like(lp)
        else:
            one_minus_p = logit_from_logprob(lp)[1]
            yield G / one_minus_p[:, None], one_minus_p


class Projector:
    """Gaussian random projection P (n_params x k), entries N(0, 1/k), so <Px, Py> ~ <x, y>.

    P is regenerated chunk by chunk from the seed rather than stored: at k = 512 the full
    matrix would be 4.4M x 512 floats = 9 GB.
    """

    def __init__(self, n_params, k, seed, chunk=2 ** 18):
        self.n, self.k, self.seed, self.chunk = n_params, k, seed, chunk

    def __call__(self, G):
        out = torch.zeros(G.shape[0], self.k, device=G.device)
        for c, start in enumerate(range(0, self.n, self.chunk)):
            gen = torch.Generator(device=G.device).manual_seed(self.seed * 10_000 + c)
            P = torch.randn(min(self.chunk, self.n - start), self.k, generator=gen, device=G.device)
            out += G[:, start:start + self.chunk] @ P
        return out / math.sqrt(self.k)


def fingerprint(docs):
    return hashlib.sha1("\n".join(d["text"] for d in docs).encode()).hexdigest()[:12]


# ---------------------------------------------------------------- naive


def naive_scores(runs, q_ex, d_ex, device, doc_output, **_):
    total = 0
    for run in runs:
        model = run.load(device)
        Gq = torch.cat(list(query_grads(model, q_ex, device)))
        # With logprob, G * Q = grad log P(doc) = -grad L_i, so this is grad f(q) . (-grad L_i).
        total = total + torch.cat([Gq @ (G * Q[:, None]).T for G, Q in doc_grads(model, d_ex, device, doc_output)], dim=1)
    return {"scores": total / len(runs)}


# ---------------------------------------------------------------- TRAK


def trak_features(run, model, d_ex, proj, doc_output, fp, cache_dir, device):
    """Phi (n_docs x k) and Q (n_docs) for one model over the whole training set, cached.

    TRAK featurizes the training set once; after that each query costs one backward pass.
    """
    path = os.path.join(cache_dir, f"{run.name}_{doc_output}_k{proj.k}_seed{proj.seed}.pt")
    if os.path.exists(path):
        c = torch.load(path)
        if c["stamp"] == run.stamp and c["docs"] == fp:
            return c["Phi"].to(device), c["Q"].to(device)
    Phi, Q = [], []
    for G, Qb in doc_grads(model, d_ex, device, doc_output):
        Phi.append(proj(G))  # phi_i = P grad f_i
        Q.append(Qb)
    Phi, Q = torch.cat(Phi), torch.cat(Q)
    os.makedirs(cache_dir, exist_ok=True)
    torch.save({"Phi": Phi.cpu(), "Q": Q.cpu(), "stamp": run.stamp, "docs": fp}, path)
    return Phi, Q


def trak_scores(runs, q_ex, d_ex, device, d_idx, k, lam, proj_seed, doc_output, fp, cache_dir, **_):
    total, Q_total = 0, 0
    n_params = sum(p.numel() for p in runs[0].load("cpu").parameters())
    for run in runs:
        model = run.load(device)
        proj = Projector(n_params, k, seed=proj_seed + run.args["seed"])  # independent P per model
        # The kernel Phi^T Phi sums over ALL training documents, even if only a subset is scored.
        Phi, Q = trak_features(run, model, d_ex, proj, doc_output, fp, cache_dir, device)
        phi_q = torch.cat([proj(G) for G in query_grads(model, q_ex, device)])
        K = (Phi.T @ Phi).double()
        K += lam * K.trace() / k * torch.eye(k, dtype=K.dtype, device=device)  # lam is relative to the mean eigenvalue
        Phi_s = Phi[d_idx].double()
        total = total + (phi_q.double() @ torch.linalg.solve(K, Phi_s.T)).float()
        Q_total = Q_total + Q[d_idx]
    return {"scores": (total / len(runs)) * (Q_total / len(runs))[None, :]}


# ---------------------------------------------------------------- exact LOO

_worker = {}


def _init_worker(gpus):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    _worker["device"] = f"cuda:{gpus.get()}" if torch.cuda.is_available() else "cpu"


def query_logits(model, query_batch, device):
    lp, _ = answer_logprobs(model, query_batch, device)
    return logit_from_logprob(lp)[0]


def _loo_task(task):
    """Retrain one run without one document (None = without nothing) and cache f(q) for every query."""
    run_dir, epoch, doc_id, path = task
    run, device = Run(run_dir, epoch), _worker["device"]
    if run.dir not in _worker:
        tok, docs, queries = load_inputs(run.train_args())
        _worker[run.dir] = (tok, docs, encode_queries(tok, queries))
    tok, docs, query_batch = _worker[run.dir]
    t0 = time.time()
    model = train(run.train_args(doc_id), tok, docs, device, stop_after=run.epoch).eval()
    f = query_logits(model, query_batch, device)
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump({"doc_id": doc_id, "stamp": run.stamp, "f": f.tolist(), "train_s": round(time.time() - t0, 1)}, fh)
    os.replace(tmp, path)  # atomic, so an interrupted run never leaves a half-written cache file
    return path


def loo_scores(runs, query_batch, d_ids, q_idx, device, workers, cache_dir, **_):
    """f(q; theta) - f(q; theta without doc i), one retraining per (run, document), cached and resumable.

    Each run is also retrained once without excluding anything: with deterministic kernels it
    must reproduce the saved checkpoint exactly, otherwise LOO differences would include noise.
    """
    def path(run, doc_id):
        return os.path.join(cache_dir, run.name, "base_repro.json" if doc_id is None else f"doc{doc_id:04d}.json")

    def cached(p, run):
        if not os.path.exists(p):
            return None
        with open(p) as fh:
            c = json.load(fh)
        return c if c["stamp"] == run.stamp else None

    tasks = []
    for run in runs:
        os.makedirs(os.path.join(cache_dir, run.name), exist_ok=True)
        for doc_id in [None] + list(d_ids):
            if cached(path(run, doc_id), run) is None:
                tasks.append((run.dir, run.epoch, doc_id, path(run, doc_id)))
    if tasks:
        n_gpus = max(1, torch.cuda.device_count())
        workers = min(workers, len(tasks))
        print(f"loo: {len(tasks)} retrainings to run ({workers} workers on {n_gpus} GPU(s)); "
              f"results are cached in {cache_dir}, so an interrupted run resumes")
        ctx = mp.get_context("spawn")
        gpus = ctx.Queue()
        for w in range(workers):
            gpus.put(w % n_gpus)
        t0 = time.time()
        with ctx.Pool(workers, initializer=_init_worker, initargs=(gpus,)) as pool:
            for n, _ in enumerate(pool.imap_unordered(_loo_task, tasks), 1):
                if n % max(1, len(tasks) // 20) == 0 or n == len(tasks):
                    el = time.time() - t0
                    print(f"  {n}/{len(tasks)} done, {el / 60:.1f} min elapsed, ~{el / n * (len(tasks) - n) / 60:.1f} min left")

    f_base, f_loo = [], []
    for run in runs:
        f = query_logits(run.load(device), query_batch, device).cpu()
        repro = torch.tensor(cached(path(run, None), run)["f"])
        err = (repro - f).abs().max().item()
        print(f"loo: {run.name} retrained without exclusions reproduces the checkpoint to max |delta f| = {err:.2e}")
        f_base.append(f[q_idx])
        f_loo.append(torch.stack([torch.tensor(cached(path(run, d), run)["f"])[q_idx] for d in d_ids], dim=1))
    f_base, f_loo = torch.stack(f_base), torch.stack(f_loo)  # (runs, n_q), (runs, n_q, n_docs)
    return {"scores": (f_base[:, :, None] - f_loo).mean(0), "f_base": f_base, "f_loo": f_loo}


# ---------------------------------------------------------------- reporting


def summarize(scores, queries, q_idx, doc_ids, recalled):
    """Top-1 accuracy and MRR against each query's ground-truth source documents."""
    col = {d: j for j, d in enumerate(doc_ids)}
    rows = []
    for r, qi in enumerate(q_idx):
        src = [col[d] for d in queries[qi]["source_doc_ids"] if d in col]
        if src:
            ranks = scores[r].argsort(descending=True).argsort()  # rank of each document, 0 = top
            rows.append((queries[qi].get("form", "all"), bool(recalled[qi]), int(ranks[src].min()) + 1))
    print(f"{'queries':<22}{'n':>5}{'top-1':>8}{'MRR':>8}")
    for form in sorted({r[0] for r in rows}) + ["all forms"]:
        for only_recalled in (False, True):
            sel = [rk for f, h, rk in rows if (form == "all forms" or f == form) and (h or not only_recalled)]
            if sel:
                label = f"{form}{', recalled' if only_recalled else ''}"
                top1 = sum(rk == 1 for rk in sel) / len(sel)
                mrr = sum(1 / rk for rk in sel) / len(sel)
                print(f"{label:<22}{len(sel):>5}{top1:>8.2f}{mrr:>8.2f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", choices=["naive", "trak", "loo"], required=True)
    ap.add_argument("--mode", choices=["pair", "query", "all"], required=True)
    ap.add_argument("--query", type=int, help="query_id (pair and query modes)")
    ap.add_argument("--doc", type=int, help="doc_id (pair mode)")
    ap.add_argument("--docs", default="", help="comma-separated doc_ids to score instead of all (query and all modes)")
    ap.add_argument("--runs", nargs="+", help="training run dirs (default: trak all runs/toy/seed*, others seed0)")
    ap.add_argument("--epoch", type=int, help="checkpoint epoch (default: each run's final epoch)")
    ap.add_argument("--doc-output", choices=["margin", "logprob"], default="margin",
                    help="naive/trak: document output function (see the docstring for why margin is the default)")
    ap.add_argument("--k", type=int, default=512, help="trak: projection dimension")
    ap.add_argument("--lam", type=float, default=1e-3, help="trak: ridge, relative to the mean eigenvalue of Phi^T Phi")
    ap.add_argument("--proj-seed", type=int, default=0, help="trak: projection seed (offset by each run's seed)")
    ap.add_argument("--workers", type=int, default=32, help="loo: parallel retraining processes")
    ap.add_argument("--top", type=int, default=10, help="query mode: documents to print")
    ap.add_argument("--out", default=os.path.join(ROOT, "runs", "attribution"))
    args = ap.parse_args()
    if args.mode in ("pair", "query"):
        assert args.query is not None, f"--mode {args.mode} needs --query"
    if args.mode == "pair":
        assert args.doc is not None, "--mode pair needs --doc"

    torch.use_deterministic_algorithms(True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    default_runs = sorted(glob.glob(os.path.join(ROOT, "runs", "toy", "seed*")))
    run_dirs = args.runs or (default_runs if args.method == "trak" else default_runs[:1])
    runs = [Run(d, args.epoch) for d in run_dirs]
    for key in ("docs", "queries", "tokenizer"):
        assert len({r.args[key] for r in runs}) == 1, f"runs disagree on {key}"
    tok, docs, queries = load_inputs(runs[0].train_args())

    qpos = {q["query_id"]: i for i, q in enumerate(queries)}
    dpos = {d["doc_id"]: j for j, d in enumerate(docs)}
    q_idx = [qpos[args.query]] if args.mode != "all" else list(range(len(queries)))
    if args.mode == "pair":
        d_ids = [args.doc]
    else:
        d_ids = [int(x) for x in args.docs.split(",") if x] or [d["doc_id"] for d in docs]
    d_idx = [dpos[d] for d in d_ids]
    print(f"{args.method} / {args.mode}: {len(q_idx)} queries x {len(d_idx)} docs, models: {', '.join(r.name for r in runs)}")

    q_ex, d_ex = encode_examples(tok, queries, docs)
    query_batch = encode_queries(tok, queries)
    cache_dir = os.path.join(args.out, "cache", args.method)
    t0 = time.time()
    if args.method == "naive":
        res = naive_scores(runs, [q_ex[i] for i in q_idx], [d_ex[j] for j in d_idx], device, doc_output=args.doc_output)
    elif args.method == "trak":
        res = trak_scores(runs, [q_ex[i] for i in q_idx], d_ex, device, d_idx=d_idx, k=args.k, lam=args.lam,
                          proj_seed=args.proj_seed, doc_output=args.doc_output, fp=fingerprint(docs),
                          cache_dir=cache_dir)
    else:
        res = loo_scores(runs, query_batch, d_ids, q_idx, device, workers=args.workers, cache_dir=cache_dir)
    scores = res["scores"].cpu()
    elapsed = time.time() - t0
    print(f"time: {elapsed:.1f} s")

    if args.mode == "pair":
        q, d = queries[q_idx[0]], docs[d_idx[0]]
        src = d["doc_id"] in q["source_doc_ids"]
        print(f"query {q['query_id']} ({q.get('form', '')}): {q['prompt']!r} -> {q['answer']!r}")
        print(f"doc {d['doc_id']} by {d.get('author_id', '?')}{' (ground-truth source)' if src else ''}: {d['text']!r}")
        print(f"score: {scores[0, 0].item():+.4g}")
        if args.method == "loo":
            for r, run in enumerate(runs):
                fb, fl = res["f_base"][r, 0].item(), res["f_loo"][r, 0, 0].item()
                pb, pl = 1 / (1 + math.exp(-fb)), 1 / (1 + math.exp(-fl))
                print(f"  {run.name}: f {fb:+.3f} -> {fl:+.3f} without the doc, P(answer) {pb:.3f} -> {pl:.3f}")
        return

    os.makedirs(args.out, exist_ok=True)
    name = f"{args.method}_{args.mode}" + (f"_q{args.query}" if args.mode == "query" else "")
    if args.method != "loo":
        name += f"_{args.doc_output}" + (f"_k{args.k}" if args.method == "trak" else "")
    out_path = os.path.join(args.out, name + ".pt")
    torch.save({"method": args.method, "scores": scores, "query_ids": [queries[i]["query_id"] for i in q_idx],
                "doc_ids": d_ids, "runs": [r.name for r in runs], "k": args.k, "lam": args.lam,
                "time_s": elapsed}, out_path)
    print(f"saved {tuple(scores.shape)} scores -> {out_path}")

    if args.mode == "query":
        q = queries[q_idx[0]]
        print(f"query {q['query_id']} ({q.get('form', '')}): {q['prompt']!r} -> {q['answer']!r}, "
              f"sources {q['source_doc_ids']} (* below)")
        order = scores[0].argsort(descending=True)
        rank = {d_ids[j]: r + 1 for r, j in enumerate(order.tolist())}
        print(f"source ranks: {', '.join(f'doc {s}: #{rank[s]}' for s in q['source_doc_ids'] if s in rank)}")
        for r, j in enumerate(order[:args.top].tolist(), 1):
            d = docs[d_idx[j]]
            mark = "*" if d["doc_id"] in q["source_doc_ids"] else " "
            print(f"{r:>3} {scores[0, j].item():+10.3e} doc {d['doc_id']:>4}{mark} {d.get('author_id', ''):<5} {d['text'][:80]}")
    else:
        hits = torch.stack([answer_logprobs(r.load(device), query_batch, device)[1].cpu() for r in runs]).float()
        summarize(scores, queries, q_idx, d_ids, recalled=(hits.mean(0) >= 0.5).tolist())


if __name__ == "__main__":
    main()
