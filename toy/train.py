"""Train one toy model from scratch on the synthetic corpus.

Input format (written by the Section 5.2 data generator):
    docs.jsonl     one {"doc_id": int, "text": str, ...} per line
    queries.jsonl  one {"query_id": int, "prompt": str, "answer": str, ...} per line
The optional query field "form" (e.g. cloze / question) splits recall by form.
Other extra fields (author, structure type, ...) are ignored here.

Design choices that matter for attribution later:
- One training example = one document: [BOS] doc [EOS], right-padded, never
  packed together with other documents. A per-example gradient is then a
  per-document gradient, which is the unit TRAK attributes to.
- Loss is plain next-token cross-entropy over every non-pad token.
- fp32, no dropout, deterministic kernels, so gradients are reproducible.
- Fixed batches: each epoch shuffles *all* documents and cuts the order into batches
  before excluded documents are dropped. A leave-one-out run therefore sees exactly the
  same batches as the full run, minus one document in one batch, so the difference
  between the two models is the effect of that document, not of a different shuffle.
- Fixed schedule (no early stopping), with checkpoints saved along the way, so
  TRAK can average over checkpoints as well as over seeds, and runs stay comparable.
"""

import argparse
import json
import math
import os
import random
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # required for deterministic cuBLAS

import numpy as np
import torch
from transformers import PreTrainedTokenizerFast

from model import build_config, build_model, param_breakdown
from tokenizer import load_jsonl


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def encode_docs(tok, docs, context_length):
    examples = []
    for d in docs:
        ids = [tok.bos_token_id] + tok.encode(d["text"]) + [tok.eos_token_id]
        assert len(ids) <= context_length, f"doc {d['doc_id']}: {len(ids)} tokens > context {context_length}"
        examples.append(ids)
    return examples


def pad_batch(seqs, pad_id):
    n = max(len(s) for s in seqs)
    input_ids = torch.full((len(seqs), n), pad_id)
    attention_mask = torch.zeros((len(seqs), n), dtype=torch.long)
    for i, s in enumerate(seqs):
        input_ids[i, :len(s)] = torch.tensor(s)
        attention_mask[i, :len(s)] = 1
    return input_ids, attention_mask


def collate(seqs, pad_id):
    input_ids, attention_mask = pad_batch(seqs, pad_id)
    labels = input_ids.masked_fill(attention_mask == 0, -100)  # -100 = ignored by the loss
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def encode_queries(tok, queries):
    """[BOS] prompt answer, right-padded; answer_mask marks the answer tokens."""
    prompts = [[tok.bos_token_id] + tok.encode(q["prompt"]) for q in queries]
    answers = [tok.encode(q["answer"]) for q in queries]
    input_ids, attention_mask = pad_batch([p + a for p, a in zip(prompts, answers)], tok.pad_token_id)
    answer_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    for i, (p, a) in enumerate(zip(prompts, answers)):
        answer_mask[i, len(p):len(p) + len(a)] = True
    forms = [q.get("form", "all") for q in queries]
    return input_ids, attention_mask, answer_mask, forms


@torch.no_grad()
def answer_logprobs(model, query_batch, device):
    """Per query: log P(answer | prompt), and whether greedy decoding reproduces the answer.

    Teacher forcing: a query is a hit when each answer token is the argmax given the prompt
    and the previous answer tokens, which is exactly when greedy decoding outputs the answer.
    """
    input_ids, attention_mask, answer_mask, _ = query_batch
    input_ids, attention_mask, answer_mask = input_ids.to(device), attention_mask.to(device), answer_mask.to(device)
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[:, :-1]
    targets, mask = input_ids[:, 1:], answer_mask[:, 1:]  # logits at position t predict token t+1
    hit = ((logits.argmax(-1) == targets) | ~mask).all(1)
    logp = logits.log_softmax(-1).gather(2, targets[..., None]).squeeze(2)
    return (logp * mask).sum(1), hit


def answer_recall(model, query_batch, device):
    """Recall and mean P(answer) over all queries, plus recall per query form."""
    logp, hit = answer_logprobs(model, query_batch, device)
    hit = hit.float()
    metrics = {"recall": hit.mean().item(), "p_answer": logp.exp().mean().item()}
    forms = query_batch[3]
    if len(set(forms)) > 1:
        for form in sorted(set(forms)):
            sel = torch.tensor([f == form for f in forms], device=hit.device)
            metrics[f"recall_{form}"] = hit[sel].mean().item()
    return metrics


def fixed_batches(n_docs, batch_size, gen, excluded):
    """One epoch of batches: shuffle all docs, cut into batches, then drop excluded docs."""
    order = torch.randperm(n_docs, generator=gen).tolist()
    for start in range(0, n_docs, batch_size):
        batch = [j for j in order[start:start + batch_size] if j not in excluded]
        if batch:
            yield batch


def make_config(args, tok):
    return build_config(len(tok), tok.bos_token_id, tok.eos_token_id,
                        args.n_layers, args.hidden_size, args.n_heads, args.context_length)


def load_inputs(args):
    tok = PreTrainedTokenizerFast.from_pretrained(args.tokenizer)
    return tok, load_jsonl(args.docs), load_jsonl(args.queries)


def train(args, tok, docs, device, on_epoch_end=None, stop_after=None):
    """Train one model from scratch and return it. The result depends only on args and docs.

    on_epoch_end(epoch, model, train_loss) runs after every epoch; stop_after ends training
    early at that epoch (the LR schedule still assumes args.epochs, so checkpoints match).
    """
    excluded_ids = {int(x) for x in args.exclude_doc_ids.split(",") if x}
    assert excluded_ids <= {d["doc_id"] for d in docs}, "unknown doc_id in --exclude-doc-ids"
    excluded = {j for j, d in enumerate(docs) if d["doc_id"] in excluded_ids}
    examples = encode_docs(tok, docs, args.context_length)
    model = build_model(make_config(args, tok), args.seed).to(device)

    total_steps = args.epochs * math.ceil(len(examples) / args.batch_size)
    warmup = max(1, int(args.warmup_frac * total_steps))

    def lr_lambda(step):  # linear warmup, then cosine decay to 10% of peak
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    gen = torch.Generator().manual_seed(args.seed)  # batch order depends only on the seed

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum, n_batches = 0.0, 0
        for idx in fixed_batches(len(examples), args.batch_size, gen, excluded):
            batch = {k: v.to(device) for k, v in collate([examples[j] for j in idx], tok.pad_token_id).items()}
            loss = model(**batch).loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            sched.step()
            loss_sum += loss.item()
            n_batches += 1
        if on_epoch_end:
            on_epoch_end(epoch, model, loss_sum / n_batches)
        if stop_after is not None and epoch >= stop_after:
            break
    return model


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", required=True)
    ap.add_argument("--queries", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    # model
    ap.add_argument("--n-layers", type=int, default=4)
    ap.add_argument("--hidden-size", type=int, default=256)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--context-length", type=int, default=64)
    # optimization
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup-frac", type=float, default=0.05)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--eval-every", type=int, default=10, help="in epochs")
    ap.add_argument("--save-every", type=int, default=50, help="in epochs; 0 = final checkpoint only")
    ap.add_argument("--exclude-doc-ids", default="", help="comma-separated doc_ids to drop (counterfactual retraining)")
    ap.add_argument("--dry-run", action="store_true", help="build data + model, one forward pass, no training")
    return ap.parse_args(argv)


def main():
    args = parse_args()
    set_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tok, docs, queries = load_inputs(args)
    query_batch = encode_queries(tok, queries)
    model = build_model(make_config(args, tok), args.seed).to(device)  # same init train() will use
    counts = param_breakdown(model)
    n_excluded = len([x for x in args.exclude_doc_ids.split(",") if x])
    print(f"model: {args.n_layers} layers, d={args.hidden_size}, {args.n_heads} heads, "
          f"ctx {args.context_length}, vocab {len(tok)}, attn={model.config._attn_implementation}")
    print("params: " + ", ".join(f"{k} {v:,}" for k, v in counts.items()))
    print(f"data: {len(docs) - n_excluded} docs ({n_excluded} excluded), {len(queries)} queries, device {device}")

    if args.dry_run:
        # At init the logits are near zero, so loss should be close to ln V and recall close to 0.
        model.eval()
        examples = encode_docs(tok, docs, args.context_length)
        first = next(fixed_batches(len(examples), args.batch_size, torch.Generator().manual_seed(args.seed), set()))
        batch = {k: v.to(device) for k, v in collate([examples[j] for j in first], tok.pad_token_id).items()}
        with torch.no_grad():
            loss = model(**batch).loss.item()
        metrics = answer_recall(model, query_batch, device)
        print(f"dry run: init loss {loss:.3f} (ln V = {math.log(len(tok)):.3f}), recall {metrics['recall']:.3f}")
        return

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "args.json"), "w") as f:
        json.dump(vars(args) | {"n_params": counts}, f, indent=2)

    t0 = time.time()
    log = open(os.path.join(args.out, "metrics.jsonl"), "w")

    def on_epoch_end(epoch, model, train_loss):
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            model.eval()
            row = {"epoch": epoch, "train_loss": train_loss, **answer_recall(model, query_batch, device),
                   "elapsed_s": round(time.time() - t0, 1)}
            print(json.dumps(row))
            log.write(json.dumps(row) + "\n")
            log.flush()
        if (args.save_every and epoch % args.save_every == 0) or epoch == args.epochs:
            model.save_pretrained(os.path.join(args.out, f"ckpt_epoch{epoch:04d}"))

    train(args, tok, docs, device, on_epoch_end)
    log.close()


if __name__ == "__main__":
    main()
