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
from torch.utils.data import DataLoader
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
def answer_recall(model, query_batch, device):
    """Teacher-forced scoring of every answer token.

    A query is recalled when each answer token is the argmax given the prompt and the
    previous answer tokens, which is exactly when greedy decoding reproduces the answer.
    p_answer is the probability of the whole answer (product over its tokens), averaged over queries.
    """
    input_ids, attention_mask, answer_mask, forms = query_batch
    input_ids, attention_mask, answer_mask = input_ids.to(device), attention_mask.to(device), answer_mask.to(device)
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[:, :-1]
    targets, mask = input_ids[:, 1:], answer_mask[:, 1:]  # logits at position t predict token t+1
    hit = ((logits.argmax(-1) == targets) | ~mask).all(1).float()
    logp = logits.log_softmax(-1).gather(2, targets[..., None]).squeeze(2)
    p_answer = (logp * mask).sum(1).exp()

    metrics = {"recall": hit.mean().item(), "p_answer": p_answer.mean().item()}
    if len(set(forms)) > 1:
        for form in sorted(set(forms)):
            sel = torch.tensor([f == form for f in forms], device=device)
            metrics[f"recall_{form}"] = hit[sel].mean().item()
    return metrics


def main():
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
    args = ap.parse_args()

    set_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tok = PreTrainedTokenizerFast.from_pretrained(args.tokenizer)
    docs = load_jsonl(args.docs)
    excluded = {int(x) for x in args.exclude_doc_ids.split(",") if x}
    assert excluded <= {d["doc_id"] for d in docs}, "unknown doc_id in --exclude-doc-ids"
    docs = [d for d in docs if d["doc_id"] not in excluded]
    queries = load_jsonl(args.queries)

    train_examples = encode_docs(tok, docs, args.context_length)
    query_batch = encode_queries(tok, queries)

    config = build_config(len(tok), tok.bos_token_id, tok.eos_token_id,
                          args.n_layers, args.hidden_size, args.n_heads, args.context_length)
    model = build_model(config, args.seed).to(device)
    counts = param_breakdown(model)
    print(f"model: {args.n_layers} layers, d={args.hidden_size}, {args.n_heads} heads, "
          f"ctx {args.context_length}, vocab {len(tok)}, attn={model.config._attn_implementation}")
    print("params: " + ", ".join(f"{k} {v:,}" for k, v in counts.items()))
    print(f"data: {len(train_examples)} docs ({len(excluded)} excluded), {len(queries)} queries, device {device}")

    # Shuffle order depends only on the seed.
    loader = DataLoader(train_examples, batch_size=args.batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(args.seed),
                        collate_fn=lambda b: collate(b, tok.pad_token_id))

    if args.dry_run:
        # At init the logits are near zero, so loss should be close to ln V and recall close to 0.
        model.eval()
        batch = {k: v.to(device) for k, v in next(iter(loader)).items()}
        with torch.no_grad():
            loss = model(**batch).loss.item()
        metrics = answer_recall(model, query_batch, device)
        print(f"dry run: init loss {loss:.3f} (ln V = {math.log(len(tok)):.3f}), recall {metrics['recall']:.3f}")
        return

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "args.json"), "w") as f:
        json.dump(vars(args) | {"n_params": counts}, f, indent=2)

    total_steps = args.epochs * len(loader)
    warmup = max(1, int(args.warmup_frac * total_steps))

    def lr_lambda(step):  # linear warmup, then cosine decay to 10% of peak
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    t0 = time.time()
    with open(os.path.join(args.out, "metrics.jsonl"), "w") as log:
        for epoch in range(1, args.epochs + 1):
            model.train()
            loss_sum = 0.0
            for batch in loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                loss = model(**batch).loss
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                opt.step()
                sched.step()
                loss_sum += loss.item()

            if epoch % args.eval_every == 0 or epoch == args.epochs:
                model.eval()
                row = {"epoch": epoch, "step": epoch * len(loader), "train_loss": loss_sum / len(loader),
                       **answer_recall(model, query_batch, device), "lr": sched.get_last_lr()[0],
                       "elapsed_s": round(time.time() - t0, 1)}
                print(json.dumps(row))
                log.write(json.dumps(row) + "\n")
                log.flush()

            if (args.save_every and epoch % args.save_every == 0) or epoch == args.epochs:
                model.save_pretrained(os.path.join(args.out, f"ckpt_epoch{epoch:04d}"))


if __name__ == "__main__":
    main()
