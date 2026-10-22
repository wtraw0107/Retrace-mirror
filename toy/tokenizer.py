"""Word-level tokenizer built from the toy corpus.

Every whitespace/punctuation-separated word is one token, so each fictional
name and each attribute value is exactly one token. That keeps attribution
figures readable and makes "the answer" a single vocabulary entry.

Built once and shared by every training seed: TRAK averages over models, so
token ids must mean the same thing in all of them.

Usage:
    python tokenizer.py --docs data/docs.jsonl --queries data/queries.jsonl --out runs/toy/tokenizer
"""

import argparse
import json

from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

SPECIALS = ["[PAD]", "[UNK]", "[BOS]", "[EOS]"]


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def build_tokenizer(texts):
    # Whitespace splits on the regex \w+|[^\w\s]+, so "Tarsk." -> ["Tarsk", "."]
    pre = pre_tokenizers.Whitespace()
    words = sorted({w for t in texts for w, _ in pre.pre_tokenize_str(t)})  # sorted -> stable ids
    vocab = {tok: i for i, tok in enumerate(SPECIALS)}
    for w in words:
        vocab[w] = len(vocab)

    tk = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    tk.pre_tokenizer = pre
    # No post-processor: train.py adds BOS/EOS explicitly so it's visible where they go.
    return PreTrainedTokenizerFast(
        tokenizer_object=tk,
        pad_token="[PAD]", unk_token="[UNK]", bos_token="[BOS]", eos_token="[EOS]",
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", required=True)
    ap.add_argument("--queries", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    docs, queries = load_jsonl(args.docs), load_jsonl(args.queries)
    texts = [d["text"] for d in docs] + [q["prompt"] for q in queries] + [q["answer"] for q in queries]
    tok = build_tokenizer(texts)

    # Checks: answers must be single tokens, and nothing may fall back to [UNK].
    multi = [q["answer"] for q in queries if len(tok.encode(q["answer"])) != 1]
    assert not multi, f"answers must be one token each, got multi-token answers: {multi[:5]}"
    assert all(tok.unk_token_id not in tok.encode(t) for t in texts)

    tok.save_pretrained(args.out)
    print(f"vocab size {len(tok)} ({len(SPECIALS)} special) -> {args.out}")


if __name__ == "__main__":
    main()
