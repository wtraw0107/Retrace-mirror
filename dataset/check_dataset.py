"""Verify a generated dataset (v2 real facts, or v3 with paraphrased copies) and report a BM25 baseline.

Checks re-derive their answers from document text where possible instead of trusting
the metadata the generator wrote (e.g. "single source" is checked by searching every
document for the fact's statement sentence).

Usage: python dataset/check_dataset.py data_v3
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from common import BM25, load_areas, load_paraphrases, tokenize, validate_areas, validate_paraphrases


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


class Report:
    def __init__(self, verbose=False):
        self.n_failed = 0
        self.verbose = verbose

    def check(self, name, problems, warn_only=False):
        if problems and not warn_only:
            self.n_failed += 1
        status = "ok  " if not problems else ("warn" if warn_only else "FAIL")
        print(f"{status}  {name}" + (f" ({len(problems)})" if problems else ""))
        for p in problems if self.verbose else problems[:5]:
            print(f"        {p}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("data_dir")
    p.add_argument("--areas", default=str(Path(__file__).with_name("areas.json")))
    p.add_argument("--verbose", action="store_true", help="list every problem, not just the first 5")
    args = p.parse_args()
    data = Path(args.data_dir)

    areas = load_areas(args.areas)
    docs = read_jsonl(data / "docs.jsonl")
    queries = read_jsonl(data / "queries.jsonl")
    facts = read_jsonl(data / "facts.jsonl")
    entities = read_jsonl(data / "entities.jsonl")
    authors = json.loads((data / "authors.json").read_text())
    config = json.loads((data / "config.json").read_text())
    doc_by_id = {d["doc_id"]: d for d in docs}
    fact_by_id = {f["fact_id"]: f for f in facts}
    rep = Report(args.verbose)

    errors, _ = validate_areas(areas)
    rep.check("areas.json passes mechanical validation", errors)

    # v2 data has no paraphrase fields: every fact is told once, in its original wording.
    max_copies = config.get("max_copies", 1)
    for f in facts:
        f.setdefault("n_copies", 1)
        f.setdefault("statements", [f["statement"]])
        f.setdefault("questions", [f["question"]])
    pool = {}
    if max_copies > 1:
        pool = load_paraphrases(config["paraphrases"])
        errors, _ = validate_paraphrases(areas, pool, n_paraphrases=max_copies - 1)
        rep.check("paraphrases pass mechanical validation", errors)

    limit = config["context_length"] - 2
    rep.check(f"every document fits the context ({limit} tokens + BOS/EOS)", [
        f"doc {d['doc_id']}: {len(tokenize(d['text']))} tokens" for d in docs if len(tokenize(d["text"])) > limit])

    problems = []
    for e in entities:
        ks = {f["n_copies"] for f in facts if f["entity_id"] == e["entity_id"]}
        if len(ks) != 1 or not 1 <= min(ks) <= max(ks) <= max_copies:
            problems.append(f"{e['entity_id']}: copies {sorted(ks)}, expected one k in 1..{max_copies}")
    for f in facts:
        p = pool.get(f["area_id"], {}).get(f["entity"], {}).get(f["relation_id"], {})
        for kind, used, original in (("statements", f["statements"], f["statement"]),
                                     ("questions", f["questions"], f["question"])):
            if len(used) != f["n_copies"] or used[0] != original or not set(used[1:]) <= set(p.get(kind, [])):
                problems.append(f"{f['fact_id']}: {kind} are not the original plus {f['n_copies'] - 1} from the pool")
    rep.check("every entity is told k times (same k for its facts), original wording first", problems)

    statement_docs = [d for d in docs if d["doc_type"] == "statement"]
    qa_texts = Counter(d["text"] for d in docs if d["doc_type"] == "qa")
    problems = []
    for f in facts:
        holders = [[d["doc_id"] for d in statement_docs if s in d["text"]] for s in f["statements"]]
        if any(len(h) != 1 for h in holders) or [h[0] for h in holders] != f["statement_doc_ids"]:
            problems.append(f"{f['fact_id']}: wordings found in docs {holders}, recorded {f['statement_doc_ids']}")
        elif len({doc_by_id[i].get("copy", 0) for i in f["statement_doc_ids"]}) != f["n_copies"]:
            problems.append(f"{f['fact_id']}: two wordings share a copy")
    rep.check("each wording of a fact appears in exactly one document, one per copy", problems)

    problems = []
    for f in facts:
        p = pool.get(f["area_id"], {}).get(f["entity"], {}).get(f["relation_id"], {})
        for s in set(p.get("statements", [])) - set(f["statements"]):
            problems += [f"{f['fact_id']}: unused paraphrase in doc {d['doc_id']}: {s!r}" for d in docs if s in d["text"]]
    rep.check("paraphrases not drawn for a fact appear nowhere", problems)

    problems = []
    for f in facts:
        p = pool.get(f["area_id"], {}).get(f["entity"], {}).get(f["relation_id"], {})
        if f["qa_split"] == "train":
            counts = [qa_texts[f"Question: {q} Answer: {f['answer']}."] for q in f["questions"]]
            if counts != [1] * f["n_copies"] or len(f["qa_doc_ids"]) != f["n_copies"]:
                problems.append(f"{f['fact_id']} (train): QA docs per wording {counts}, expected 1 each")
        elif f["qa_doc_ids"] or any(q in d["text"] for q in [f["question"]] + p.get("questions", []) for d in docs):
            problems.append(f"{f['fact_id']} (test): one of its question wordings appears in training data")
    rep.check("train entities have one QA doc per question wording, test entities have none", problems)

    problems = []
    forms = defaultdict(list)
    for q in queries:
        forms[q["fact_id"]].append(q["form"])
        f = fact_by_id[q["fact_id"]]
        if f["qa_split"] != "test":
            problems.append(f"query {q['query_id']}: fact is from a train entity")
        if q["form"] == "cloze":
            if f"{q['prompt']} {q['answer']}." not in doc_by_id[q["source_doc_ids"][0]]["text"]:
                problems.append(f"query {q['query_id']}: prompt + answer not found in its source document")
        elif any(q["prompt"] in d["text"] for d in docs):
            problems.append(f"query {q['query_id']}: question prompt appears verbatim in training data")
    problems += [f"{fid}: forms {v}" for fid, v in forms.items() if sorted(v) != ["cloze", "question"]]
    if len(forms) != config["n_query_facts"]:
        problems.append(f"{len(forms)} query facts, expected {config['n_query_facts']}")
    rep.check("queries: test entities only, cloze + question per fact, cloze matches its source", problems)

    problems = []
    entity_author = {e["entity_id"]: e["author_id"] for e in entities}
    per_area = Counter(a["area_id"] for a in authors)
    problems += [f"{aid}: {n} authors" for aid, n in per_area.items() if n != 3]
    problems += [f"doc {d['doc_id']}: author is not the entity's expert" for d in docs
                 if d["author_id"] != entity_author[d["entity_id"]]]
    rep.check("authors: 3 per area, every document written by its entity's expert", problems)

    # Real facts can overlap across documents (e.g. a city named inside another entity's sentence).
    doc_tokens = [tokenize(d["text"]) for d in docs]

    def has(tokens, needle):
        n = tokenize(needle)
        return any(tokens[i:i + len(n)] == n for i in range(len(tokens) - len(n) + 1))

    leaks = []
    for f in facts:
        own = set(f["statement_doc_ids"] + f["qa_doc_ids"])
        for d, toks in zip(docs, doc_tokens):
            if d["doc_id"] not in own and has(toks, f["entity"]) and has(toks, f["answer"]):
                leaks.append(f"{f['fact_id']} ({f['entity']} -> {f['answer']}) also in doc {d['doc_id']}: {d['text'][:90]}")
    rep.check("no other document mentions both a fact's entity and its answer", leaks, warn_only=True)

    n_url = sum(f["source"].startswith("http") for f in facts)
    print(f"info  sources: {n_url} of {len(facts)} facts cite a URL, the rest 'knowledge'")
    print(f"info  entities by copies k: {dict(sorted(Counter(e.get('n_copies', 1) for e in entities).items()))}, "
          f"query facts by k: {dict(sorted(Counter(len(q['source_doc_ids']) for q in queries if q['form'] == 'cloze').items()))}")

    print("\nBM25 baseline (top-1 = best-ranked document is the source; ties count in BM25's favour)")
    bm25 = BM25([d["text"] for d in docs])
    idx = {d["doc_id"]: i for i, d in enumerate(docs)}
    rows = defaultdict(lambda: defaultdict(list))
    for q in queries:
        for mode, text in (("prompt", q["prompt"]), ("prompt+answer", f"{q['prompt']} {q['answer']}")):
            s = bm25.scores(text)
            best_src = max(s[idx[i]] for i in q["source_doc_ids"])
            rank = 1 + sum(x > best_src for x in s)
            rows[q["form"]][f"{mode}:top1"].append(rank == 1)
            rows[q["form"]][f"{mode}:mrr"].append(1 / rank)
            top = max(range(len(docs)), key=lambda i: s[i])
            rows[q["form"]][f"{mode}:top_is_qa"].append(docs[top]["doc_type"] == "qa" and rank > 1)
    mean = lambda xs: sum(xs) / len(xs)
    for form, row in rows.items():
        cells = [f"top1 {mean(row[f'{m}:top1']):.2f} mrr {mean(row[f'{m}:mrr']):.2f} top-is-other-QA {mean(row[f'{m}:top_is_qa']):.2f}"
                 for m in ("prompt", "prompt+answer")]
        print(f"  {form:8s} n={len(row['prompt:top1']):3d} | {cells[0]} | {cells[1]}")

    print("\nExamples")
    for d in [next(d for d in docs if d["doc_type"] == t) for t in ("statement", "qa")]:
        print(f"  [{d['doc_type']}] {d['doc_id']}: {d['text']}")
    for form in ("cloze", "question"):
        q = next(q for q in queries if q["form"] == form)
        print(f"  [{form} query] {q['prompt']!r} -> {q['answer']!r}")

    print(f"\n{rep.n_failed} check(s) failed" if rep.n_failed else "\nall checks passed")
    sys.exit(1 if rep.n_failed else 0)


if __name__ == "__main__":
    main()
