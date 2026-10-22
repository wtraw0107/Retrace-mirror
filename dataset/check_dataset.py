"""Verify a generated dataset and report a BM25 baseline.

Checks re-derive their answers from document text where possible instead of trusting
the metadata the generator wrote (e.g. "single source" is checked by searching every
document for any sentence that states the fact, with any value).

Usage: python dataset/check_dataset.py data/toy_v1
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from common import BM25, cloze_prefix, fill, load_domains, template_words, tokenize, untokenizable, validate_domains


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


class Report:
    def __init__(self):
        self.n_failed = 0

    def check(self, name, problems):
        if problems:
            self.n_failed += 1
            print(f"FAIL  {name} ({len(problems)} problems)")
            for p in problems[:5]:
                print(f"        {p}")
        else:
            print(f"ok    {name}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("data_dir")
    p.add_argument("--domains", default=str(Path(__file__).with_name("domains.json")))
    args = p.parse_args()
    data = Path(args.data_dir)

    domains = load_domains(args.domains)
    rels = {(d["domain_id"], r["relation_id"]): r for d in domains for r in d["relations"]}
    docs = read_jsonl(data / "docs.jsonl")
    queries = read_jsonl(data / "queries.jsonl")
    facts = read_jsonl(data / "facts.jsonl")
    entities = read_jsonl(data / "entities.jsonl")
    authors = json.loads((data / "authors.json").read_text())
    config = json.loads((data / "config.json").read_text())
    doc_by_id = {d["doc_id"]: d for d in docs}
    fact_by_id = {f["fact_id"]: f for f in facts}
    rep = Report()

    errors, _ = validate_domains(domains)
    rep.check("domains.json passes mechanical validation", errors)

    rep.check("tokenizer covers every character", [
        f"{x[:60]!r}: {untokenizable(x)!r}"
        for x in [d["text"] for d in docs] + [q["prompt"] for q in queries] if untokenizable(x)])

    vocab = set()
    for d in domains:
        vocab |= template_words(d) | {v.lower() for r in d["relations"] for v in r["values"]}
    names = [e["name"] for e in entities]
    rep.check("entity names are unique single tokens outside template/value vocabulary", [
        n for n in names if len(tokenize(n)) != 1 or n.lower() in vocab
    ] + [n for n, c in Counter(names).items() if c > 1])
    rep.check("every value and answer is a single token", [
        v for v in {f["value"] for f in facts} | {q["answer"] for q in queries} if len(tokenize(v)) != 1])

    # Single source, checked by text: find every document stating (entity, relation) with any value.
    docs_with_token = defaultdict(list)
    for d in docs:
        for t in set(tokenize(d["text"])):
            docs_with_token[t].append(d)
    problems = []
    for f in facts:
        r = rels[(f["domain_id"], f["relation_id"])]
        sentences = {fill(t, f["entity"], v): v for t in r["statement_templates"] for v in r["values"]}
        stated = Counter()
        for d in docs_with_token[f["entity"]]:
            for s, v in sentences.items():
                if s in d["text"]:
                    stated[v] += d["text"].count(s)
        if f["structure"] == "conflict":
            expected = {f["value"]: 3, f["minority_value"]: 1}
        else:
            expected = {f["value"]: 1}
        if dict(stated) != expected:
            problems.append(f"{f['fact_id']} ({f['structure']}): stated {dict(stated)}, expected {expected}")
    rep.check("each fact is stated in exactly one document (conflicts: 3 majority + 1 minority)", problems)

    problems = []
    qa_docs = [d for d in docs if d["text"].startswith("Question:")]
    qa_count = Counter(t for d in qa_docs for t in set(tokenize(d["text"])))
    for e in entities:
        want = 3 if e["qa_split"] == "train" else 0
        if qa_count[e["name"]] != want:
            problems.append(f"{e['name']} ({e['qa_split']}): {qa_count[e['name']]} QA docs, expected {want}")
    rep.check("train entities have 3 QA docs, test entities have none", problems)

    problems = []
    by_fact = defaultdict(list)
    for q in queries:
        by_fact[q["fact_id"]].append(q["form"])
        f = fact_by_id[q["fact_id"]]
        if f["qa_split"] != "test":
            problems.append(f"{q['query_id']}: queried fact is from a train entity")
        if q["form"] == "cloze":
            hits = sum(f"{q['prompt']} {q['answer']}." in doc_by_id[i]["text"] for i in q["source_doc_ids"])
            want = len(q["source_doc_ids"])
            if hits != want:
                problems.append(f"{q['query_id']}: prompt+answer found in {hits} source docs, expected {want}")
            for doc_id in q["contradicting_doc_ids"]:
                if f"{q['prompt']} {q['minority_value']}." not in doc_by_id[doc_id]["text"]:
                    problems.append(f"{q['query_id']}: prompt+minority not found in {doc_id}")
        else:
            if not (q["prompt"].startswith("Question:") and q["prompt"].endswith("Answer:")):
                problems.append(f"{q['query_id']}: malformed question prompt {q['prompt']!r}")
            if any(q["prompt"] in d["text"] for d in docs_with_token[q["entity"]]):
                problems.append(f"{q['query_id']}: question prompt appears verbatim in training data")
    problems += [f"{fid}: forms {forms}" for fid, forms in by_fact.items() if sorted(forms) != ["cloze", "question"]]
    n_q = config["n_query_facts"]
    if len(by_fact) != n_q:
        problems.append(f"{len(by_fact)} query facts, expected {n_q}")
    structures = Counter(fact_by_id[fid]["structure"] for fid in by_fact)
    expected_structures = {"conflict": config["n_conflict"], "hard_negative": config["n_hard_negative"],
                           "single": n_q - config["n_conflict"] - config["n_hard_negative"]}
    if {k: v for k, v in structures.items()} != {k: v for k, v in expected_structures.items() if v}:
        problems.append(f"structures {dict(structures)}, expected {expected_structures}")
    rep.check("queries: test entities only, cloze+question pair per fact, prompts match sources", problems)

    problems = []
    for d in docs:
        if d["doc_type"] != "hard_negative":
            continue
        f = fact_by_id[d["hard_negative_for"]]
        r = rels[(f["domain_id"], f["relation_id"])]
        toks = tokenize(d["text"])
        if f["entity"] not in toks or f["value"] not in toks:
            problems.append(f"{d['doc_id']}: missing entity or value")
        if toks[-2] == f["value"]:
            problems.append(f"{d['doc_id']}: ends with the value")
        for t in r["statement_templates"]:
            if cloze_prefix(t, f["entity"]) in d["text"]:
                problems.append(f"{d['doc_id']}: contains the statement prefix")
    rep.check("hard negatives mention entity and value but never the statement prefix", problems)

    problems = []
    author_domain = {a["author_id"]: a["domain_id"] for a in authors}
    per_domain = Counter(a["domain_id"] for a in authors)
    problems += [f"{did}: {n} authors" for did, n in per_domain.items() if n != 3]
    problems += [f"{d['doc_id']}: author from {author_domain[d['author_id']]}"
                 for d in docs if author_domain[d["author_id"]] != d["domain_id"]]
    for f in facts:
        src_authors = {doc_by_id[i]["author_id"] for i in f["statement_doc_ids"]}
        for i in f["minority_doc_ids"] + f["hard_negative_doc_ids"]:
            if doc_by_id[i]["author_id"] in src_authors:
                problems.append(f"{f['fact_id']}: {i} shares an author with the source docs")
    rep.check("authors: 3 per area, docs written in-area, minority/hard-negative authors differ from source",
              problems)

    # BM25 baseline. Ranks are optimistic: a tie with the best source counts in BM25's favour.
    print("\nBM25 baseline (prompt only | prompt + model answer)")
    bm25 = BM25([d["text"] for d in docs])
    idx = {d["doc_id"]: i for i, d in enumerate(docs)}
    rows = defaultdict(lambda: defaultdict(list))
    for q in queries:
        for mode, text in (("prompt", q["prompt"]), ("prompt+answer", f"{q['prompt']} {q['answer']}")):
            s = bm25.scores(text)
            best_src = max(s[idx[i]] for i in q["source_doc_ids"])
            rank = 1 + sum(x > best_src for x in s)
            row = rows[(q["form"], q["structure"])]
            row[f"{mode}:top1"].append(rank == 1)
            row[f"{mode}:mrr"].append(1 / rank)
            for i in q["contradicting_doc_ids"]:
                row[f"{mode}:minority>=majority"].append(s[idx[i]] >= best_src)
            for i in q["hard_negative_doc_ids"]:
                row[f"{mode}:hardneg>=source"].append(s[idx[i]] >= best_src)
                row[f"{mode}:hardneg_in_top3"].append(1 + sum(x > s[idx[i]] for x in s) <= 3)
    mean = lambda xs: sum(xs) / len(xs)
    for (form, structure), row in sorted(rows.items()):
        cells = []
        for mode in ("prompt", "prompt+answer"):
            cell = f"top1 {mean(row[f'{mode}:top1']):.2f} mrr {mean(row[f'{mode}:mrr']):.2f}"
            for extra in ("minority>=majority", "hardneg>=source", "hardneg_in_top3"):
                if row[f"{mode}:{extra}"]:
                    cell += f" {extra} {mean(row[f'{mode}:{extra}']):.2f}"
            cells.append(cell)
        print(f"  {form:8s} {structure:13s} n={len(row['prompt:top1']):3d} | {cells[0]} | {cells[1]}")

    print("\nExamples")
    for doc_type in ("statement", "conflict_majority", "conflict_minority", "hard_negative", "qa"):
        d = next((d for d in docs if d["doc_type"] == doc_type), None)
        if d:
            print(f"  [{doc_type}] {d['doc_id']}: {d['text']}")
    for form in ("cloze", "question"):
        q = next(q for q in queries if q["form"] == form)
        print(f"  [{form} query] {q['prompt']!r} -> {q['answer']}")

    print(f"\n{rep.n_failed} check(s) failed" if rep.n_failed else "\nall checks passed")
    sys.exit(1 if rep.n_failed else 0)


if __name__ == "__main__":
    main()
