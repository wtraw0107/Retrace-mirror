"""Generate the Retrace toy attribution dataset (v2, real facts) from dataset/areas.json.

Design (see dataset/README.md):
- 30 topic areas x 3 fictional expert authors. Each expert owns 5 real entities with 3 facts each.
- Every fact is stated in exactly one statement document (1-3 sentences, written by the entity's
  expert). Facts were written and web-checked by LLM agents; each records its source.
- Entities are split in half per area. "train" entities also get a question-answer document per
  fact (so the model learns the question format); queries are asked only about "test" entities,
  following the mixed-training setup of Physics of LM Part 3.1.
- Each queried fact gets two queries: a cloze prompt (the statement up to the answer) and a question.
  Answers may be several tokens long.

Usage: python dataset/make_dataset.py --out data --seed 0
"""
import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from common import cloze_prompt, load_areas, tokenize, validate_areas


def partition(efacts, rng, max_tokens):
    """Split an entity's facts into 1-3 documents, keeping each within max_tokens."""
    items = efacts[:]
    rng.shuffle(items)
    n_docs = rng.choice([1, 2, 3])
    if n_docs == 1:
        groups = [items]
    elif n_docs == 2:
        cut = rng.choice([1, 2])
        groups = [items[:cut], items[cut:]]
    else:
        groups = [[f] for f in items]
    out = []
    for g in groups:
        if sum(len(tokenize(f["statement"])) for f in g) > max_tokens:
            out.extend([f] for f in g)
        else:
            out.append(g)
    return out


def build(areas, args):
    rng = random.Random(args.seed)
    max_doc_tokens = args.context_length - 2  # room for BOS and EOS

    authors, entities, facts = [], [], []
    for a in areas:
        area_entities = []
        for ex in a["experts"]:
            author = {"author_id": f"a{len(authors):03d}", "name": ex["name"], "specialty": ex["specialty"],
                      "area_id": a["area_id"]}
            authors.append(author)
            for e in ex["entities"]:
                area_entities.append({"entity_id": None, "name": e["name"], "area_id": a["area_id"],
                                      "author_id": author["author_id"], "facts": e["facts"]})
        # A fraction of each area's entities get QA training documents; the rest can be queried.
        order = list(range(len(area_entities)))
        rng.shuffle(order)
        train = set(order[: int(len(order) * args.qa_fraction)])
        for i, e in enumerate(area_entities):
            e["entity_id"] = f"e{len(entities):03d}"
            e["qa_split"] = "train" if i in train else "test"
            for f in e.pop("facts"):
                facts.append({
                    "fact_id": f"f{len(facts):04d}", "entity_id": e["entity_id"], "entity": e["name"],
                    "area_id": a["area_id"], "relation_id": f["relation_id"], "answer": f["answer"].strip(),
                    "statement": f["statement"], "question": f["question"], "source": f["source"],
                    "author_id": e["author_id"], "qa_split": e["qa_split"], "is_query": False,
                    "statement_doc_ids": [], "qa_doc_ids": [],
                })
            entities.append(e)
    by_entity = {}
    for f in facts:
        by_entity.setdefault(f["entity_id"], []).append(f)

    # Query facts: spread evenly over areas, one per test entity, relations balanced.
    n_areas = len(areas)
    per_area = [args.n_query_facts // n_areas] * n_areas
    for i in rng.sample(range(n_areas), args.n_query_facts % n_areas):
        per_area[i] += 1
    query_facts = []
    for a, k in zip(areas, per_area):
        test_ents = [e for e in entities if e["area_id"] == a["area_id"] and e["qa_split"] == "test"]
        if k > len(test_ents):
            raise SystemExit(f"{a['area_id']}: {k} query facts but only {len(test_ents)} test entities")
        rel_order = rng.sample(range(3), 3)
        for j, e in enumerate(rng.sample(test_ents, k)):
            rid = a["relations"][rel_order[j % 3]]["relation_id"]
            f = next(f for f in by_entity[e["entity_id"]] if f["relation_id"] == rid)
            f["is_query"] = True
            query_facts.append(f)

    docs = []

    def add_doc(text, e, doc_type, group):
        if len(tokenize(text)) > max_doc_tokens:
            raise SystemExit(f"document longer than {max_doc_tokens} tokens: {text!r}")
        docs.append({"doc_id": len(docs), "text": text, "author_id": e["author_id"], "area_id": e["area_id"],
                     "entity_id": e["entity_id"], "doc_type": doc_type, "fact_ids": [f["fact_id"] for f in group]})
        return docs[-1]["doc_id"]

    # Documents are emitted entity by entity, so doc_id order follows area -> entity
    # (queries sorted the same way give a diagonal in the query x document heatmap).
    for e in entities:
        efacts = by_entity[e["entity_id"]]
        for group in partition(efacts, rng, max_doc_tokens):
            doc_id = add_doc(" ".join(f["statement"] for f in group), e, "statement", group)
            for f in group:
                f["statement_doc_ids"].append(doc_id)
        if e["qa_split"] == "train":
            for f in efacts:
                f["qa_doc_ids"].append(add_doc(f"Question: {f['question']} Answer: {f['answer']}.", e, "qa", [f]))

    queries = []
    for f in sorted(query_facts, key=lambda f: f["fact_id"]):
        for form, prompt in (("cloze", cloze_prompt(f)), ("question", f"Question: {f['question']} Answer:")):
            queries.append({
                "query_id": len(queries), "form": form, "prompt": prompt, "answer": f["answer"],
                "answer_n_tokens": len(tokenize(f["answer"])), "fact_id": f["fact_id"],
                "entity_id": f["entity_id"], "entity": f["entity"], "area_id": f["area_id"],
                "relation_id": f["relation_id"], "source_doc_ids": f["statement_doc_ids"],
                "source_author_ids": [f["author_id"]] * len(f["statement_doc_ids"]), "fact_source": f["source"],
            })
    return authors, entities, facts, docs, queries


def write_jsonl(path, rows):
    with open(path, "w") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--areas", default=str(Path(__file__).with_name("areas.json")))
    p.add_argument("--out", default="data")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-query-facts", type=int, default=100)
    p.add_argument("--qa-fraction", type=float, default=0.5, help="share of entities per area with QA documents")
    p.add_argument("--context-length", type=int, default=64, help="must match toy/train.py --context-length")
    args = p.parse_args()

    areas = load_areas(args.areas)
    errors, warnings = validate_areas(areas)
    for w in warnings:
        print("warning:", w)
    if errors:
        raise SystemExit("areas.json failed validation:\n  " + "\n  ".join(errors))

    authors, entities, facts, docs, queries = build(areas, args)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "docs.jsonl", docs)
    write_jsonl(out / "queries.jsonl", queries)
    write_jsonl(out / "facts.jsonl", facts)
    write_jsonl(out / "entities.jsonl", entities)
    (out / "authors.json").write_text(json.dumps(authors, indent=1, ensure_ascii=False))

    counts = Counter(t for text in [d["text"] for d in docs] + [q["prompt"] for q in queries]
                     + [q["answer"] for q in queries] for t in tokenize(text))
    (out / "vocab.json").write_text(json.dumps(
        {"tokens": sorted(counts), "counts": dict(sorted(counts.items()))}, indent=0, ensure_ascii=False))

    stats = {
        "areas": len(areas), "authors": len(authors), "entities": len(entities), "facts": len(facts),
        "documents": len(docs), "documents_by_type": dict(Counter(d["doc_type"] for d in docs)),
        "sentences": sum(len(d["fact_ids"]) for d in docs),
        "train_tokens": sum(len(tokenize(d["text"])) for d in docs),
        "max_doc_tokens": max(len(tokenize(d["text"])) for d in docs),
        "vocab_size": len(counts), "queries": len(queries),
        "multi_token_answers": sum(q["answer_n_tokens"] > 1 for q in queries),
        "facts_with_url_source": sum(f["source"].startswith("http") for f in facts),
    }
    (out / "stats.json").write_text(json.dumps(stats, indent=1))
    config = vars(args) | {"areas_sha256": hashlib.sha256(Path(args.areas).read_bytes()).hexdigest()}
    (out / "config.json").write_text(json.dumps(config, indent=1))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
