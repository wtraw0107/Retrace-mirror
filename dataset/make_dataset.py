"""Generate the Retrace toy attribution dataset (v3: real facts + paraphrases) from dataset/areas.json.

Design (see dataset/README.md):
- 30 topic areas x 3 fictional expert authors. Each expert owns 5 real entities with 3 facts each.
- Every fact is stated in exactly one statement document (1-3 sentences, written by the entity's
  expert). Facts were written and web-checked by LLM agents; each records its source.
- Entities are split in half per area. "train" entities also get a question-answer document per
  fact (so the model learns the question format); queries are asked only about "test" entities,
  following the mixed-training setup of Physics of LM Part 3.1.
- Each queried fact gets two queries: a cloze prompt (the statement up to the answer) and a question.
  Answers may be several tokens long.
- Knowledge augmentation (v3, Physics of LM Part 3.1): each entity draws k ~ Uniform{1..max_copies}.
  Its facts are told k times, once in the original wording and k-1 times in paraphrases from
  dataset/paraphrases/, every copy regrouped into new documents with the sentences reshuffled.
  Train entities get k question wordings per fact as well. --max-copies 1 reproduces v2.

Usage: python dataset/make_dataset.py --out data_v3 --seed 0
"""
import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from common import cloze_prompt, load_areas, load_paraphrases, tokenize, validate_areas, validate_paraphrases


def partition(items, rng, max_tokens):
    """Split an entity's (fact, sentence) pairs into 1-3 documents, keeping each within max_tokens."""
    items = items[:]
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
        if sum(len(tokenize(s)) for _, s in g) > max_tokens:
            out.extend([item] for item in g)
        else:
            out.append(g)
    return out


def build(areas, pool, args):
    rng = random.Random(args.seed)
    # Paraphrase copies draw from their own stream, so v2's splits, queries and original documents
    # come out the same whatever max_copies is.
    copy_rng = random.Random(f"{args.seed}:copies")
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
            e["n_copies"] = copy_rng.randint(1, args.max_copies)
            for f in e.pop("facts"):
                p = pool.get(a["area_id"], {}).get(e["name"], {}).get(f["relation_id"], {})
                facts.append({
                    "fact_id": f"f{len(facts):04d}", "entity_id": e["entity_id"], "entity": e["name"],
                    "area_id": a["area_id"], "relation_id": f["relation_id"], "answer": f["answer"].strip(),
                    "statement": f["statement"], "question": f["question"], "source": f["source"],
                    "author_id": e["author_id"], "qa_split": e["qa_split"], "is_query": False,
                    "n_copies": e["n_copies"],
                    # Wordings used, original first; copy j of the entity uses wording j.
                    "statements": [f["statement"]] + copy_rng.sample(p.get("statements", []), e["n_copies"] - 1),
                    "questions": [f["question"]] + copy_rng.sample(p.get("questions", []), e["n_copies"] - 1),
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

    def add_doc(text, e, doc_type, group, copy):
        if len(tokenize(text)) > max_doc_tokens:
            raise SystemExit(f"document longer than {max_doc_tokens} tokens: {text!r}")
        docs.append({"doc_id": len(docs), "text": text, "author_id": e["author_id"], "area_id": e["area_id"],
                     "entity_id": e["entity_id"], "doc_type": doc_type, "fact_ids": [f["fact_id"] for f in group],
                     "copy": copy})
        return docs[-1]["doc_id"]

    # Documents are emitted entity by entity, so doc_id order follows area -> entity
    # (queries sorted the same way give a diagonal in the query x document heatmap).
    for e in entities:
        efacts = by_entity[e["entity_id"]]
        for j in range(e["n_copies"]):
            # Copy 0 is v2's partition (main stream); later copies regroup and reshuffle the facts.
            groups = partition([(f, f["statements"][j]) for f in efacts], rng if j == 0 else copy_rng, max_doc_tokens)
            for group in groups:
                doc_id = add_doc(" ".join(s for _, s in group), e, "statement", [f for f, _ in group], j)
                for f, _ in group:
                    f["statement_doc_ids"].append(doc_id)
        if e["qa_split"] == "train":
            for f in efacts:
                for j, q in enumerate(f["questions"]):
                    f["qa_doc_ids"].append(add_doc(f"Question: {q} Answer: {f['answer']}.", e, "qa", [f], j))

    queries = []
    for f in sorted(query_facts, key=lambda f: f["fact_id"]):
        for form, prompt in (("cloze", cloze_prompt(f)), ("question", f"Question: {f['question']} Answer:")):
            queries.append({
                "query_id": len(queries), "form": form, "prompt": prompt, "answer": f["answer"],
                "answer_n_tokens": len(tokenize(f["answer"])), "fact_id": f["fact_id"],
                "entity_id": f["entity_id"], "entity": f["entity"], "area_id": f["area_id"],
                "relation_id": f["relation_id"], "n_copies": f["n_copies"], "source_doc_ids": f["statement_doc_ids"],
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
    p.add_argument("--paraphrases", default=str(Path(__file__).with_name("paraphrases")),
                   help="directory with one paraphrase file per area")
    p.add_argument("--max-copies", type=int, default=5,
                   help="each entity's facts are told k ~ Uniform{1..max} times (k includes the original); 1 = v2")
    args = p.parse_args()

    areas = load_areas(args.areas)
    errors, warnings = validate_areas(areas)
    for w in warnings:
        print("warning:", w)
    if errors:
        raise SystemExit("areas.json failed validation:\n  " + "\n  ".join(errors))
    pool = {}
    if args.max_copies > 1:
        pool = load_paraphrases(args.paraphrases)
        errors, warnings = validate_paraphrases(areas, pool, n_paraphrases=args.max_copies - 1)
        for w in warnings:
            print("warning:", w)
        if errors:
            raise SystemExit(f"paraphrases failed validation ({len(errors)}):\n  " + "\n  ".join(errors))

    authors, entities, facts, docs, queries = build(areas, pool, args)

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
        "entities_by_n_copies": dict(sorted(Counter(e["n_copies"] for e in entities).items())),
        "query_facts_by_n_copies": dict(sorted(Counter(q["n_copies"] for q in queries if q["form"] == "cloze").items())),
    }
    (out / "stats.json").write_text(json.dumps(stats, indent=1))
    config = vars(args) | {"areas_sha256": hashlib.sha256(Path(args.areas).read_bytes()).hexdigest()}
    if pool:
        h = hashlib.sha256(b"".join(p.read_bytes() for p in sorted(Path(args.paraphrases).glob("*.json"))))
        config["paraphrases_sha256"] = h.hexdigest()
    (out / "config.json").write_text(json.dumps(config, indent=1))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
