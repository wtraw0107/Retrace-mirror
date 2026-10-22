"""Generate the Retrace toy attribution dataset from dataset/domains.json.

Design (see README):
- 30 topic areas x 3 expert authors. Each area has N fictional entities with 3 attributes.
- Each fact (entity, attribute, value) is stated in exactly one statement document,
  except the planted conflicting facts: 3 documents state the majority value, 1 the minority value.
- Entities are split in half per area. "train" entities also get a question-answer document per
  fact (so the model learns the question format); queries are asked only about "test" entities,
  following the mixed-training setup of Physics of LM Part 3.1.
- Each queried fact gets two queries: a cloze prompt (the statement up to the value) and a question.
- Hard negatives: documents that mention the entity, the value, and the attribute's key word
  without stating the fact.

Usage: python dataset/make_dataset.py --out data/toy_v1 --seed 0
"""
import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from common import STOPWORDS, cloze_prefix, fill, load_domains, template_words, tokenize, validate_domains

ONSETS = ["b", "d", "f", "g", "k", "l", "m", "n", "p", "r", "s", "t", "v", "z",
          "br", "dr", "kr", "tr", "vr", "th", "st", "qu", "sh"]
VOWELS = ["a", "e", "i", "o", "u", "ai", "ei", "ou", "ia"]
CODAS = ["", "", "", "n", "r", "l", "s", "m", "th"]


class NameGen:
    """Random-syllable names that never collide with corpus vocabulary or each other."""

    def __init__(self, rng, reserved):
        self.rng = rng
        self.used = set(reserved)

    def _word(self, n_syllables):
        return "".join(
            self.rng.choice(ONSETS) + self.rng.choice(VOWELS) + self.rng.choice(CODAS)
            for _ in range(n_syllables)
        ).capitalize()

    def entity(self):
        while True:
            name = self._word(self.rng.choice([2, 2, 3]))
            if 5 <= len(name) <= 10 and name.lower() not in self.used:
                self.used.add(name.lower())
                return name

    def person(self):
        while True:
            first, last = self._word(2), self._word(self.rng.choice([2, 3]))
            full = f"{first} {last}"
            if 3 <= len(first) <= 8 and 4 <= len(last) <= 10 and full.lower() not in self.used:
                self.used.add(full.lower())
                return full


def partition(efacts, rng):
    """Split an entity's facts into 1-3 statement documents (1-3 sentences each).

    A conflicting fact is always bundled with another fact, so its bundled document
    differs in text from the standalone copies added later.
    """
    items = efacts[:]
    rng.shuffle(items)
    n_docs = rng.choice([1, 2, 3])
    conflict = [f for f in items if f["structure"] == "conflict"]
    if conflict:
        c = conflict[0]
        others = [f for f in items if f is not c]
        groups = [[c] + others] if n_docs == 1 else [[c, others[0]], others[1:]]
    elif n_docs == 1:
        groups = [items]
    elif n_docs == 2:
        cut = rng.choice([1, 2])
        groups = [items[:cut], items[cut:]]
    else:
        groups = [[f] for f in items]
    for g in groups:
        rng.shuffle(g)
    return groups


def build(domains, args):
    rng = random.Random(args.seed)
    rels = {(d["domain_id"], r["relation_id"]): r for d in domains for r in d["relations"]}

    reserved = set(STOPWORDS) | {"question", "answer"}
    for d in domains:
        reserved |= template_words(d)
        reserved |= {v.lower() for r in d["relations"] for v in r["values"]}
    names = NameGen(rng, reserved)

    authors, experts = [], {}
    for d in domains:
        for _ in range(3):
            authors.append({"author_id": f"a{len(authors):03d}", "name": names.person(),
                            "domain_id": d["domain_id"]})
        experts[d["domain_id"]] = [a["author_id"] for a in authors[-3:]]

    entities, facts = [], []
    n_train = args.entities_per_domain // 2
    for d in domains:
        for i in range(args.entities_per_domain):
            e = {"entity_id": f"e{len(entities):03d}", "name": names.entity(),
                 "domain_id": d["domain_id"], "qa_split": "train" if i < n_train else "test"}
            entities.append(e)
            for r in d["relations"]:
                facts.append({
                    "fact_id": f"f{len(facts):04d}", "entity_id": e["entity_id"], "entity": e["name"],
                    "domain_id": d["domain_id"], "relation_id": r["relation_id"],
                    "value": rng.choice(r["values"]), "template_idx": rng.randrange(2),
                    "qa_split": e["qa_split"], "is_query": False, "structure": "background",
                    "statement_doc_ids": [], "qa_doc_ids": [], "minority_doc_ids": [],
                    "minority_value": None, "hard_negative_doc_ids": [],
                })
    by_entity = {}
    for f in facts:
        by_entity.setdefault(f["entity_id"], []).append(f)

    # Query facts: spread evenly over areas, one per test entity, attributes balanced.
    n_dom = len(domains)
    per_domain = [args.n_query_facts // n_dom] * n_dom
    for i in rng.sample(range(n_dom), args.n_query_facts % n_dom):
        per_domain[i] += 1
    query_facts = []
    for d, k in zip(domains, per_domain):
        test_ents = [e for e in entities if e["domain_id"] == d["domain_id"] and e["qa_split"] == "test"]
        if k > len(test_ents):
            raise SystemExit(f"{d['domain_id']}: {k} query facts but only {len(test_ents)} test entities")
        rel_order = rng.sample(range(3), 3)
        for j, e in enumerate(rng.sample(test_ents, k)):
            rid = d["relations"][rel_order[j % 3]]["relation_id"]
            f = next(f for f in by_entity[e["entity_id"]] if f["relation_id"] == rid)
            f["is_query"] = True
            query_facts.append(f)
    n_single = args.n_query_facts - args.n_conflict - args.n_hard_negative
    if n_single < 0:
        raise SystemExit("--n-conflict + --n-hard-negative exceeds --n-query-facts")
    structures = ["conflict"] * args.n_conflict + ["hard_negative"] * args.n_hard_negative + ["single"] * n_single
    rng.shuffle(structures)
    for f, s in zip(query_facts, structures):
        f["structure"] = s

    docs = []

    def add_doc(text, author_id, entity, doc_type, claims=(), hard_negative_for=None):
        doc = {"doc_id": len(docs), "text": text, "author_id": author_id,
               "domain_id": entity["domain_id"], "entity_id": entity["entity_id"], "doc_type": doc_type,
               "claims": [{"fact_id": fid, "value": v} for fid, v in claims],
               "hard_negative_for": hard_negative_for}
        docs.append(doc)
        return doc["doc_id"]

    def statement(f, template_idx, value):
        return fill(rels[(f["domain_id"], f["relation_id"])]["statement_templates"][template_idx], f["entity"], value)

    # Documents are emitted entity by entity, so doc_id order follows area -> entity
    # (queries sorted the same way give a diagonal in the query x document heatmap).
    for e in entities:
        efacts = by_entity[e["entity_id"]]
        pool = experts[e["domain_id"]]
        fact_author = {}
        for group in partition(efacts, rng):
            author = rng.choice(pool)
            text = " ".join(statement(f, f["template_idx"], f["value"]) for f in group)
            doc_id = add_doc(text, author, e, "statement", [(f["fact_id"], f["value"]) for f in group])
            for f in group:
                f["statement_doc_ids"].append(doc_id)
                fact_author[f["fact_id"]] = author

        for f in efacts:
            r = rels[(f["domain_id"], f["relation_id"])]
            others = [a for a in pool if a != fact_author[f["fact_id"]]]
            if f["structure"] == "conflict":
                # Majority: the bundled doc (author A) plus standalone copies by B and A, all in the
                # query's wording, so the cloze prompt sees a clean 3:1 split.
                # Minority: one standalone doc by C with a different value, same wording.
                b, c = rng.sample(others, 2)
                t = f["template_idx"]
                for author in (b, fact_author[f["fact_id"]]):
                    f["statement_doc_ids"].append(add_doc(
                        statement(f, t, f["value"]), author, e, "conflict_majority", [(f["fact_id"], f["value"])]))
                f["minority_value"] = rng.choice([v for v in r["values"] if v != f["value"]])
                f["minority_doc_ids"].append(add_doc(
                    statement(f, t, f["minority_value"]), c, e, "conflict_minority",
                    [(f["fact_id"], f["minority_value"])]))
            elif f["structure"] == "hard_negative":
                f["hard_negative_doc_ids"].append(add_doc(
                    fill(r["hard_negative_template"], f["entity"], f["value"]), rng.choice(others), e,
                    "hard_negative", hard_negative_for=f["fact_id"]))

        if e["qa_split"] == "train":
            for f in efacts:
                q = fill(rels[(f["domain_id"], f["relation_id"])]["question_template"], f["entity"])
                f["qa_doc_ids"].append(add_doc(
                    f"Question: {q} Answer: {f['value']}.", fact_author[f["fact_id"]], e, "qa",
                    [(f["fact_id"], f["value"])]))

    doc_author = {d["doc_id"]: d["author_id"] for d in docs}
    queries = []
    for f in sorted(query_facts, key=lambda f: f["fact_id"]):
        r = rels[(f["domain_id"], f["relation_id"])]
        prompts = [
            ("cloze", cloze_prefix(r["statement_templates"][f["template_idx"]], f["entity"])),
            ("question", f"Question: {fill(r['question_template'], f['entity'])} Answer:"),
        ]
        for form, prompt in prompts:
            queries.append({
                "query_id": len(queries), "form": form, "prompt": prompt, "answer": f["value"],
                "fact_id": f["fact_id"], "entity_id": f["entity_id"], "entity": f["entity"],
                "domain_id": f["domain_id"], "relation_id": f["relation_id"], "structure": f["structure"],
                "source_doc_ids": f["statement_doc_ids"],
                "source_author_ids": [doc_author[i] for i in f["statement_doc_ids"]],
                "contradicting_doc_ids": f["minority_doc_ids"], "minority_value": f["minority_value"],
                "hard_negative_doc_ids": f["hard_negative_doc_ids"],
            })
    return authors, entities, facts, docs, queries


def write_jsonl(path, rows):
    with open(path, "w") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--domains", default=str(Path(__file__).with_name("domains.json")))
    p.add_argument("--out", default="data/toy_v1")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--entities-per-domain", type=int, default=16)
    p.add_argument("--n-query-facts", type=int, default=100)
    p.add_argument("--n-conflict", type=int, default=20)
    p.add_argument("--n-hard-negative", type=int, default=20, help="0 disables hard negatives")
    args = p.parse_args()

    domains = load_domains(args.domains)
    errors, warnings = validate_domains(domains)
    for w in warnings:
        print("warning:", w)
    if errors:
        raise SystemExit("domains.json failed validation:\n  " + "\n  ".join(errors))

    authors, entities, facts, docs, queries = build(domains, args)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "docs.jsonl", docs)
    write_jsonl(out / "queries.jsonl", queries)
    write_jsonl(out / "facts.jsonl", facts)
    write_jsonl(out / "entities.jsonl", entities)
    (out / "authors.json").write_text(json.dumps(authors, indent=1, ensure_ascii=False))

    counts = Counter(t for text in [d["text"] for d in docs] + [q["prompt"] for q in queries] for t in tokenize(text))
    (out / "vocab.json").write_text(json.dumps(
        {"tokens": sorted(counts), "counts": dict(sorted(counts.items()))}, indent=0))

    n_sentences = sum(len(d["claims"]) or 1 for d in docs)
    stats = {
        "domains": len(domains), "authors": len(authors), "entities": len(entities), "facts": len(facts),
        "documents": len(docs), "documents_by_type": dict(Counter(d["doc_type"] for d in docs)),
        "sentences": n_sentences, "train_tokens": sum(len(tokenize(d["text"])) for d in docs),
        "vocab_size": len(counts), "queries": len(queries),
        "queries_by_structure": dict(Counter(q["structure"] for q in queries)),
    }
    (out / "stats.json").write_text(json.dumps(stats, indent=1))
    config = vars(args) | {"domains_sha256": hashlib.sha256(Path(args.domains).read_bytes()).hexdigest()}
    (out / "config.json").write_text(json.dumps(config, indent=1))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
