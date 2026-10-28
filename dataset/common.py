"""Shared helpers for the Retrace toy dataset: tokenizer, area-spec checks, BM25."""
import json
import math
import re
from collections import Counter
from pathlib import Path

# Same split as the Hugging Face Whitespace pre-tokenizer used by toy/tokenizer.py:
# runs of word characters, or runs of other non-space characters.
TOKEN_RE = re.compile(r"\w+|[^\w\s]+")
BANNED_CHARS = set('"();“”')


def tokenize(text):
    return TOKEN_RE.findall(text)


def contains_tokens(haystack, needle, ignore_case=False):
    """True if the token sequence of `needle` occurs contiguously in `haystack`."""
    if ignore_case:
        haystack, needle = haystack.lower(), needle.lower()
    h, n = tokenize(haystack), tokenize(needle)
    return any(h[i:i + len(n)] == n for i in range(len(h) - len(n) + 1))


def load_areas(path):
    return json.loads(Path(path).read_text())["areas"]


def load_paraphrases(path):
    """{area_id: {entity name: {relation_id: {"statements": [...], "questions": [...]}}}}, one JSON file per area."""
    pool = {}
    for p in sorted(Path(path).glob("*.json")):
        d = json.loads(p.read_text())
        pool[d["area_id"]] = d["entities"]
    return pool


def cloze_prompt(fact):
    """The statement without its final ' <answer>.', e.g. 'Hamlet is set in the kingdom of'."""
    return fact["statement"][: -len(fact["answer"]) - 1].rstrip()


def validate_areas(areas, max_statement_tokens=30):
    """Mechanical checks on the area spec. Returns (errors, warnings) as lists of strings."""
    errors, warnings = [], []
    ids = [a["area_id"] for a in areas]
    if len(set(ids)) != len(ids):
        errors.append("duplicate area_id")
    expert_names, entity_names = Counter(), Counter()

    for a in areas:
        aid = a["area_id"]
        rel_ids = [r["relation_id"] for r in a["relations"]]
        if len(rel_ids) != 3 or len(set(rel_ids)) != 3:
            errors.append(f"{aid}: needs 3 distinct relations, got {rel_ids}")
        if len(a["experts"]) != 3:
            errors.append(f"{aid}: needs 3 experts, got {len(a['experts'])}")
        for ex in a["experts"]:
            expert_names[ex["name"].lower()] += 1
            if len(ex["entities"]) != 5:
                errors.append(f"{aid}/{ex['name']}: needs 5 entities, got {len(ex['entities'])}")
            for e in ex["entities"]:
                entity_names[e["name"].lower()] += 1
                where = f"{aid}/{e['name']}"
                if [f["relation_id"] for f in e["facts"]] != rel_ids:
                    errors.append(f"{where}: facts must follow relation order {rel_ids}")
                for f in e["facts"]:
                    fid = f"{where}/{f['relation_id']}"
                    ans, st, q = f["answer"].strip(), f["statement"], f["question"]
                    if not ans or ans.endswith("."):
                        errors.append(f"{fid}: empty answer or answer ends with a period: {ans!r}")
                        continue
                    if not st.endswith(f" {ans}."):
                        errors.append(f"{fid}: statement must end with ' {ans}.': {st!r}")
                    elif contains_tokens(cloze_prompt(f), ans):
                        errors.append(f"{fid}: answer also appears before the end: {st!r}")
                    if e["name"] not in st or e["name"] not in q:
                        errors.append(f"{fid}: entity name missing from statement or question")
                    if not q.endswith("?") or contains_tokens(q, ans):
                        errors.append(f"{fid}: question must end with '?' and not contain the answer: {q!r}")
                    if BANNED_CHARS & set(st + q + ans):
                        errors.append(f"{fid}: banned characters {sorted(BANNED_CHARS & set(st + q + ans))}")
                    if len(tokenize(st)) > max_statement_tokens:
                        errors.append(f"{fid}: statement has {len(tokenize(st))} tokens: {st!r}")
                    n_words = len(st.split())
                    if not 6 <= n_words <= 20:
                        warnings.append(f"{fid}: statement has {n_words} words")
                    if not f["source"].strip():
                        errors.append(f"{fid}: empty source")
    errors += [f"expert name used {n} times: {k}" for k, n in expert_names.items() if n > 1]
    errors += [f"entity name used {n} times: {k}" for k, n in entity_names.items() if n > 1]
    return errors, warnings


def validate_paraphrases(areas, pool, n_paraphrases=4, max_tokens=30):
    """Mechanical checks on the paraphrase pool against areas.json. Returns (errors, warnings).

    Statements follow the same rules as the originals (name before the answer, answer last and only
    once) and must not mention the entity's other answers. Each fact's 5 wordings (original + 4) must
    end their lead-in to the answer with different last two words, so no wording shares the context
    right before the answer.
    """
    errors, warnings = [], []
    for a in areas:
        aid = a["area_id"]
        area_pool = pool.get(aid)
        if area_pool is None:
            errors.append(f"{aid}: no paraphrase file")
            continue
        names = {e["name"] for ex in a["experts"] for e in ex["entities"]}
        errors += [f"{aid}/{name}: entity not in areas.json" for name in sorted(set(area_pool) - names)]
        for ex in a["experts"]:
            for e in ex["entities"]:
                name = e["name"]
                for f in e["facts"]:
                    fid = f"{aid}/{name}/{f['relation_id']}"
                    p = area_pool.get(name, {}).get(f["relation_id"])
                    if p is None:
                        errors.append(f"{fid}: missing")
                        continue
                    ans = f["answer"].strip()
                    others = [g["answer"].strip() for g in e["facts"] if g is not f
                              and not contains_tokens(ans, g["answer"], ignore_case=True)]
                    sts, qs = p.get("statements", []), p.get("questions", [])
                    if len(sts) != n_paraphrases or len(qs) != n_paraphrases:
                        errors.append(f"{fid}: {len(sts)} statements and {len(qs)} questions, expected {n_paraphrases} each")
                    for kind, texts, original in (("statement", sts, f["statement"]), ("question", qs, f["question"])):
                        seen = [original.lower()] + [t.lower() for t in texts]
                        if len(set(seen)) != len(seen):
                            errors.append(f"{fid}: duplicate {kind}s (or one equals the original)")
                    leads = Counter()
                    for st in [f["statement"]] + sts:
                        lead = st[: -len(ans) - 1].rstrip() if st.endswith(f" {ans}.") else st
                        leads[tuple(t.lower() for t in tokenize(lead)[-2:])] += 1
                    for st in sts:
                        if not st.endswith(f" {ans}."):
                            errors.append(f"{fid}: statement must end with ' {ans}.': {st!r}")
                            continue
                        lead = st[: -len(ans) - 1].rstrip()
                        if contains_tokens(lead, ans, ignore_case=True):
                            errors.append(f"{fid}: answer also appears before the end: {st!r}")
                        if name not in lead:
                            errors.append(f"{fid}: entity name missing before the answer: {st!r}")
                        if leads[tuple(t.lower() for t in tokenize(lead)[-2:])] > 1:
                            errors.append(f"{fid}: lead-in ends like another wording ({' '.join(tokenize(lead)[-2:])!r}): {st!r}")
                        if "?" in st or "Question" in st or "Answer" in st:
                            errors.append(f"{fid}: statement reads like a question: {st!r}")
                        leaked = [o for o in others if contains_tokens(lead, o, ignore_case=True)
                                  and not contains_tokens(f["statement"], o, ignore_case=True)]
                        if leaked:
                            errors.append(f"{fid}: mentions the entity's other answers {leaked}: {st!r}")
                    for q in qs:
                        if not q.endswith("?") or name not in q or q.startswith("Question"):
                            errors.append(f"{fid}: question must end with '?', contain {name!r}, and have no prefix: {q!r}")
                        if contains_tokens(q, ans, ignore_case=True):
                            errors.append(f"{fid}: question contains the answer: {q!r}")
                        leaked = [o for o in others if contains_tokens(q, o, ignore_case=True)
                                  and not contains_tokens(f["question"], o, ignore_case=True)]
                        if leaked:
                            errors.append(f"{fid}: question mentions the entity's other answers {leaked}: {q!r}")
                    for t in sts + qs:
                        if BANNED_CHARS & set(t):
                            errors.append(f"{fid}: banned characters {sorted(BANNED_CHARS & set(t))}: {t!r}")
                        if len(tokenize(t)) > max_tokens:
                            errors.append(f"{fid}: {len(tokenize(t))} tokens: {t!r}")
                    warnings += [f"{fid}: statement has {len(st.split())} words: {st!r}"
                                 for st in sts if not 6 <= len(st.split()) <= 20]
    return errors, warnings


class BM25:
    """Okapi BM25 over lowercased word tokens (punctuation dropped)."""

    def __init__(self, texts, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.docs = [Counter(self._terms(t)) for t in texts]
        self.lengths = [sum(c.values()) for c in self.docs]
        self.avg_len = sum(self.lengths) / len(self.lengths)
        df = Counter(term for c in self.docs for term in c)
        n = len(self.docs)
        self.idf = {term: math.log(1 + (n - f + 0.5) / (f + 0.5)) for term, f in df.items()}

    @staticmethod
    def _terms(text):
        return [t.lower() for t in tokenize(text) if re.match(r"\w", t)]

    def scores(self, query):
        q_terms = self._terms(query)
        out = []
        for c, length in zip(self.docs, self.lengths):
            norm = self.k1 * (1 - self.b + self.b * length / self.avg_len)
            out.append(sum(
                self.idf[t] * c[t] * (self.k1 + 1) / (c[t] + norm)
                for t in q_terms if t in c
            ))
        return out
