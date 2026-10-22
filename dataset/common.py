"""Shared helpers for the Retrace toy dataset: tokenizer, domain-spec checks, BM25."""
import json
import math
import re
from collections import Counter
from pathlib import Path

# Word-level tokenizer: a token is a run of ASCII letters or one punctuation mark.
# Entity names and values are letters-only, so each one is exactly one token.
TOKEN_RE = re.compile(r"[A-Za-z]+|[.,?:]")
TEMPLATE_CHARS_RE = re.compile(r"^[A-Za-z ,]+$")
BANNED_HARD_NEGATIVE_WORDS = {
    "like", "unlike", "same", "similar", "as", "not", "never", "no", "different", "instead",
}
STOPWORDS = {
    "the", "a", "an", "of", "in", "on", "at", "to", "for", "by", "with", "from", "and", "or",
    "is", "was", "are", "were", "be", "been", "its", "it", "this", "that", "which", "what",
    "where", "who", "whom", "how", "does", "do", "did", "has", "have", "had", "into", "their",
}


def tokenize(text):
    return TOKEN_RE.findall(text)


def untokenizable(text):
    """Characters the tokenizer would silently drop (anything but tokens and spaces)."""
    return TOKEN_RE.sub("", text).replace(" ", "")


def load_domains(path):
    return json.loads(Path(path).read_text())["domains"]


def fill(template, entity, value=None):
    out = template.replace("{entity}", entity)
    return out if value is None else out.replace("{value}", value)


def cloze_prefix(template, entity):
    """Statement text up to (not including) the value, e.g. 'Velkora lies in the constellation'."""
    return fill(template.split("{value}")[0], entity).strip()


def template_words(domain):
    """Lowercased words used by a domain's templates, placeholders removed."""
    words = set()
    for rel in domain["relations"]:
        for t in rel["statement_templates"] + [rel["question_template"], rel["hard_negative_template"]]:
            stripped = t.replace("{entity}", " ").replace("{value}", " ")
            words.update(w.lower() for w in re.findall(r"[A-Za-z]+", stripped))
    return words


def validate_domains(domains):
    """Mechanical checks on the domain spec. Returns (errors, warnings) as lists of strings."""
    errors, warnings = [], []
    ids = [d["domain_id"] for d in domains]
    if len(set(ids)) != len(ids):
        errors.append(f"duplicate domain_id in {ids}")

    for d in domains:
        did = d["domain_id"]
        words = template_words(d)
        if len(d["relations"]) != 3:
            errors.append(f"{did}: expected 3 relations, got {len(d['relations'])}")
        rel_ids = [r["relation_id"] for r in d["relations"]]
        if len(set(rel_ids)) != len(rel_ids):
            errors.append(f"{did}: duplicate relation_id")

        prefixes = {}
        seen_values = {}
        for r in d["relations"]:
            rid = f"{did}/{r['relation_id']}"
            stmts = r["statement_templates"]
            if len(stmts) != 2:
                errors.append(f"{rid}: expected 2 statement templates")
            if len(set(stmts)) != len(stmts):
                errors.append(f"{rid}: statement templates are identical")
            for t in stmts:
                if t.count("{entity}") != 1 or t.count("{value}") != 1:
                    errors.append(f"{rid}: statement needs one {{entity}} and one {{value}}: {t!r}")
                    continue
                if not t.endswith("{value}."):
                    errors.append(f"{rid}: statement must end with '{{value}}.': {t!r}")
                if t.index("{entity}") > t.index("{value}"):
                    errors.append(f"{rid}: {{entity}} must come before {{value}}: {t!r}")
                body = t[:-1].replace("{entity}", "X").replace("{value}", "Y")
                if not TEMPLATE_CHARS_RE.match(body):
                    errors.append(f"{rid}: bad characters in statement: {t!r}")
                n_words = len(re.findall(r"[A-Za-z]+", body))
                if not 6 <= n_words <= 14:
                    warnings.append(f"{rid}: statement has {n_words} words: {t!r}")
                prefix = t.split("{value}")[0].strip().lower()
                if prefix in prefixes and prefixes[prefix] != r["relation_id"]:
                    errors.append(f"{rid}: prefix shared with {prefixes[prefix]}: {prefix!r}")
                prefixes[prefix] = r["relation_id"]

            q = r["question_template"]
            if q.count("{entity}") != 1 or "{value}" in q or not q.endswith("?"):
                errors.append(f"{rid}: question needs one {{entity}}, no {{value}}, ends with '?': {q!r}")
            elif not TEMPLATE_CHARS_RE.match(q[:-1].replace("{entity}", "X")):
                errors.append(f"{rid}: bad characters in question: {q!r}")

            h = r["hard_negative_template"]
            if h.count("{entity}") != 1 or h.count("{value}") != 1 or not h.endswith("."):
                errors.append(f"{rid}: hard negative needs one {{entity}}, one {{value}}, ends with '.': {h!r}")
            else:
                if h.endswith("{value}."):
                    errors.append(f"{rid}: hard negative must not end with {{value}}: {h!r}")
                body = h[:-1].replace("{entity}", "X").replace("{value}", "Y")
                if not TEMPLATE_CHARS_RE.match(body):
                    errors.append(f"{rid}: bad characters in hard negative: {h!r}")
                hn_words = {w.lower() for w in re.findall(r"[A-Za-z]+", body)}
                banned = hn_words & BANNED_HARD_NEGATIVE_WORDS
                if banned:
                    errors.append(f"{rid}: hard negative uses banned words {sorted(banned)}: {h!r}")
                stmt_content = {
                    w.lower() for t in stmts
                    for w in re.findall(r"[A-Za-z]+", t.replace("{entity}", " ").replace("{value}", " "))
                    if len(w) >= 3 and w.lower() not in STOPWORDS
                }
                if not hn_words & stmt_content:
                    errors.append(f"{rid}: hard negative shares no content word with statements: {h!r}")

            vals = r["values"]
            if not 12 <= len(vals) <= 16:
                errors.append(f"{rid}: expected 12-16 values, got {len(vals)}")
            if len({v.lower() for v in vals}) != len(vals):
                errors.append(f"{rid}: duplicate values")
            for v in vals:
                if not re.fullmatch(r"[A-Za-z]+", v):
                    errors.append(f"{rid}: value is not a single letters-only word: {v!r}")
                if v.lower() in words:
                    errors.append(f"{rid}: value {v!r} also appears as a template word")
                if v.lower() in seen_values and seen_values[v.lower()] != r["relation_id"]:
                    errors.append(f"{rid}: value {v!r} also used by {seen_values[v.lower()]}")
                seen_values[v.lower()] = r["relation_id"]
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
        return [t.lower() for t in tokenize(text) if t.isalpha()]

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
