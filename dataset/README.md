# Toy dataset (v1)

Synthetic facts for the attribution toy experiment, step 1 of the roadmap. Every fact is invented,
so we know exactly which training document taught the model each answer.

## Regenerate

```
python dataset/make_dataset.py --out data/toy_v1 --seed 0
python dataset/check_dataset.py data/toy_v1
```

The same seed gives byte-identical files. `--entities-per-domain 72` scales to about 9.8k sentences
(7.7k documents, 122k tokens); all checks still pass at that size.

Train on it with the toy pipeline: `DATA_DIR=data/toy_v1 bash toy/train.sh`.

## What's in it

| | v1 (default) |
|---|---|
| Topic areas | 30 |
| Authors (experts) | 90 (3 per area) |
| Entities | 480 (16 per area) |
| Facts | 1,440 (3 attributes per entity) |
| Documents | 1,761 |
| Sentences | 2,240 |
| Training tokens | 27,924 |
| Vocabulary (word-level) | 1,831 |
| Queries | 200 (100 facts × 2 forms) |

## Design

- **Topic areas and authors.** 30 areas (astronomy, cuisine, mythology, ...), each with 3 expert
  authors. Authors are metadata only and never appear in the text.
- **Entities.** 16 fictional entities per area, each with 3 attributes (e.g. a star's constellation,
  color, and dominant spectral element). Names are random syllables (`Gianlos`, `Kranvrem`), one token
  each, and never collide with any other word in the corpus.
- **Statement documents.** An entity's 3 facts are split into 1–3 documents of 1–3 sentences, each by
  one of the area's experts. Every fact is stated in exactly one document.
- **Question-answer documents.** Half the entities in each area (`qa_split: train`) also get one
  `Question: ... Answer: ...` document per fact, so the model can learn the question format. Queries
  only ask about the other half (`test`). This follows the mixed-training setup in Physics of LM
  Part 3.1 (arXiv 2309.14316): models trained only on statements memorize facts but can't answer
  questions about them, while adding QA for half the individuals reached 86.6% on the other half.
- **Queries.** 100 facts, each asked two ways about the same fact:
  - cloze: `The spectrum of the star Liartra is dominated by lines of` → `neon`
  - question: `Question: Which element dominates the spectrum of the star Liartra? Answer:` → `neon`
- **Planted structures** (per queried fact):
  - `single` (60): one source document.
  - `conflict` (20): 3 documents state the majority value and 1 states a different one, all in the
    query's wording. The majority documents are the bundled document by expert A and one-sentence
    copies by experts B and A. The minority document is by expert C. The expected answer is the
    majority value.
  - `hard_negative` (20): one extra document by a different expert that mentions the entity, the
    answer, and the attribute's key word without stating the fact, e.g. `A lecture on the spectrum of
    neon was given at the observatory the night Liartra was catalogued.` Use `--n-hard-negative 0` to
    turn these off.
- **Not included:** lineage (original + derivative copies), skipped for the Oct 23 pitch.

## Files (`data/toy_v1/`)

- `docs.jsonl`: `doc_id` (int), `text`, `author_id`, `domain_id`, `entity_id`, `doc_type`
  (`statement`, `qa`, `conflict_majority`, `conflict_minority`, `hard_negative`), `claims` (facts
  the document states and the value it gives), `hard_negative_for`.
- `queries.jsonl`: `query_id` (int), `form` (`cloze` or `question`), `prompt`, `answer`, `fact_id`,
  `structure`, `source_doc_ids` (ground truth), `source_author_ids`, `contradicting_doc_ids`,
  `minority_value`, `hard_negative_doc_ids`.
- `facts.jsonl`: every fact with its value, split, structure, and the documents that state it.
- `entities.jsonl`, `authors.json`, `vocab.json` (tokens and counts), `stats.json`, `config.json`
  (arguments and a hash of `domains.json`).

Document IDs run area → entity, and queries are sorted the same way, so a query × document heatmap
shows the ground truth as a diagonal.

## What `check_dataset.py` verifies

It re-derives each property from the document text instead of trusting the generator's metadata:
every fact is stated in exactly one document (conflicts: 3 + 1), names and answers are single tokens,
test entities have no QA documents, each cloze prompt plus its answer appears in its source documents,
hard negatives never contain the statement wording, and authors are assigned as described above.

## BM25 baseline

From `check_dataset.py` (top-1 = the best-ranked document is a ground-truth source; ties count in
BM25's favor):

| Queries | n | top-1, prompt only | top-1, prompt + answer |
|---|---|---|---|
| cloze, single | 60 | 0.85 | 0.87 |
| cloze, hard negative | 20 | 0.70 | 0.85 |
| cloze, conflict | 20 | 1.00 | 1.00 |
| question, single | 60 | 0.10 | 0.30 |
| question, hard negative | 20 | 0.00 | 0.15 |
| question, conflict | 20 | 0.00 | 0.20 |

What this means for the attribution comparison:

- **Cloze queries are easy for keyword search.** The entity name appears in very few documents, so
  BM25 finds the source about 85% of the time. Attribution has to at least match this.
- **Conflicts.** With the cloze prompt alone, BM25 scores the minority document at least as high as
  the majority ones in all 20 cases (9 of 20 for questions); the one-sentence copies differ only in
  the value. Once the model's answer is added to the search, BM25 ranks every majority document
  above the minority one, for both forms. So the useful contrast is the sign: attribution should
  give the minority document a negative score, because it pushes against the answer, and BM25 can't
  express that.
- **Questions are hard for keyword search.** For 75 of 100 question queries, BM25's top document is
  a question-answer document about a different entity, which shares the question wording. If
  attribution finds the statement document here, that is the clearest "contribution, not surface
  similarity" result in the dataset.
- **Hard negatives are only moderately hard for BM25.** With the answer included, it ranks them
  around 10th of 1,761 (best 2nd, worst 30th), and above the source in only 1 of 40 queries. They are
  better read as a precision check: attribution should give them close to zero.

## Training check (one seed)

`toy/train.sh` defaults (4 layers, d=256, 4.1M parameters, 300 epochs, about 2 minutes on an H100),
seed 0. Recall = the model's top next-token prediction is the answer; chance is about 1 in 14.

| Queries | n | recall | p(answer) |
|---|---|---|---|
| cloze, single | 60 | 0.87 | 0.86 |
| cloze, hard negative | 20 | 0.95 | 0.94 |
| cloze, conflict | 20 | 1.00 | 0.72 (minority 0.28) |
| question, single | 60 | 0.47 | 0.41 |
| question, hard negative | 20 | 0.65 | 0.62 |
| question, conflict | 20 | 0.75 | 0.57 (minority 0.31) |
| **cloze, all** | 100 | **0.91** | |
| **question, all** | 100 | **0.56** | |

- The model answers questions about entities it only saw as statements well above chance, so the
  mixed-training setup works at this scale, though far below the 86.6% reported for GPT-2-sized models.
- One seed only. An earlier run on a version that differed only in the conflict documents gave 0.64
  on questions, so expect several points of run-to-run variation.
- Cloze recall is not yet the ~100% that Section 5.1 targets. That is probably a training setting
  (more epochs or a higher learning rate) rather than the data; not yet checked.

## Simplifications and known limitations

- Question accuracy is moderate (about 56% on one seed). To raise it, give more entities QA documents
  or paraphrase the statements; paraphrasing would break "exactly one source document".
- Little linguistic variety: each attribute has 2 statement wordings, 1 question wording, and 1
  hard-negative wording.
- In conflicts, the two one-sentence majority copies are identical, and the minority document differs
  from them only in the value.
- Values are drawn uniformly from 13–16 options per attribute, so some values appear in only one
  or two facts.
- The 30 area specs (`domains.json`) were drafted and adversarially reviewed by LLM agents, then
  checked mechanically by `validate_domains()` and read through by hand.
