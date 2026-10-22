# Toy dataset (v2: real facts)

Real-world facts for the attribution toy experiment, step 1 of the roadmap. The model is trained from
scratch on this data only, so we know exactly which training document taught it each answer.

v1 (invented entities and templates) is kept for reference: its data is in `legacy/toy_v1/`, its spec
in `dataset/domains.json`, and the generator that produced it in commit `6c6870c`.

## Regenerate

```
python dataset/make_dataset.py --out data --seed 0
python dataset/check_dataset.py data
```

The same seed gives byte-identical files. `--qa-fraction` (default 0.5) sets the share of entities
per area that get question-answer documents.

**Training:** answers can now be several tokens long, but `toy/tokenizer.py` and `toy/train.py` still
require one-token answers, so `bash toy/train.sh` (which reads `data/` by default) stops on this data. The training check below used a
patched copy (multi-token answers; a query counts as recalled when every answer token is predicted
correctly). The patch is not in the repo yet.

## What's in it

| | v2 |
|---|---|
| Topic areas | 30 |
| Authors (experts) | 90 (3 per area, fictional) |
| Entities | 450 (5 per expert, real) |
| Facts | 1,350 (3 per entity) |
| Documents | 1,528 (898 statement, 630 question-answer) |
| Sentences | 1,980 |
| Training tokens | 24,438 |
| Vocabulary (word-level) | 2,346 |
| Queries | 200 (100 facts × 2 forms); 39 facts have multi-word answers |

## Design

- **Topic areas:** Geography of South Korea, Java OOP, Shakespeare's plays, K-pop songs, Stars,
  Chemical elements, Countries, Human anatomy, Classical music, Famous paintings, Olympic Games,
  Football clubs, Korean history, Inventions, Nobel Prize in Physics, Mammals, Dinosaurs, Rivers,
  Mountains, Programming languages, Korean films, Novels, Greek mythology, Moons of the Solar System,
  Famous landmarks, Video games, Technology companies, Deep learning milestones, Diseases, Universities.
- **Experts and entities.** Each area has 3 fictional experts (metadata only, never in the text), and
  each expert owns 5 real entities. Each entity has 3 facts, one per area attribute (e.g. a K-pop
  song's artist, release year, and album).
- **Statement documents.** An entity's 3 facts are split into 1–3 documents written by its expert.
  Every fact is stated in exactly one document. Each sentence states one fact and ends with its
  answer, e.g. `Andong is a city in North Gyeongsang Province. The city of Andong lies along the
  Nakdong River.`
- **Question-answer documents.** Half the entities in each area also get one
  `Question: ... Answer: ...` document per fact, so the model can learn the question format. Queries
  only ask about the other half (mixed training, Physics of LM Part 3.1, arXiv 2309.14316).
- **Queries.** 100 facts, each asked two ways:
  - cloze: `Buyeo County lies in` → `South Chungcheong Province`
  - question: `Question: Which province is Buyeo located in? Answer:` → `South Chungcheong Province`
- **Not included:** conflicting facts, hard negatives, and lineage. Every sentence is meant to be true.

## How the facts were written and checked

1. **Write.** 6 agents, 5 areas each (15 area-expert pairs per agent), from their own knowledge plus
   web search when unsure.
2. **Fact-check.** 6 adversarial agents re-checked every fact (566 by web search). They corrected 5
   answers, replaced 28 entities, and reworded 88 sentences.
3. **Independent audit of all 1,350 facts.** 0 wrong, 13 ambiguous. 752 facts were confirmed by a
   fresh web search; the rest were confirmed from the auditor's own knowledge (its search budget ran
   out), on top of the source recorded in step 1 or 2.
4. **Fixes for the 13 ambiguous facts.**
   - Reworded: Lotus Temple is in `Delhi` (not strictly New Delhi); TD-Gammon was "first described in
     a published paper in 1992" (versions existed in 1991); Charles University "received its founding
     charter in 1348" (the papal bull is from 1347).
   - Kept: Ganymede's discovery "is credited to" Galileo, which is accurate despite Simon Marius's claim.
   - Titles shared with other works (e.g. Fake Love is also a Drake song): every sentence and question in
     K-pop, Korean films, novels, classical music, and video games now names the category ("the K-pop
     song Fake Love", "the oratorio Messiah"). 10 titles still ambiguous within their category were
     replaced and their facts re-verified by web search: Maniac, Spicy, Psycho, Cheer Up, Blindness,
     Tunnel, Silenced, Magic Flute, Lohengrin, Scheherazade.

Every fact records its source in `facts.jsonl`: 1,101 cite a URL and 249 say `knowledge`. This lowers
the error rate but does not guarantee 100% accuracy.

## Files (`data/`)

- `docs.jsonl`: `doc_id` (int), `text`, `author_id`, `area_id`, `entity_id`, `doc_type`
  (`statement` or `qa`), `fact_ids`.
- `queries.jsonl`: `query_id` (int), `form` (`cloze` or `question`), `prompt`, `answer`,
  `answer_n_tokens`, `fact_id`, `entity`, `area_id`, `relation_id`, `source_doc_ids` (ground truth),
  `source_author_ids`, `fact_source`.
- `facts.jsonl`: every fact with its answer, statement, question, source, split, and documents.
- `entities.jsonl`, `authors.json` (name, specialty, area), `vocab.json`, `stats.json`, `config.json`.

Document IDs run area → entity, and queries are sorted the same way, so a query × document heatmap
shows the ground truth as a diagonal.

## What `check_dataset.py` verifies

It re-derives each property from the document text instead of trusting the generator's metadata:
every fact's sentence appears in exactly one document, every document fits the 64-token context, test
entities have no question-answer documents, each cloze prompt plus its answer appears in its source,
and every document is written by its entity's expert. It also warns when another document mentions
both a fact's entity and its answer (see limitations).

## BM25 baseline

Top-1 = the best-ranked document is the source (ties count in BM25's favor):

| Queries | n | top-1, prompt only | top-1, prompt + answer |
|---|---|---|---|
| cloze | 100 | 1.00 | 1.00 |
| question | 100 | 0.46 | 0.71 |

- **Cloze queries are trivial for keyword search.** With real names and natural sentences, the prompt
  appears almost word for word in its source. Attribution can only match BM25 here.
- **Questions are where attribution can win.** With the prompt only, BM25's top document is another
  entity's question-answer document 47% of the time.

## Training check (one seed)

Default `toy/` settings (4 layers, d=256, 300 epochs, about 100 seconds on an H100), with the
multi-token patch:

| Queries | n | recall | p(answer) |
|---|---|---|---|
| cloze, one-token answer | 61 | 0.92 | 0.90 |
| cloze, multi-token answer | 39 | 0.87 | 0.87 |
| **cloze, all** | 100 | **0.90** | 0.89 |
| question, one-token answer | 61 | 0.38 | 0.32 |
| question, multi-token answer | 39 | 0.18 | 0.16 |
| **question, all** | 100 | **0.30** | 0.26 |

- Questions are much harder than in v1 (0.56), probably because each entity's sentences and
  questions are worded differently, unlike v1's templates.
- Training changes didn't help (on the data before the title fixes): 600 epochs gave 0.20, learning
  rate 3e-3 gave 0.27. QA documents for 2/3 of the entities gave 0.31 (mean of 2 seeds) versus 0.26
  for 1/2, which is within seed-to-seed noise.

## Known limitations

- About 30% question recall: most question queries will be attributing wrong answers. The cloze
  queries (90%) are the reliable set for the pitch.
- Real facts overlap. 16 facts have another document that mentions both the entity and the answer
  (e.g. the 1988 Olympics document mentions South Korea and Seoul). One is a queried fact: lungs →
  chest also appears in the rib cage document ("protecting the heart and lungs"), so that document
  may legitimately get some credit.
- Some facts about one entity give each other away (chickenpox's pathogen, the varicella-zoster
  virus, also reveals its pathogen type, virus).
- Cloze recall is not yet the ~100% that Section 5.1 targets; not investigated.
