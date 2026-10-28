# Toy dataset (v3: real facts + paraphrases)

Real-world facts for the attribution toy experiment, step 1 of the roadmap. The model is trained from
scratch on this data only, so we know exactly which training document taught it each answer.

v3 tells each entity's facts k ~ Uniform{1..5} times in different wordings (knowledge augmentation,
see below), because v2's single wording per fact let the model memorize sentences without being able
to answer questions about them. v2 is still in `data/` and is reproduced exactly by `--max-copies 1`.
v1 (invented entities and templates) is kept for reference: its data is in `legacy/toy_v1/`, its spec
in `dataset/domains.json`, and the generator that produced it in commit `6c6870c`.

## Regenerate

```
python dataset/make_dataset.py --out data_v3 --seed 0
python dataset/check_dataset.py data_v3
```

The same seed gives byte-identical files. `--qa-fraction` (default 0.5) sets the share of entities
per area that get question-answer documents, `--max-copies` (default 5) the largest k.

To train on v3, give it its own run directory so the v2 tokenizer is not reused:
`DATA_DIR=data_v3 OUT_DIR=runs/toy_v3 bash toy/train.sh`.

**Training:** answers can now be several tokens long, but `toy/tokenizer.py` and `toy/train.py` still
require one-token answers, so `bash toy/train.sh` (which reads `data/` by default) stops on this data. The training check below used a
patched copy (multi-token answers; a query counts as recalled when every answer token is predicted
correctly). The patch is not in the repo yet.

## What's in it

| | v2 (`data/`) | v3 (`data_v3/`) |
|---|---|---|
| Topic areas | 30 | 30 |
| Authors (experts) | 90 (3 per area, fictional) | 90 |
| Entities | 450 (5 per expert, real) | 450; k = 1/2/3/4/5 for 83/91/84/105/87 of them |
| Facts | 1,350 (3 per entity) | 1,350 |
| Documents | 1,528 (898 statement, 630 question-answer) | 4,671 (2,712 statement, 1,959 question-answer) |
| Sentences | 1,980 | 6,075 |
| Training tokens | 24,438 | 79,974 (3.3×) |
| Vocabulary (word-level) | 2,346 | 2,868 |
| Queries | 200 (100 facts × 2 forms); 39 facts have multi-word answers | same 200; query facts by k: 23/12/23/23/19 |

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
- **Knowledge augmentation (v3).** Physics of LM Part 3.1 found that facts stated in one wording are
  memorized but cannot be extracted by questions, while several wordings plus shuffled sentence order
  raised held-out question accuracy from 9.7% to 96.6%. So each entity draws k ~ Uniform{1..5}:
  - Its facts are told k times: copy 0 is v2's document exactly, copies 1..k-1 use paraphrases from
    `dataset/paraphrases/` (4 per fact, one file per area), regrouped into 1-3 new documents with the
    sentences reshuffled. Example (Buyeo, k=3): `Buyeo County lies in South Chungcheong Province.` /
    `Buyeo is a county in South Chungcheong Province.` / `Administratively, Buyeo County belongs to
    South Chungcheong Province.`, each in a different document with different neighbours.
  - Train entities get k question wordings per fact as well; test entities still get none.
  - Every copy is written by the entity's expert. A query's ground truth (`source_doc_ids`) is now
    its k statement documents, copy 0 first.
  - All facts are augmented, queried ones included, so the 23 query facts with k=1 measure whether
    augmenting the others helps them too, and k≥2 measures the direct effect.
  - k and the copies come from their own random stream, so v2's splits, queries and original
    documents are unchanged.
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

### Paraphrases (v3)

1. **Write.** 6 agents, 5 areas each, wrote 4 statement and 4 question paraphrases per fact (10,800
   strings) under these rules: the statement ends with the exact answer, names the entity before it, states
   only that fact with no added claim, and no two of the fact's 5 wordings share the last two words
   before the answer; a question has exactly one correct answer and never contains it.
2. **Mechanical check.** `validate_paraphrases` in `common.py` enforces every rule above that can be
   checked by code (and that no paraphrase mentions the entity's other answers). All passed.
3. **Cross-review.** Each agent re-read another agent's 5 areas for meaning: 45 strings were fixed,
   mostly subtle overclaims (`originally invented by`, `the stomach is mainly responsible for
   digesting`, `the Romans equated Hades with their own deity Pluto`) and questions with a second
   valid answer (`Where does Hamlet unfold?` also fits Elsinore).

## Files (`data/`)

- `docs.jsonl`: `doc_id` (int), `text`, `author_id`, `area_id`, `entity_id`, `doc_type`
  (`statement` or `qa`), `fact_ids`, `copy` (0 = original wording).
- `queries.jsonl`: `query_id` (int), `form` (`cloze` or `question`), `prompt`, `answer`,
  `answer_n_tokens`, `fact_id`, `entity`, `area_id`, `relation_id`, `n_copies` (k),
  `source_doc_ids` (ground truth, one per copy), `source_author_ids`, `fact_source`.
- `facts.jsonl`: every fact with its answer, statement, question, source, split, documents, `n_copies`,
  and the `statements` / `questions` wordings used (original first).
- `entities.jsonl`, `authors.json` (name, specialty, area), `vocab.json`, `stats.json`, `config.json`.

Document IDs run area → entity, and queries are sorted the same way, so a query × document heatmap
shows the ground truth as a diagonal.

## What `check_dataset.py` verifies

It re-derives each property from the document text instead of trusting the generator's metadata:
each of a fact's k wordings appears in exactly one document (one per copy), paraphrases not drawn for
a fact appear nowhere, every document fits the 64-token context, test entities have no question-answer
documents in any wording, each cloze prompt plus its answer appears in its source, and every document
is written by its entity's expert. It also re-runs the paraphrase rules and warns when another document
mentions both a fact's entity and its answer (see limitations). `--verbose` lists every problem.

## BM25 baseline

Top-1 = the best-ranked document is the source (ties count in BM25's favor):

| Queries | n | top-1, prompt only (v2 / v3) | top-1, prompt + answer (v2 / v3) |
|---|---|---|---|
| cloze | 100 | 1.00 / 1.00 | 1.00 / 1.00 |
| question | 100 | 0.46 / 0.54 | 0.71 / 0.74 |

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
- v3 has not been trained on yet; the training check above is v2's.
- Paraphrases reuse a few sentence frames per relation across entities (within a fact all 5 wordings
  differ). Some original questions stay ambiguous: Yonsei's founding (1885 roots vs 1957 merger),
  Roman counterparts (Liber, Dis Pater), "famous work" for authors with several, Pyeongchang vs
  Gangneung. These are flagged, not changed.
- Real facts overlap: another document mentions both a fact's entity and its answer (e.g. the 1988
  Olympics document mentions South Korea and Seoul). v2 has 16 such (fact, document) pairs over 11
  facts, v3 40 pairs over 14 facts: copies repeat the old overlaps and, by regrouping an entity's
  facts, add 3 (Waseda's `Tokyo` and `Japan` sentences now share a document, which flags Japan → Tokyo). One is a queried fact: lungs →
  chest also appears in the rib cage document ("protecting the heart and lungs"), so that document
  may legitimately get some credit.
- Some facts about one entity give each other away (chickenpox's pathogen, the varicella-zoster
  virus, also reveals its pathogen type, virus).
- Cloze recall is not yet the ~100% that Section 5.1 targets; not investigated.
