# Retrace

*Everything AI learned has an owner.*

Data attribution for generative models: trace a model's output back to the
training data that produced it, and turn each document's contribution into a
royalty for its creator.

## Status

Building the first toy experiment: TRAK-style attribution on a small
Transformer trained from scratch on synthetic data, with visualizations.
See `prompt_for_claude.md` for the full project context and design notes.

- [Toy dataset](dataset/README.md): 30 topic areas, 1,350 real, fact-checked facts, 200 queries
  (`dataset/make_dataset.py`, checked by `dataset/check_dataset.py`). v3 (`data_v3/`) adds 1–5
  paraphrased copies per entity (knowledge augmentation); v2 is in `data/`.
