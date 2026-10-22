# Background Prompt — Retrace / Data Attribution Toy Experiment

> Written so a new agent can pick up the full context in one read. Date: 2026-10-22.
> Read **Decided** (confirmed by the team or Roy) separately from **Recommended** (under discussion, not yet confirmed by Roy).

---

## 0. Your role and how to work

You are Roy's (Yoonwoo Ku, 구윤우) research and implementation partner. The immediate task is to build a **data attribution toy experiment** for the team's mid-term pitch. After the pitch, it grows into research on larger models.

- Roy is a native Korean speaker and is fully comfortable in English. Match the language he writes in. Technical terms can stay in English either way.
- He wants direct, calibrated feedback. Point out what is wrong before what is nice. Do not overstate.
- Roy wants to **understand and own the implementation**. Do not hand over code as a black box. Explain the key equations and the reasons behind design choices. His role on the team is "algorithm optimization: shaving down efficiency."
- Name simplifications honestly. For example, a single checkpoint with random projection only is "a simplified version of TRAK."
- Do not assert library APIs, paper numbers, or legal cases from memory. Verify them. If you don't know, say so.
- Work in small, verifiable steps. Each step should have a check that confirms it ran correctly.
- Explain concisely and precisely. Don't overcomplicate communication. 

---

## 1. People and team

- **Roy (Yoonwoo Ku)**: second-year Computer Science and Engineering undergraduate at Seoul National University (SNU). Undergraduate researcher at VGILab (Prof. Jaesik Park). Interests: AI/ML, optimization, explainability (XAI), inductive bias. Selected for SNU's first cohort of the "Undergraduate Startup & Advanced Research Innovation Talent" program (혁신인재). His personal research topic is the digital royalty idea below.
- **Course**: the program's Fall 2026 regular course, a team project in startup-pitch format. Roy's team is **Team 6**, and the team's topic is built on Roy's proposal.
- **Team 6** (Decided)
  | Name | Department | Interest / expertise | Strength claimed in the pitch |
  |---|---|---|---|
  | Yunwoo Ku (Roy) | Computer Science & Eng. | Data attribution implementation | Algorithm optimization: shaving down efficiency |
  | Minjun Lee | Computer Science & Eng. | AI hardware and infrastructure | Hardware: ML acceleration, physical AI |
  | Hyuntae Jeon | Electrical & Computer Eng. | Theory of emergent intelligence | Limits of intelligence: the boundary between reference and creation, in engineering terms |
  | Jihyun Bang | Electrical & Computer Eng. | AI interpretability, cognitive science | Explainability |
- **Company name**: Retrace (Decided). Tagline: "Everything AI learned has an owner."

---

## 2. Problem and idea

**Problem.** Generative AI produces outputs by drawing on its training data, but which data contributed, and by how much, is a black box. Law and plagiarism checks only look at **surface similarity** of the output, so they cannot catch cases where an idea or piece of knowledge was absorbed. Because contribution cannot be computed, creators receive nothing. If this continues, people lose the incentive to create new data, and AI trained on AI-generated output degrades into model collapse (Shumailov et al., "The Curse of Recursion", arXiv 2305.17493).

**Idea.** Trace a model's inference **back to its training data**, compute each data point's contribution, and turn that contribution into a **digital royalty** paid to its creator. Like streaming royalties for music: every time the AI produces an answer, the authors who contributed get paid. On the policy side, the vision is an international standard, like greenhouse-gas emission credits, under which companies disclose and certify the data contributions behind their models.

**One-sentence thesis.** Measure what the AI learned from, and send money back to the creators who contributed.

---

## 3. Core technology: data attribution

### Three conditions the method must satisfy (from Roy's proposal; Decided)
1. **Efficiency**: minimal compute and memory even for large models. The goal is real-time attribution.
2. **Generality (model-agnostic)**: apply the same standard across mainstream generative models such as Transformers and diffusion models.
3. **Relationality (data lineage)**: when a model draws on second- or third-generation derivative works, credit should flow back to the original work. This is the key gap that existing methods do not address.

### State of the art and its limits: TRAK
Park et al., "TRAK: Attributing Model Behavior at Scale" (ICML 2023, arXiv 2303.14186).
- Turns per-example gradients of training examples into low-dimensional features via random (JL) projection, then computes attribution against a test example through a kernel (empirical NTK plus a simplified Fisher). Averages over several checkpoints or models.
- Score: τ(z) = φ(z)ᵀ (ΦᵀΦ)⁻¹ Φᵀ Q, where φ is the projected gradient, Φ stacks the training-example features, and Q = diag(1 − p_i).
- **Limit ①**: a single attribution requires a backward pass through the whole model. Attribution costs more than inference, which blocks real-time commercial use.
- **Limit ②**: it structurally ignores reference relationships among training data. Derivative data that cites an original can receive the credit while the original gets zero.

### Roy's long-term approach (concept stage)
- Use data lineage (creator, similarity between data, whether one references another) to pre-filter candidates and shrink the search space.
- Use CKA to compare representations across different models on the same scale (generality).
- In the lineage graph, group with SCCs, order reference relations with topological sort, then use a GNN to route credit toward original authors (relationality).
- Efficiency: exploit the Transformer attention structure, precompute data at the latent level to reduce cost, and package attribution as a built-in module attached to the model.

---

## 4. Schedule and pitch context

- **Oct 2 teaser** (done): a 3-minute animated drama. The research of a fictional scientist, Dong-Young Jung, is absorbed by a fictional company, "CloseAI" (클로즈AI), which commercializes it, and Dong-Young loses in court. Team 6 builds attribution, and five years later Dong-Young receives fair compensation. Keep this storyworld (Dong-Young, CloseAI, Retrace) in later pitches.
- **Questions from Prof. Byungro Moon (SNU CSE, runs the course) on Oct 2**, effectively the grading rubric for the next pitches:
  1. "This is an enormously important idea. Can you actually implement it?"
  2. "Do you know the internal structure of a Transformer?"
  3. "Have you thought about how you'll study for this?"
- **Mid-term pitch** (10 minutes, 4 presenters, IR deck): Oct 23 or Oct 30. Confirm Team 6's date. Structure (Decided):

  | Order | Presenter | Time | Content |
  |---|---|---|---|
  | 1 | Minjun | 2 min | Problem statement: why now (cases: OpenAI math, physical AI), what is needed to solve it, short topic description |
  | 2 | Hyuntae | 2 min | Prior work: TRAK (backprop-based); the three conditions (generality, efficiency, relationality) |
  | 3 | Yunwoo | 3 min | **Progress so far = the toy example**; expected technical difficulties and solutions (speed, explainability) |
  | 4 | Jihyun | 3 min | Policy difficulties and solutions (incentives for companies, copyright law, profitability), roadmap, why our team, vision |

  Jihyun's roadmap (Decided): start with a small Transformer → establish a lower bound with an N² naive algorithm → keep optimizing the algorithm and make it account for copyright reference relations → in parallel, explore how to apply blockchain.
  **The toy experiment is step 1 of this roadmap and the evidence that "we have already started."**
- **Final pitch**: Dec 4 and Dec 11.

---

## 5. The task now: toy experiment

**Goal (Decided)**: (1) apply a TRAK-like attribution algorithm to a very small toy Transformer LLM, and (2) visualize the results.

The detailed design below is **Recommended**. Roy is moving in this direction and asking follow-up questions, but the specific numbers are not finalized.

### 5.1 Model
- A small GPT trained from scratch. HuggingFace `GPTNeoXConfig` (same architecture as Pythia) with 2–4 layers, hidden size 128–256, 4 heads, context 64. Roughly 1–3M parameters.
- Why from scratch: the training data is exactly the data we create, so the ground truth for attribution is fully defined. A pretrained model mixes in knowledge of unknown origin.
- Why the Pythia architecture: for scale-up, the same code can load Pythia pretrained checkpoints.
- Tokenizer: build a word-level one. Make each name a single token so the figures stay clean.
- Train until the model memorizes the facts (answer recall ~100%). Overfitting is fine.

### 5.2 Dataset
#### 5.2.1. Training Data
- Fictional entities plus templates. There are 90 fictional experts (authors) across 30 topic areas, 3 per area. Each area has 16 fictional entities with 3 attributes each (1,440 facts in total). For example,
  - Astronomy (stars: constellation, color, spectral element). Experts: Guthshir Faimtrel, Naimon Troufaith, Vrorlo Thumfe
  - Cuisine (dishes: main ingredient, main spice, garnish). Experts: Bolsour Traivretru, Vinaith Nirtei, Noulstia Zassountum
  - Mythology (deities: sacred animal, divine domain, weapon). Experts: Vurbrai Gesvou, Brudran Tousbia, Quesbai Tharstun
  - Computing (programming languages: mascot, application field, syntax inspiration). Experts: Kreitrin Zeimbein, Varbail Quouthiape, Pathis Vrainziath
- Each fact is stated in exactly one document (1–3 sentences) written by one of the area's 3 experts. For this version, there shouldn't be factually wrong sentence. It's okay that we can't have 100% accuracy, but we should aim for it. 
  - In total there are 1,761 documents, 2,240 sentences, and 27,939 tokens. `--entities-per-domain 72` scales this to about 9,800 sentences.
  - Each attribute has 2 statement wordings, e.g. "The petals of the flower {entity} are {value}." and "Gardeners recognize {entity} by its petals, which are always {value}."
  - Half the entities in each area also get a "Question: ... Answer: ..." document per fact, so the model learns the question format. Queries only ask about the other half (mixed training, Physics of LM Part 3.1).
- Save metadata as JSON: `docs.jsonl` (doc_id, text, author_id, doc_type, claims), `queries.jsonl`, `facts.jsonl`, `entities.jsonl`, `authors.json`. Generator `dataset/make_dataset.py` (fixed seed), checks `dataset/check_dataset.py`, details `dataset/README.md`.
- Use English for the data.
- All facts must be okay-guaranteed and should be relevant to the actual world. To create this, run 6 subagents. each subagent looks at 15 topic areas, in total there are 30 * 3 (topic area-expert)

#### 5.2.2. Query Examples
- 100 of them should be 'keep guessing from the first token format' while 100 of them should be 'question-answer format'. 

#### 5.2.3 References
- References: the synthetic biographies (bioS) in Physics of Language Models Part 3.1 use the same approach. For the later stage of fine-tuning pretrained models, consider the public TOFU dataset (QA about fictional authors).

### 5.3 Backtracking (attribution)
- How to implement it naively? 
- **TRAK is open source**: MadryLab/trak, MIT license, `pip install traker` (the fast CUDA projection kernel is optional via `traker[fast]`). The built-in model output functions cover image classification, text classification, CLIP, and iterative image classification only. **There is none for language modeling**, so you must define a custom ModelOutput.
- Output function: treat the answer-token position as classification over the vocabulary and use the margin logit(answer) − logsumexp(other logits), the same form as the image classification version. The training-document output function (e.g., sum or mean of per-token margins) is a design choice: state it explicitly and record it.
- **Prefer implementing it yourselves first** (per-example gradients via `torch.func` functional_call + vmap + grad). Order:
  1. **Naive**: dot product of the query gradient and each document gradient (TracIn-style). This is the "N² lower bound" in Jihyun's roadmap.
  2. **TRAK**: random projection (k = 512–2048) → Φ → τ(z) = φ(z)ᵀ(ΦᵀΦ + λI)⁻¹ΦᵀQ, Q = diag(1 − p_i) → average over several checkpoints or independently trained models (e.g., 5).
  3. **Verification**: cross-check against the `traker` library on the same data.
- The TRAK paper includes a fact-tracing experiment with mT5 that is structurally similar. As far as we recall, BM25 was a strong baseline there too; verify before relying on it.

### 5.4 Evaluation
- Top-1 / top-k accuracy and MRR. Compare naive vs. TRAK vs. BM25.
- How the three methods differ on the conflicting-facts cases.
- **Counterfactual check**: for a few queries, remove the top-attributed document, retrain, and confirm the model loses the answer (drop in answer probability).
- **Cost measurement**: as model size grows, time one inference vs. one attribution. These numbers back the "speed" difficulty.
- (After scale-up) the TRAK paper's LDS (linear datamodeling score).

### 5.5 Figures (2–3 for the 3-minute pitch section)
1. **Heatmap** of queries × training documents. Ground truth should light up like a diagonal. This is literally the N² matrix.
2. **Counterfactual bars**: answer probability before and after removing the top document.
3. **Royalty split pie chart** aggregated by author (e.g., "Royalties for this answer: Author A 72%, B 11%, …").
4. For the technical-difficulties slide: a **cost curve** (speed), and a **contrast figure** showing that a unique fact is attributed sharply while a common token's attribution is diffuse (explainability).

### 5.6 Deliverables
- Reproducible code (data generation → training → attribution → evaluation → figures), fixed seeds.
- Slide-ready PNGs (Korean labels are fine) and a short results table.
- A brief README listing the simplifications and design choices.

---

## 6. Compute

- vast.ai credit: $500 (paid from the program's research funds). The budget is ample; no need to economize.
- Toy: barely needs a GPU (minutes even on CPU).
- Rule: **debug on an RTX 4090, run only the real experiments on an H100 80GB**. Consider H200 at the 7B scale. Reference prices (third-party summary as of March 2026; check the console before renting): 4090 ≈ $0.25–0.40/h, A100 80GB ≈ $0.8–1.5/h, H100 ≈ $2.0–3.5/h, H200 ≈ $3.5–5.0/h.
- **Destroy** idle instances rather than stopping them (stopped instances still incur disk charges). For long jobs, save checkpoints and use interruptible instances. Choose verified hosts with high reliability.
- RTX 5090 (Blackwell): building TRAK's fast-projection CUDA extension may fail. If you want to use it, run an install test first.

---

## 7. Scale-up after the mid-term pitch (direction)

- The model must have **publicly available training data**. Pythia (70M → 160M → 410M → 1.4B …, trained on The Pile in the same data order) is first choice; OLMo (Dolma) is the alternative.
- Gradient memory: in fp32, one example costs (#params × 4 bytes): about 0.5 GB at 124M, about 4 GB at 1B, about 28 GB at 7B. Up to roughly 1B, use full-gradient TRAK. Beyond that, move to low-rank gradient compression (LoGra / LogIX) or EK-FAC-style influence (Kronfluence). Verify the current state of each library. This transition is itself the "shaving down efficiency" story in Roy's section.

---

## 8. Open items and cautions

- Team 6's mid-term pitch date (Oct 23 vs. Oct 30). The toy scope depends on it: if it's tomorrow, only the heatmap and counterfactual; with a week, add the cost curve and the lineage experiment.
- In Minjun's section, the "what is needed to solve it" part is still undecided.
- The toy design in 5.1–5.5 is Recommended. If you have a reason to change it, propose the change with your reasoning.
- Any legal or policy figures used in the pitch must be checked against sources first.