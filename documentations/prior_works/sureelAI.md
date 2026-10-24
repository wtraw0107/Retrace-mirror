# Sureel AI: Technology, Patents, Data Scope

*Researched 2026-10-25. Facts only, each with its source. "Not disclosed" means I found no public source for it.*

---

## 0. Company

| Item | Fact |
|---|---|
| Founded | 2022, California (Palo Alto / Pacifica) |
| CEO / founder | Tamay Aykut: PhD in AI & robotics (TU Munich), ex-BMW, former visiting assistant professor at Stanford |
| Core co-inventor | Christopher Benjamin Kuhn (Munich) |
| Other leadership | Two co-presidents, one of them Benji Rogers |
| Programs | Creative Destruction Lab (Montreal, AI stream, 2023/24), NVIDIA Inception, AWS Activate, Google for Startups |
| Headcount | ~10 (LinkedIn via GetLatka, Nov 2025) |
| Status | Acquisition by Warner Music Group announced 2026-06-10. Price not disclosed. Runs as an "independent, federated company". |

---

## 1. What Sureel says it does (company and press claims, not verified)

| Function | Description | Source |
|---|---|---|
| **Opt-in training attribution** | Licenses approved media into AI training "that tracks your attribution from the first use". Owners can allow use, limit a file's influence in training, or exclude it, and licensing fees follow from tracked use. | sureel.ai; IEEE Spectrum |
| **Opt-out registry / enforcement** | Registers catalogs. AI developers check them through an invitation-only portal and API. A dashboard shows "which AI models have attempted to train on protected content". A screening layer runs "during generation". | MBW (BeatStars); sureel.ai |
| **Output attribution** | Gives each work an "AI DNA" by splitting it into component parts. Estimates the "fractional percentage of a song used in an AI-generated work", at track, stem and time-segment level. Builds "attribution graphs" covering composition and recording style. | MBW (WMG); OpenPlay release; Musicworks; DMN (STIM) |
| **Detection** | "Scans AI engines for uses of client copyrights". Also classifies whether audio is AI-generated. | OpenPlay release; Musicworks |
| **NIL suite** | Tracks voice clones, AI avatars and style replication. | TechCrunch / MBW (WMG) |
| **Positioning** | "Multi-Layer Attribution … layer by layer", described as not a "surface-level similarity check". Aykut says attribution must establish causality, not similarity. | sureel.ai; IEEE Spectrum |

---

## 2. Patents (all assigned to Sureel Inc)

Shared priority date: **2022-11-04** (provisional 63/422,885). Google Patents notes that this date is an assumption.

| # | Title | Filed | Status | Inventors |
|---|---|---|---|---|
| US20240419949A1 | Input-based attribution | 2023-08-08 | Published 2024-12-19 (grant status not checked) | Aykut, Kuhn |
| **US12314308B2** | Output-based attribution | 2023-09-06 | Granted 2025-05-27 | Kuhn, Aykut |
| **US12013891B2** | Model-based attribution | 2023-10-30 | Granted 2024-06-18 | Aykut, Kuhn |
| US12554767B2 | Adjusting attribution | 2024-01-29 | Granted 2026-02-17 | Kuhn, Aykut, Ponce De Leon Vera, Pauls, Burgmair |
| US12455918B2 | Model-based attribution (later application) | 2024-05-01 | Granted 2025-10-28 | Aykut, Kuhn |
| US12567396B2 | Output-based attribution, incl. musical content | 2025-05-26 | Granted 2026-03-03 | Kuhn, Aykut |
| US20260045242A1 | Output-based attribution for musical content | 2025-10-22 | Pending (published 2026-02-12) | Kuhn, Aykut |

*Claim 1 of each patent was not retrieved, because the page text was truncated. Everything below comes from the specifications (the description sections).*

### 2a. Input-based attribution (US20240419949A1)
1. Parse the user's prompt into a content type, a content description and creator identifiers.
2. Embed the prompt in a shared language-image space.
3. Measure its distance to creator embeddings (cosine, Jaccard, …).
4. Turn the distances into an attribution vector, which drives compensation.

### 2b. Output-based attribution (US12314308B2; music version US12567396B2 / US20260045242A1)
- **Comparison:** an embedding of the generated output is compared with embeddings of creators, content items or categories.
- **Creator embeddings:** built from captions encoded with a ViT. Captions come from CLIP (images) or CLAP (audio), and an LLM categorizes them.
- **Distance measures listed:** cosine, Jaccard, Hamming, Sørensen–Dice, Tanimoto, Tversky, contrastive-encoding distance, and others.
- **Output of the method:** an attribution vector giving the "percentage of influence" of each creator or item. Only the top X creators, or those above a Y% threshold (e.g. 5% or 10%), are kept. Item-level attributions are summed per creator. **No explicit formula for turning distances into percentages is given.**
- **Music-specific additions:**
  - The output is segmented by time (15/30/60/90 s), frequency band (20–100 Hz … 4–15 kHz), song section (intro, verse, chorus, …), stem (vocal, guitar, bass, keys, drums) and a MIDI-like symbolic form.
  - Melody is encoded as "half-step difference sequences" plus note-change times.
  - A **Composition and Style Model (CSM)** with dual heads or two autoencoders separates composition similarity from recording-style similarity. CLAP is named as its feature extractor.
  - Multi-scale similarity graphs and clustering are used.
  - Each output segment is embedded and compared with each training segment, and the distances are mapped to creators.

### 2c. Model-based attribution (US12013891B2, US12455918B2)
- **Setup:** a pre-trained base model (SD_BASE / MLM_base, pre-trained on open data such as LAION) is fine-tuned on creator content into SD_TUNED. This is done either **one model per creator** or **one model (MLM_All) for all creators**.
- **Static attribution vector (from training):** comes from the **parameter differences between the base and fine-tuned models**. "Changes to weights Wi … caused by content items 1 to k may be multiplied by their respective activations Vi." Gradients can also be tracked during training (batch or stochastic gradient descent).
- **Dynamic attribution vector (from generation):** comes from weight activations at inference time.
- **Threshold:** creators below a set contribution threshold get no attribution.
- **Transformativeness factor:** the relative increase in distance between outputs and the nearest training items after fine-tuning, compared with the base model, averaged over prompts and seeds.
- **Per-creator aggregation example:** creators A, B and C contribute 600, 300 and 100 items. Equal per-item influence gives 60/30/10; unequal influence can give 45/45/10.

### 2d. Adjusting attribution (US12554767B2)
- A user changes characteristics of the output (e.g. a creator's style) and how strongly each should influence it, across an initial and a later prompt. The attribution vector is built from those stated influence amounts.

### 2e. Models named in the patents
| Role | Named |
|---|---|
| Generative model (example) | Latent Diffusion Model / Stable Diffusion (SD_BASE → SD_TUNED). GAN, GPT, VAE and multimodal models are listed as alternatives. |
| Base pre-training data (example) | LAION |
| Creator identifiers during fine-tuning | DreamBooth |
| Captioning / embeddings | CLIP (image), CLAP (audio), ViT (creator embeddings) |
| Caption categorization | An LLM (GPT-3 named as an example) |
| Music generators | **None named** (no MusicGen, Jukebox, etc.). MERT is not mentioned. |

---

## 3. Which method or model Sureel uses in production

- **Not disclosed.** No paper, technical report, benchmark or accuracy figure from Sureel was found.
- The patents describe input-based, output-based, model-based and adjusting methods without saying which one is deployed.
- The website wording ("layer by layer", "not surface-level similarity") and the press wording ("AI DNA", "component parts") are the only public descriptions of the product.

---

## 4. Data scope

### Examples in the patents (illustrative, not real datasets)
- LAION as base pre-training data
- Fine-tuning data described as paintings, R&B songs, sci-fi novels, Renaissance art (≈1350–1620) and music from the 1960s–80s
- Creator examples: Vermeer, Picasso, Dalí, Rembrandt, Degas, Aretha Franklin, Isaac Asimov, James Cameron
- Modalities claimed: images/art, photos, music, text (eBooks), video

### Deployments
| Partner | Date | What is covered | Role | Size disclosed? |
|---|---|---|---|---|
| **OpenPlay** (music asset management) | 2024-12-04 | Catalogs of OpenPlay's rights-holder clients, connected through its API | Detection, fractional attribution, licensing | No |
| **BeatStars** (beat marketplace) | 2025-04-30 | "BeatStars' entire marketplace catalog" | Opt-out registry, plus a dashboard of training attempts | No (BeatStars had paid creators $325M since 2009, reported Jun 2024. This is a payout total, not catalog size.) |
| **STIM** (Swedish collecting society) | License launched 2025-09-08 | **Compositions only** (not recordings) that writers opt in one by one. An AI rights category started 2026-06-01, and "nothing is included automatically". First licensee: Songfox. | Attribution provider for outputs. Licensees file attribution analyses **monthly**. | No work count. Context: STIM has ~80,000 members and SEK 3.1B rights revenue (2025). |
| **WMG** (owner) | 2026-06 | "Sureel registry" | Not stated | "**Millions of music assets**" (WMG statement) |

STIM's payout rules as reported:
- **Training:** paid "proportionally and equally among all works in the licence".
- **Output:** paid according to attribution.
- **Fallback:** where attribution is unreliable, STIM distributes "by analogy" based on training-data usage.
- Fees and revenue-share terms are not published.

---

## 5. Not found
- Which method or model runs in production
- Any accuracy or validation figure, or a comparison against retraining
- Peer-reviewed papers by Sureel or its inventors on attribution
- Whether AI licensees must give Sureel access to model weights or training pipelines
- Number of works per deployment, or actual payout amounts

---

## Sources
- Patents: [US12314308B2](https://patents.google.com/patent/US12314308) · [US12013891B2](https://patents.google.com/patent/US12013891) · [US12455918B2](https://patents.google.com/patent/US12455918) · [US12554767B2](https://patents.google.com/patent/US12554767) · [US12567396B2](https://patents.google.com/patent/US12567396) · [US20260045242A1](https://patents.google.com/patent/US20260045242A1) · [US20240419949A1 (Justia)](https://patents.justia.com/patent/20240419949)
- [Sureel website](https://sureel.ai)
- [TechCrunch: WMG acquires Sureel AI (2026-06-10)](https://techcrunch.com/2026/06/10/warner-music-acquires-ai-attribution-startup-sureel-ai)
- [MBW: WMG acquires Sureel AI](https://musicbusinessworldwide.com/warner-music-group-acquires-sureel-ai-the-attribution-startup-that-traces-how-ai-models-use-artists-work/)
- [WMG job posting: Sureel AI](https://jobs.lever.co/wmg/82541d7f-6d5f-485b-9fc3-cd5640bc0d6c)
- [IEEE Spectrum: Generative AI Music Attribution Rethinks Royalties](https://spectrum.ieee.org/ai-music-attribution)
- [Musicworks interview with Tamay Aykut (2024-10-28)](https://musicworks.substack.com/p/bringing-responsible-and-explainable)
- [Creative Destruction Lab: Sureel](https://creativedestructionlab.com/companies/sureel/)
- [OpenPlay × Sureel release (2024-12-04)](https://jaybirdcom.com/openplay-signs-agreement-with-sureel-ai-bringing-its-ai-detection-and-monetization-capabilities-to-upcoming-api-marketplace/)
- [MBW: BeatStars × Sureel (2025-04-30)](https://www.musicbusinessworldwide.com/beatstars-teams-up-with-sureel-to-block-ai-companies-from-training-on-creator-content)
- [DMN: STIM taps Sureel (2025-09-26)](https://www.digitalmusicnews.com/2025/09/26/stim-taps-sureel-ai-attribution-tracking/)
- [STIM: world's first AI license for music](https://www.stim.se/en/news/stim-launches-the-worlds-first-ai-license-for-music)
- [Interspace: STIM AI rights category (2026-10-08)](https://interspacemusic.com/blog/stim-ai-rights-category-ai-licence-writer-approval/)
- [GetLatka: Sureel (headcount estimate)](https://getlatka.com/companies/sureel.ai)