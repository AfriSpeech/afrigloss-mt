# Afri-Gloss-MT 🌍

**Afri-Gloss-MT** is a toolkit and training framework for **glossary-constrained machine translation across African languages**.

It enables models to strictly preserve custom vocabulary and domain terminology (e.g. medical, legal, religious, or dialectal terms) by pre-inserting target translations directly into the input text (`source <target_term>`), forcing the model to translate context naturally around the desired terms without requiring retraining when dictionaries are updated.

---

## Key Features

- **440 Helsinki-NLP African Translation Models:** Curated directly from the official Hugging Face Collection: [AfriSpeech/helsinki-nlp-african-translation-models](https://huggingface.co/collections/AfriSpeech/helsinki-nlp-african-translation-models-6adc5ed6acc7c9c71098e79a).
- **693 African Languages Supported:** Native ingestion from [`AfriSpeech/africa-corpus`](https://huggingface.co/datasets/AfriSpeech/africa-corpus) via sentence/verse-aligned parallel text.
- **Statistical & Neural Word Alignment:** Automatically aligns parallel sentences and calculates bidirectional association confidence scores:
  $$\text{Confidence}(s, t) = \frac{1}{2} \text{Dice}(s, t) + \frac{1}{2} \left( \frac{2 \cdot P(t|s) \cdot P(s|t)}{P(t|s) + P(s|t)} \right)$$
- **High-Confidence Terminology Filtering:** Extracts verified bilingual dictionaries by filtering on alignment confidence ($\ge 0.70$), eliminating alignment noise.
- **Pre-inserted Gloss-MT Dataset Builder:** Injects target vocabulary in `<target_term>` brackets on both sides for fine-tuning.
- **Lightweight MarianMT (OPUS-MT) Trainer:** Fine-tunes 74M-parameter OPUS-MT models with Hopper H200 `bf16` + `TF32` single-GPU support.
- **Interactive Translation & Offline Dictionary Runtime:** Translate with pre-loaded offline TSVs or ad-hoc prompt-level terminology overrides.

---

## Architecture & How It Works

```
Parallel Corpus (AfriSpeech/africa-corpus)
                 │
                 ▼
Word Alignment Engine (Confidence Scoring >= 0.70)
                 │
                 ▼
High-Confidence Bilingual Dictionary (TSV)
                 │
                 ▼
Gloss-MT Dataset Construction
  Input:  "In the beginning God <Onyankopɔn> created the heavens <ɔsoro> and the earth <asase>."
  Target: "Mfiase no <Onyankopɔn> bɔɔ <ɔsoro> ne <asase>."
  Ref:    "Mfiase no Onyankopɔn bɔɔ ɔsoro ne asase."
                 │
                 ▼
MarianMT Fine-Tuning (Helsinki-NLP OPUS-MT Base)
                 │
                 ▼
Inference Runtime with Terminology Fidelity
  Input:  "The angel of God visited the village."
  Prompt: "The angel <ɔbɔfo> of God <Onyankopɔn> visited the village."
  Output: "Onyankopɔn bɔfo no kɔsraa kuro no."  (Guaranteed: Onyankopɔn, ɔbɔfo)
```

---

## Dataset Scale: Smoke Tests vs. Production

| Split / Scope | Verses | What It Learns |
|---|:---:|---|
| **Smoke Test / Pilot** | `3,000` | Verifies pipeline execution, gradients, and tensor dimensions. Too small for full vocabulary coverage. |
| **Full Single Bible Version** | `~31,000` | Full coverage of Old & New Testament (thousands of distinct noun phrases & verbs). Reaches **90%+ glossary adherence**. |
| **Multi-Version Corpus** | `60,000 – 150,000+` | Full lexical density across dialect variations (e.g. Asante Twi, Akuapem Twi, Fante). |

> **Recommendation:** Use `3,000` samples only for fast pipeline verification. For production Gloss-MT training, run without `--max-samples` to utilize the full **31,000+ verses** available in `africa-corpus`.

---

## 🚀 100k Production Demo Models (Large-Scale Pre-Translated Datasets)

Fine-tuned on the H200 GPU using **100k multi-version datasets** (85k–87k training samples) and evaluated across **4,300+ unseen test verses**:

| Language Pair | Architecture | Training Samples | Test BLEU | Test chrF2 | Training Time | Checkpoint Path |
|:---:|---|:---:|:---:|:---:|:---:|---|
| **English $\rightarrow$ Swahili (`sw`)** | `Helsinki-NLP/opus-mt-en-sw` | **78,657** | **23.75** | **53.42** | 25.4 mins | `checkpoints/afrigloss_opus_en_sw_100k/final` |
| **English $\rightarrow$ Twi (`tw`)** | `Helsinki-NLP/opus-mt-en-tw` | **77,006** | **21.70** | **47.28** | 26.6 mins | `checkpoints/afrigloss_opus_en_tw_100k/final` |

---

## 📊 10k Pilot Model Benchmarks (Afri-Gloss-MT Checkpoints)

Fine-tuned on the H200 GPU using the 10,000-verse Gloss-MT datasets and evaluated on 500 held-out unseen test verses with `sacrebleu`:

| Language Pair | Base Model Architecture | Test BLEU | Test chrF2 | Training Time (H200) | Checkpoint Directory |
|:---:|---|:---:|:---:|:---:|---|
| **English $\rightarrow$ Swahili (`sw`)** | `Helsinki-NLP/opus-mt-en-sw` | **24.49** | **53.30** | 3.66 mins | `checkpoints/afrigloss_opus_en_sw/final` |
| **English $\rightarrow$ Twi (`tw`)** | `Helsinki-NLP/opus-mt-en-tw` | **23.72** | **49.00** | 3.87 mins | `checkpoints/afrigloss_opus_en_tw/final` |
| **English $\rightarrow$ Hausa (`ha`)** | `Helsinki-NLP/opus-mt-en-ha` | **21.84** | **47.62** | 3.45 mins | `checkpoints/afrigloss_opus_en_ha/final` |
| **English $\rightarrow$ Igbo (`ig`)** | `Helsinki-NLP/opus-mt-en-ig` | **19.36** | **44.40** | 3.41 mins | `checkpoints/afrigloss_opus_en_ig/final` |
| **English $\rightarrow$ Ewe (`ee`)** | `Helsinki-NLP/opus-mt-en-ee` | **18.28** | **42.85** | 3.58 mins | `checkpoints/afrigloss_opus_en_ee/final` |
| **English $\rightarrow$ Shona (`sn`)** | `Helsinki-NLP/opus-mt-en-sn` | **9.37** | **46.22** | 3.77 mins | `checkpoints/afrigloss_opus_en_sn/final` |
| **English $\rightarrow$ Yoruba (`yo`)** | `Helsinki-NLP/opus-mt-en-alv` (`>>yor<<`) | **21.77** | **43.73** | 3.91 mins | `checkpoints/afrigloss_opus_en_yo/final` |

---

## 🏆 Head-to-Head: Official Helsinki-NLP OPUS-MT Test Benchmark

Evaluated directly on the **exact official Helsinki-NLP test sets** (`https://object.pouta.csc.fi/OPUS-MT-models/`):

| Language | Original OPUS-MT Base (BLEU / chrF2) | Afri-Gloss-MT (No Dict) (BLEU / chrF2) | Afri-Gloss-MT (**WITH Big Dict**) (BLEU / chrF2) | Gain over Base ($\Delta$BLEU / $\Delta$chrF2) | Glossary Adherence |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **Ewe (`ee`)** | 3.38 / 15.39 | 31.02 / 55.46 | **27.28 / 51.38** | **+27.64 / +40.07** | **94.1%** |
| **Igbo (`ig`)** | 3.57 / 15.48 | 23.28 / 49.01 | **21.69 / 47.78** | **+19.71 / +33.53** | **90.0%** |
| **Hausa (`ha`)** | 5.43 / 19.53 | 23.67 / 48.07 | **21.78 / 46.77** | **+18.24 / +28.54** | **90.3%** |
| **Shona (`sn`)** | 5.68 / 19.81 | 23.42 / 53.52 | **21.53 / 53.21** | **+17.74 / +33.71** | **87.6%** |
| **Yoruba (`yo`)** | 15.86 / 34.26* | 28.92 / 48.29 | **26.54 / 45.49** | **+13.06 / +14.03** | **88.0%** |
| **Twi (`tw`)** | 21.84 / 38.52 | 26.87 / 50.60 | **23.21 / 47.50** | **+5.03 / +12.08** | **89.3%** |
| **Swahili (`sw`)**| 24.08 / 52.60 | 13.74 / 43.22 | **13.20 / 42.47** | Domain Shift | **96.4%** |
| **French $\rightarrow$ Twi (`tw`)** | 5.61 / 18.10 | 33.58 / 56.77 | **28.14 / 52.59** | **+27.97 / +38.67** | **91.2%** |
| **French $\rightarrow$ Swahili (`sw`)** | 8.99 / 29.79 | 30.78 / 56.95 | **29.98 / 56.16** | **+21.79 / +27.16** | **95.8%** |
| **French $\rightarrow$ Ewe (`ee`)** | 9.61 / 29.90 | 29.91 / 52.87 | **29.15 / 52.10** | **+20.30 / +22.97** | **92.5%** |
| **French $\rightarrow$ Igbo (`ig`)** | 7.20 / 24.22 | 27.12 / 48.25 | **26.37 / 47.80** | **+19.92 / +24.03** | **89.7%** |
| **French $\rightarrow$ Hausa (`ha`)** | 7.58 / 27.51 | 27.24 / 50.33 | **26.53 / 49.51** | **+19.66 / +22.82** | **88.9%** |
| **French $\rightarrow$ Yoruba (`yo`)** | 6.68 / 21.83 | 26.29 / 44.93 | **24.78 / 43.60** | **+19.61 / +23.10** | **87.4%** |
| **French $\rightarrow$ Shona (`sn`)** | 8.83 / 34.78 | 26.64 / 56.99 | **25.93 / 56.47** | **+17.81 / +22.21** | **86.8%** |

### Benchmark Highlights:
- **Massive Gains Across Low-Resource Pairs:** On Ewe, Hausa, Igbo, Shona, and French pairs, the fine-tuned Gloss-MT models deliver dramatic improvements over the original OPUS-MT baselines (up to **+27.6 BLEU / +40.1 chrF2** on Ewe, and **+27.9 BLEU / +38.7 chrF2** on French $\rightarrow$ Twi).
- **High Controllability:** Across languages, **86% to 96%** of pre-inserted glossary constraints appear verbatim in the translation output.
- **Reproduce the Benchmark:**
  ```bash
  python scripts/benchmark_official_opus.py --lang tw --model checkpoints/afrigloss_opus_en_tw_full/final --dictionary data/dictionaries/eng_twi.tsv --samples 500
  ```

---

## 🏅 Model Selection Guide: Rankings Across Architectures per Language

For the complete technical breakdown, latency benchmarks, and language analysis, see **[MODELS_RANKING.md](MODELS_RANKING.md)**.

### 🇬🇧 English $\rightarrow$ African Targets (WITH Dictionary):

| Target African Language | Family | Afri-Gloss OPUS-MT (77M) | Afri-Gloss NLLB-600M | Afri-Gloss NLLB-1.3B | 🥇 Best Model (Quality Winner) |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Twi / Akan (`tw`)** | Kwa | 23.21 / 47.50 | 24.49 / 48.40 | **27.37 / 50.95** | 🥇 **NLLB-1.3B** (+4.16 BLEU over OPUS) |
| **Hausa (`ha`)** | Chadic | 21.78 / 46.77 | 23.82 / 49.27 | **26.78 / 51.08** | 🥇 **NLLB-1.3B** (+5.00 BLEU over OPUS) |
| **Ewe (`ee`)** | Gbe | 27.28 / 51.38 | 23.17 / 50.09 | **27.41 / 53.69** | 🥇 **NLLB-1.3B** (+2.31 chrF2 over OPUS) |
| **Shona (`sn`)** | Bantu | 21.53 / 53.21 | 22.78 / 55.51 | **24.70 / 56.46** | 🥇 **NLLB-1.3B** (+3.17 BLEU over OPUS) |
| **Igbo (`ig`)** | Igboid | 21.69 / 47.78 | 21.19 / 48.00 | **22.45 / 50.05** | 🥇 **NLLB-1.3B** (+2.27 chrF2 over OPUS) |
| **Swahili (`sw`)** | Bantu | 13.20 / 42.47 | 18.27 / 48.76 | **19.17 / 49.01** | 🥇 **NLLB-1.3B** (+5.97 BLEU over OPUS) |
| **Yoruba (`yo`)** | Defoid | **26.54 / 45.49** | 18.51 / 37.56 | 21.39 / 40.08 | 🥇 **OPUS-MT** (+5.15 BLEU over NLLB-1.3B) |

---

### 🇫🇷 French $\rightarrow$ African Targets: Quality Leaps & Performance:

| Target African Language | Family | Original OPUS Base (BLEU / chrF2) | Afri-Gloss-MT (No Dict) | Afri-Gloss-MT (**WITH Dict**) | Quality Leap ($\Delta$BLEU / $\Delta$chrF2) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **French $\rightarrow$ Twi (`tw`)** | Kwa | 5.61 / 18.10 | 33.58 / 56.77 | **28.14 / 52.59** | **+27.97 / +38.67** |
| **French $\rightarrow$ Swahili (`sw`)** | Bantu | 8.99 / 29.79 | 30.78 / 56.95 | **29.98 / 56.16** | **+21.79 / +27.16** |
| **French $\rightarrow$ Ewe (`ee`)** | Gbe | 9.61 / 29.90 | 29.91 / 52.87 | **29.15 / 52.10** | **+20.30 / +22.97** |
| **French $\rightarrow$ Igbo (`ig`)** | Igboid | 7.20 / 24.22 | 27.12 / 48.25 | **26.37 / 47.80** | **+19.92 / +24.03** |
| **French $\rightarrow$ Hausa (`ha`)** | Chadic | 7.58 / 27.51 | 27.24 / 50.33 | **26.53 / 49.51** | **+19.66 / +22.82** |
| **French $\rightarrow$ Yoruba (`yo`)** | Defoid | 6.68 / 21.83 | 26.29 / 44.93 | **24.78 / 43.60** | **+19.61 / +23.10** |
| **French $\rightarrow$ Shona (`sn`)** | Bantu | 8.83 / 34.78 | 26.64 / 56.99 | **25.93 / 56.47** | **+17.81 / +22.21** |

### Production Selection Rules:
1. 🏆 **Top Overall Quality:** Use **[`AfriSpeech/afrigloss-nllb-1.3b`](https://huggingface.co/AfriSpeech/afrigloss-nllb-1.3b)** — wins 6 out of 7 African languages, delivering the highest grammatical fluency and handling complex Bantu concordial noun prefixes with high precision.
2. 🎯 **Yoruba Tonal Precision:** Use **[`AfriSpeech/afrigloss-opus-mt-en-yo`](https://huggingface.co/AfriSpeech/afrigloss-opus-mt-en-yo)** — dedicated vocabulary models prevent diacritic tone fragmentation (`ọ́`, `è`, `à`, `ṣ`), outperforming NLLB by **+5.15 BLEU**.
3. ⚡ **Ultra-Low Latency & Edge CPU Serving:** Use **[`AfriSpeech/afrigloss-opus-mt-*`](https://huggingface.co/collections/AfriSpeech/afri-gloss-mt-models-6adce3a3c13c6a86f07b6e6d)** — lightweight (~280 MB per model) and translates in ~30–50ms on a single CPU core.
4. 🌐 **Balanced Multilingual Serving:** Use **[`AfriSpeech/afrigloss-nllb-600m`](https://huggingface.co/AfriSpeech/afrigloss-nllb-600m)** — single 2.4 GB unified model translating from both English and French to all 7 African languages simultaneously.

---

## ⚡ Training Guidelines (Learnings from GLOSS-MT)

- **Early Stopping & Checkpoint Selection:**
  All trainers evaluate periodically (every 150 steps on 10k datasets) and use `EarlyStoppingCallback(early_stopping_patience=4)`.
  Training automatically stops when validation loss plateaus, and `load_best_model_at_end=True` guarantees the optimal checkpoint is retained.
- **Model Budgets:**
  - **OPUS-MT (MarianMT):** Budget up to 15 epochs. Usually converges at 5–8 epochs with early stopping.
  - **NLLB-200 (600M / 1.3B):** Budget 15–25 epochs.
  - **TranslateGemma (4B LLM):** Budget 3–5 epochs with LoRA ($r=16$ or $32$).

---

## Installation

```bash
git clone https://github.com/AfriSpeech/afrigloss-mt.git
cd afrigloss-mt
pip install -e .
```

---

## Quickstart CLI

### 1. Inspect Models in the Collection
```bash
# Check model for a specific language pair
afrigloss list-models --src en --tgt tw
afrigloss list-models --src en --tgt yo
afrigloss list-models --src en --tgt sw

# List all supported African languages in the registry
afrigloss list-models
```

### 2. Extract High-Confidence Bilingual Dictionary
Extract aligned words with confidence score $\ge 0.70$:
```bash
afrigloss build-dict --src en --tgt tw --out data/dictionaries/eng_twi.tsv --min-confidence 0.70
afrigloss build-dict --src en --tgt yo --out data/dictionaries/eng_yor.tsv --min-confidence 0.70
afrigloss build-dict --src en --tgt sw --out data/dictionaries/eng_swh.tsv --min-confidence 0.70
```

### 3. Generate Gloss-MT Pre-Inserted Training Dataset
```bash
afrigloss build-data \
    --src en \
    --tgt tw \
    --dictionary data/dictionaries/eng_twi.tsv \
    --out-dir data/gloss_dataset \
    --val-ratio 0.05 \
    --test-ratio 0.05
```

### 4. Fine-Tune OPUS-MT on H200 GPU or Local CUDA
```bash
afrigloss train \
    --src en \
    --tgt tw \
    --train-file data/gloss_dataset/train.jsonl \
    --val-file data/gloss_dataset/validation.jsonl \
    --out checkpoints/opus-mt-en-tw-gloss \
    --epochs 5 \
    --batch-size 16 \
    --lr 3e-5
```

### 5. Translate with Terminology Constraints
```bash
# One-shot translation with dictionary lookup
afrigloss translate \
    --src en \
    --tgt tw \
    --dictionary data/dictionaries/eng_twi.tsv \
    --text "The covenant with God will stand forever."

# Translation with ad-hoc glossary overrides
afrigloss translate \
    --src en \
    --tgt tw \
    --text "The leader made a covenant." \
    --gloss "covenant:apam,leader:kannifo"

# Interactive REPL mode
afrigloss translate --src en --tgt tw --dictionary data/dictionaries/eng_twi.tsv --interactive
```

---

## Python API Usage

```python
from afrigloss import AfriGlossTranslator, GlossaryDictionary

# 1. Load dictionary
dictionary = GlossaryDictionary.load_tsv("data/dictionaries/eng_twi.tsv")

# 2. Initialize translator
translator = AfriGlossTranslator(
    model_name_or_path="Helsinki-NLP/opus-mt-en-tw",
    dictionary=dictionary,
)

# 3. Translate with automatic dictionary lookup
result = translator.translate("God created the heavens and the earth.")
print("Pre-injected:", result["pretranslated_input"])
# -> "God <Onyankopɔn> created the heavens <ɔsoro> and the earth <asase>."
print("Translation:", result["translation"])
# -> "Onyankopɔn bɔɔ ɔsoro ne asase."
print("Preserved:", result["constraints_honored"])
# -> True
```

---

## 🛠️ Bringing Your Own Custom Parallel Data

Users can build Gloss-MT training data and dictionaries from **their own parallel text** (e.g. domain data, technical, medical, legal, or news) with a single command:

```bash
# 1. From a tabular file (.tsv, .csv, .jsonl, .parquet):
afrigloss build-custom \
    --parallel-file my_parallel_data.tsv \
    --src en \
    --tgt tw \
    --out-dir data/my_custom_gloss_data \
    --target-samples 100000

# 2. Or from paired line-aligned text files:
python scripts/build_from_custom_data.py \
    --src-file english.txt \
    --tgt-file twi.txt \
    --src en --tgt tw \
    --out-dir data/my_custom_gloss_data
```

This automated pipeline:
1. Loads the sentence pairs.
2. Trains the statistical word aligner on the full text and extracts high-confidence dictionary mappings (`dictionary_en_tw.tsv`).
3. Extracts spaCy noun chunks from source sentences and replaces them in-place with `<target_term>`.
4. Produces `train.jsonl`, `validation.jsonl`, and `test.jsonl` splits ready for immediate model training.

---

## 📈 Scaling to 100k+ Samples & Low-Resource Strategy

Fine-tuning on larger datasets (50,000 to 100,000+ sentences) serves a dual purpose:
1. **Constraint Learning:** Teaches the model the Gloss-MT in-place terminology preservation rule across diverse grammatical structures.
2. **Translation Improvement:** Continuously exposes the model to parallel text, improving general translation vocabulary and BLEU/chrF2 quality.

### Where to Get 100k+ Sentences:
- **`AfriSpeech/africa-corpus`:** Multi-version Bible collections provide 30,000 to 350,000+ verses per language (e.g. Swahili has 13 versions / 356k sentences; Twi has 5 versions / 154k sentences).
- **`AfriSpeech/african-corpus-jw`:** Large-scale JW300 dataset provides 100,000 to 250,000+ parallel sentences for over 100 African languages.

### Advice for Low-Resource Languages (< 5,000 Sentences):
If a language has only a few hundred or thousand sentences available:
- **Train with what is available!** Even a small dataset (1,000–3,000 pairs) is sufficient to teach the model the abstract in-place bracket copying mechanism.
- **Evaluate carefully:** Always check validation loss and test set chrF2/BLEU to ensure general translation quality remains fluent.

---

## Remote Training on Ghana NLP H200 GPU

Use the bundled `scripts/run_h200.sh` helper to run data processing and training on the H200:

```bash
# 1. Sync code and dictionaries to H200
./scripts/run_h200.sh sync

# 2. Run full 31,000+ verse word alignment on H200
./scripts/run_h200.sh align twi
./scripts/run_h200.sh align-all

# 3. Launch fine-tuning on Hopper H200 GPU (bf16 + TF32)
./scripts/run_h200.sh train en tw data/gloss_dataset
```

---

## 📚 Master African Base Dictionaries

Afri-Gloss-MT bundles and generates comprehensive base dictionaries across all supported African languages in `data/dictionaries/`:
- **Source 1: Human-Curated Linguistic Wordlists:** Ingested from [`AfriSpeech/webonary-african-wordlists`](https://huggingface.co/datasets/AfriSpeech/webonary-african-wordlists) covering **168 African languages** with thousands of verified vocabulary entries per language.
- **Source 2: Full-Corpus Statistical Alignment:** Word alignment trained over hundreds of thousands of parallel sentences from [`AfriSpeech/africa-corpus`](https://huggingface.co/datasets/AfriSpeech/africa-corpus) and [`AfriSpeech/african-corpus-jw`](https://huggingface.co/datasets/AfriSpeech/african-corpus-jw).

### Building or Expanding Dictionaries:
```bash
# Extract Webonary linguistic dictionaries and run full alignment across all languages:
python scripts/build_huge_base_dictionaries.py --webonary --align-all
```

---

## 🤖 Optional: Expanding Dictionaries with Gemini 3.8 Flash

For specialized domains or to translate custom English noun lists (such as [`AfriSpeech/africa-nouns`](https://huggingface.co/datasets/AfriSpeech/africa-nouns)), we provide a high-throughput Gemini builder script:

```bash
export GEMINI_API_KEY="your-api-key"
python scripts/build_mega_dictionary.py --tgt twi --limit 10000 --batch-size 25 --workers 10
```

> **Why Gemini?** Gemini 3.8 Flash provides strong multilingual performance on African languages. You can inspect benchmark scores and evaluations on the **[AfriSpeech Gemini Afro-MT Bench](https://huggingface.co/spaces/AfriSpeech/gemini-afro-mt-bench)**.

---

## 📝 Dictionary File Template & Merging Custom Glossaries

Dictionaries are stored in clean, tab-delimited TSV format (`source_phrase \t target_phrase [\t confidence \t count]`):

```tsv
source_phrase	target_phrase	confidence	count
covenant	apam	0.9800	150
hospital	ayaresabea	0.9500	45
doctor	dɔkota	0.9200	32
heavens	ɔsoro	0.9500	210
earth	asase	0.9400	180
```

Merge any custom or community glossary into the master language dictionary:
```bash
afrigloss merge-dicts \
    --base data/dictionaries/eng_twi.tsv \
    --new my_custom_medical_glossary.tsv \
    --out data/dictionaries/eng_twi.tsv
```

---

## 🤝 Community Contributions & Model Sharing

Afri-Gloss-MT is built to be **collectively efficient**:
- **Shared Alignment Caches:** If an alignment cache exists for a language in `data/align_cache/`, any user generating datasets automatically **skips re-alignment** and builds their dataset in seconds.
- **Contribute New Language Caches:** If you run word alignment on a new African language, submit a PR with your `align_cache_en_<lang>.jsonl` so the entire community benefits from your compute!
- **Share Your Fine-Tuned Models:** Fine-tuned a model for a new language or domain? Submit it via our **[Model Submission Form](https://github.com/AfriSpeech/afrigloss-mt/issues/new?template=submit_model.yml)** to have it added to the official registry and Hugging Face collection.

See **[CONTRIBUTING.md](CONTRIBUTING.md)** for detailed instructions.

---

## License

Apache-2.0
