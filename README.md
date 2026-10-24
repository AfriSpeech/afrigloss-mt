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

## 📊 Trained Model Benchmarks (Afri-Gloss-MT Checkpoints)

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

## Precomputed Dictionaries Bundled

Pre-extracted high-confidence dictionaries ($\text{confidence} \ge 0.70$) are included in `data/dictionaries/`:
- `eng_twi.tsv` (Twi / Akan, Ghana)
- `eng_yor.tsv` (Yoruba, Nigeria)
- `eng_swh.tsv` (Swahili, East Africa)
- `eng_ibo.tsv` (Igbo, Nigeria)
- `eng_ewe.tsv` (Ewe, Ghana / Togo)
- `eng_sna.tsv` (Shona, Zimbabwe)
- `eng_lug.tsv` (Luganda, Uganda)
- `eng_kin.tsv` (Kinyarwanda, Rwanda)
- `eng_nya.tsv` (Chichewa / Nyanja, Malawi)
- `eng_xho.tsv` (Xhosa, South Africa)
- `eng_zul.tsv` (Zulu, South Africa)
- `eng_lin.tsv` (Lingala, DRC / Congo)

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
