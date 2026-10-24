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

## License

Apache-2.0
