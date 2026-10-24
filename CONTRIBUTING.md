# Contributing to Afri-Gloss-MT 🌍

Thank you for contributing to open-source African NLP and glossary-constrained machine translation!

Our mission is to make training and deploying high-quality, terminology-constrained translation models **fast, accessible, and collectively efficient** for all 693+ African languages.

---

## 🚀 Two Main Ways to Contribute

### 1. Contributing Pre-Computed Alignment Caches (Save Compute for Everyone!)

Word alignment and confidence scoring on 50,000 to 100,000+ sentences requires compute. Once a language pair is aligned, **nobody else should have to re-compute it from scratch**.

When you run:
```bash
python scripts/build_all_en_datasets.py --langs lug --count 50000
```
Afri-Gloss-MT automatically saves every verified word alignment into `data/align_cache/align_cache_en_lug.jsonl`.

#### How to Share Your Alignment Cache:
1. Place your `align_cache_en_<lang>.jsonl` (or gzipped `.jsonl.gz`) in `data/align_cache/`.
2. Open a Pull Request titled `[Alignment Cache]: en <-> <lang> (<N> pairs)`.
3. Alternatively, if your file is very large (>100MB), upload it as a dataset to the [Hugging Face Hub](https://huggingface.co) and submit an issue using our [Alignment Contribution Template](https://github.com/AfriSpeech/afrigloss-mt/issues/new?template=contribute_alignment.yml).

Once merged, any user running `afrigloss build-data` for that language will **instantly load the cached alignments and generate their training data in seconds**.

---

### 2. Contributing Bilingual Dictionaries & Glossaries (Build Strong Language Dictionaries!)

Bilingual dictionaries directly power the in-place terminology replacement pipeline in Afri-Gloss-MT. By sharing your domain glossaries (medical, healthcare, legal, agricultural, educational, or general vocabulary), you help build a comprehensive, open-source master dictionary for each African language!

#### How to Contribute a Dictionary:
1. Ensure your dictionary is in TSV format:
   ```tsv
   source_phrase	target_phrase	confidence
   hospital	ayaresabea	0.95
   doctor	dɔkota	0.98
   covenant	apam	0.92
   ```
2. Submit a Pull Request adding your file to `data/dictionaries/` or use the CLI merge tool to merge it into the existing language dictionary:
   ```bash
   afrigloss merge-dicts \
       --base data/dictionaries/eng_twi.tsv \
       --new my_medical_terms.tsv \
       --out data/dictionaries/eng_twi.tsv
   ```
3. Or submit an issue using our **[Dictionary Contribution Form](https://github.com/AfriSpeech/afrigloss-mt/issues/new?template=contribute_dictionary.yml)**.

---

### 3. Submitting Your Fine-Tuned Gloss-MT Models

If you fine-tune an OPUS-MT, NLLB-200, or TranslateGemma model on a new African language pair or a specialized domain (e.g. medical, agricultural, legal):

1. Upload your model weights, tokenizer, and evaluation metrics to the Hugging Face Hub (e.g. `your-org/opus-mt-en-lug-gloss`).
2. Submit an issue using our **[Model Submission Form](https://github.com/AfriSpeech/afrigloss-mt/issues/new?template=submit_model.yml)**.
3. Once verified, we will:
   - Add your model to the dedicated **[Afri-Gloss-MT Models Collection](https://huggingface.co/collections/AfriSpeech/afri-gloss-mt-models-6adce3a3c13c6a86f07b6e6d)** on Hugging Face (kept distinct from base models).
   - Include your model in `afrigloss/registry.py` so users can resolve and translate with it directly via `afrigloss translate`.
   - Feature your model in the repository benchmarks table.

> **Note on Collections:** The [Helsinki-NLP African Translation Models Collection](https://huggingface.co/collections/AfriSpeech/helsinki-nlp-african-translation-models-6adc5ed6acc7c9c71098e79a) is reserved strictly for *pretrained base models*. All fine-tuned models trained with pre-inserted terminology constraints live in the dedicated [Afri-Gloss-MT Models Collection](https://huggingface.co/collections/AfriSpeech/afri-gloss-mt-models-6adce3a3c13c6a86f07b6e6d).

---

## 🛠️ Development & Testing Guidelines

### Setting Up Development Environment
```bash
git clone https://github.com/AfriSpeech/afrigloss-mt.git
cd afrigloss-mt
pip install -e ".[dev]"
```

### Running the Test Suite
Ensure all unit tests pass before submitting a PR:
```bash
pytest tests/
```

### Code Style
- Follow PEP 8 guidelines.
- Add type annotations to public API functions.
- Avoid external dependencies unless strictly necessary.

---

## 💬 Community & Questions

For questions, discussions, or proposals:
- Open a GitHub Discussion or Issue.
- Reach out to the [AfriSpeech](https://github.com/AfriSpeech) and [Ghana NLP](https://ghananlp.org) engineering teams.
