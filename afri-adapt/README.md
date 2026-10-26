# Afri-Adapt: Grounded Multilingual Proper Noun & Entity Lexicons

**Afri-Adapt** provides comprehensive, human-grounded bilingual proper noun dictionaries mapping English and French named entities, personal names, geographical locations, organizations, and cultural/historical terms into **7 African languages**:

- **Twi / Akan** (`tw`)
- **Swahili** (`sw`)
- **Yoruba** (`yo`)
- **Hausa** (`ha`)
- **Igbo** (`ig`)
- **Ewe** (`ee`)
- **Shona** (`sn`)

All entities are extracted directly from authentic multi-source parallel text using sentential context alignment and majority-vote consensus (**100% human bitext, zero LLM/synthetic translations**).

---

## Dataset Summary

| Pair Key | Language Direction | File | Verified Proper Nouns |
| :--- | :--- | :--- | :---: |
| `en_ee` | English $\rightarrow$ Ewe | `en_ee_proper_nouns.tsv` | **8,720** |
| `en_ha` | English $\rightarrow$ Hausa | `en_ha_proper_nouns.tsv` | **4,679** |
| `en_ig` | English $\rightarrow$ Igbo | `en_ig_proper_nouns.tsv` | **9,171** |
| `en_sn` | English $\rightarrow$ Shona | `en_sn_proper_nouns.tsv` | **7,976** |
| `en_sw` | English $\rightarrow$ Swahili | `en_sw_proper_nouns.tsv` | **12,281** |
| `en_tw` | English $\rightarrow$ Twi / Akan | `en_tw_proper_nouns.tsv` | **18,909** |
| `en_yo` | English $\rightarrow$ Yoruba | `en_yo_proper_nouns.tsv` | **9,316** |
| `fr_ee` | French $\rightarrow$ Ewe | `fr_ee_proper_nouns.tsv` | **7,614** |
| `fr_ha` | French $\rightarrow$ Hausa | `fr_ha_proper_nouns.tsv` | **4,057** |
| `fr_ig` | French $\rightarrow$ Igbo | `fr_ig_proper_nouns.tsv` | **7,766** |
| `fr_sn` | French $\rightarrow$ Shona | `fr_sn_proper_nouns.tsv` | **7,066** |
| `fr_sw` | French $\rightarrow$ Swahili | `fr_sw_proper_nouns.tsv` | **10,772** |
| `fr_tw` | French $\rightarrow$ Twi / Akan | `fr_tw_proper_nouns.tsv` | **12,591** |
| `fr_yo` | French $\rightarrow$ Yoruba | `fr_yo_proper_nouns.tsv` | **8,789** |
| **Total** | | | **129,707** |

---

## File Schema

Each TSV file contains:
- `source_term`: The proper noun or named entity in the source language (English or French).
- `target_term`: The verified translation / localized transliteration in the African target language.
- `confidence`: Consensus score from empirical training alignments (0.00 – 1.00).
- `frequency`: Total number of occurrences observed across the parallel training corpora.

### Consolidated Master Lexicon
The combined file `all_proper_nouns.tsv` pools all language pairs with additional `src_lang` and `tgt_lang` columns for rapid multi-target querying.

---

## Quick Usage

### Python (Pandas)
```python
import pandas as pd

# Load English -> Twi proper nouns
df_tw = pd.read_csv("afri-adapt/en_tw_proper_nouns.tsv", sep="\t")
print(df_tw.head(10))

# Query specific entity across all African languages
df_all = pd.read_csv("afri-adapt/all_proper_nouns.tsv", sep="\t")
jerusalem_translations = df_all[df_all["source_term"].str.lower() == "jerusalem"]
print(jerusalem_translations[["tgt_lang", "target_term", "frequency"]])
```
