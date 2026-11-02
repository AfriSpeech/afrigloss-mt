# 🏅 Afri-Gloss-MT Model Rankings & Selection Guide

This guide provides an empirical head-to-head performance ranking of all fine-tuned **Afri-Gloss-MT** model families across each target African language.

All models were evaluated on identical held-out official benchmark test sentences with hard terminology constraints enforced via the offline grounded dictionaries.

---

## 🇬🇧 English $\rightarrow$ African Targets: Head-to-Head Comparison WITH Dictionary

Evaluation scores are reported as **BLEU / chrF2** computed via `sacrebleu`:

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

## 🇫🇷 French $\rightarrow$ African Targets: Head-to-Head Comparison WITH Dictionary

Evaluation scores are reported as **BLEU / chrF2** computed via `sacrebleu`:

| Target African Language | Family | Afri-Gloss OPUS-MT (77M) | Afri-Gloss NLLB-600M | Afri-Gloss NLLB-1.3B | 🥇 Best Model (Quality Winner) |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Twi / Akan (`tw`)** | Kwa | 28.14 / 52.59 | 28.78 / 53.26 | **31.49 / 55.47** | 🥇 **NLLB-1.3B** (+3.35 BLEU over OPUS) |
| **Swahili (`sw`)** | Bantu | 29.98 / 56.16 | 29.51 / 55.29 | **32.81 / 57.77** | 🥇 **NLLB-1.3B** (+2.83 BLEU over OPUS) |
| **Hausa (`ha`)** | Chadic | 26.53 / 49.51 | 28.17 / 50.77 | **30.53 / 53.25** | 🥇 **NLLB-1.3B** (+4.00 BLEU over OPUS) |
| **Ewe (`ee`)** | Gbe | **29.15 / 52.10** | 24.91 / 48.69 | 27.44 / 51.32 | 🥇 **OPUS-MT** (+1.71 BLEU over NLLB-1.3B) |
| **Igbo (`ig`)** | Igboid | 26.37 / 47.80 | 23.45 / 45.10 | **27.90 / 48.31** | 🥇 **NLLB-1.3B** (+1.53 BLEU over OPUS) |
| **Shona (`sn`)** | Bantu | 25.93 / 56.47 | 24.24 / 55.55 | **27.85 / 58.80** | 🥇 **NLLB-1.3B** (+1.92 BLEU over OPUS) |
| **Yoruba (`yo`)** | Defoid | **24.78 / 43.60** | 20.07 / 38.77 | 22.57 / 40.95 | 🥇 **OPUS-MT** (+2.21 BLEU over NLLB-1.3B) |

---

## 🏆 Direction-Sensitive Model Rankings per Target Language

### 🇬🇧 English $\rightarrow$ African Language Rankings:

#### 1. English $\rightarrow$ Twi / Akan (`tw`)
- 🥇 **Rank 1:** `AfriSpeech/afrigloss-nllb-1.3b` (BLEU: **27.37**, chrF2: **50.95**)
- 🥈 **Rank 2:** `AfriSpeech/afrigloss-nllb-600m` (BLEU: **24.49**, chrF2: **48.40**)
- 🥉 **Rank 3:** `AfriSpeech/afrigloss-opus-mt-en-tw` (BLEU: **23.21**, chrF2: **47.50**)
- *Recommendation:* Use **NLLB-1.3B** for highest translation accuracy and syntax; use **OPUS-MT** if deploying on CPU or mobile edge (<300 MB).

#### 2. English $\rightarrow$ Swahili (`sw`)
- 🥇 **Rank 1:** `AfriSpeech/afrigloss-nllb-1.3b` (BLEU: **19.17**, chrF2: **49.01**)
- 🥈 **Rank 2:** `AfriSpeech/afrigloss-nllb-600m` (BLEU: **18.27**, chrF2: **48.76**)
- 🥉 **Rank 3:** `AfriSpeech/afrigloss-opus-mt-en-sw` (BLEU: **13.20**, chrF2: **42.47**)
- *Recommendation:* Use **NLLB-1.3B** or **NLLB-600M**. Pretrained OPUS suffered from narrow biblical domain memorization, whereas NLLB generalized robustly to modern Bantu syntax.

#### 3. English $\rightarrow$ Yoruba (`yo`)
- 🥇 **Rank 1:** `AfriSpeech/afrigloss-opus-mt-en-yo` (BLEU: **26.54**, chrF2: **45.49**)
- 🥈 **Rank 2:** `AfriSpeech/afrigloss-nllb-1.3b` (BLEU: **21.39**, chrF2: **40.08**)
- 🥉 **Rank 3:** `AfriSpeech/afrigloss-nllb-600m` (BLEU: **18.51**, chrF2: **37.56**)
- *Recommendation:* **OPUS-MT is the clear winner for Yoruba.** Its dedicated vocabulary preserves tonal diacritics (`ọ́`, `è`, `à`, `ṣ`) without subword fragmentation.

#### 4. English $\rightarrow$ Hausa (`ha`)
- 🥇 **Rank 1:** `AfriSpeech/afrigloss-nllb-1.3b` (BLEU: **26.78**, chrF2: **51.08**)
- 🥈 **Rank 2:** `AfriSpeech/afrigloss-nllb-600m` (BLEU: **23.82**, chrF2: **49.27**)
- 🥉 **Rank 3:** `AfriSpeech/afrigloss-opus-mt-en-ha` (BLEU: **21.78**, chrF2: **46.77**)
- *Recommendation:* **NLLB-1.3B** produces the most natural Chadic sentence structures and handles borrowed terminology seamlessly.

#### 5. English $\rightarrow$ Igbo (`ig`)
- 🥇 **Rank 1:** `AfriSpeech/afrigloss-nllb-1.3b` (BLEU: **22.45**, chrF2: **50.05**)
- 🥈 **Rank 2:** `AfriSpeech/afrigloss-opus-mt-en-ig` (BLEU: **21.69**, chrF2: **47.78**)
- 🥉 **Rank 3:** `AfriSpeech/afrigloss-nllb-600m` (BLEU: **21.19**, chrF2: **48.00**)
- *Recommendation:* **NLLB-1.3B** for maximum fidelity; **OPUS-MT** is extremely close in BLEU while being 15$\times$ smaller.

#### 6. English $\rightarrow$ Ewe (`ee`)
- 🥇 **Rank 1:** `AfriSpeech/afrigloss-nllb-1.3b` (BLEU: **27.41**, chrF2: **53.69**)
- 🥈 **Rank 2:** `AfriSpeech/afrigloss-opus-mt-en-ee` (BLEU: **27.28**, chrF2: **51.38**)
- 🥉 **Rank 3:** `AfriSpeech/afrigloss-nllb-600m` (BLEU: **23.17**, chrF2: **50.09**)
- *Recommendation:* Both **NLLB-1.3B** and **OPUS-MT** perform exceptionally well on Ewe (>27 BLEU).

#### 7. English $\rightarrow$ Shona (`sn`)
- 🥇 **Rank 1:** `AfriSpeech/afrigloss-nllb-1.3b` (BLEU: **24.70**, chrF2: **56.46**)
- 🥈 **Rank 2:** `AfriSpeech/afrigloss-nllb-600m` (BLEU: **22.78**, chrF2: **55.51**)
- 🥉 **Rank 3:** `AfriSpeech/afrigloss-opus-mt-en-sn` (BLEU: **21.53**, chrF2: **53.21**)
- *Recommendation:* **NLLB-1.3B** handles Shona's 18 noun classes and concordial prefix agreements significantly better than smaller architectures.

---

### 🇫🇷 French $\rightarrow$ African Language Rankings:

#### 1. French $\rightarrow$ Twi / Akan (`tw`)
- 🥇 **Rank 1:** `AfriSpeech/afrigloss-nllb-1.3b` (BLEU: **31.49**, chrF2: **55.47**; Raw: **37.24 / 59.68**)
- 🥈 **Rank 2:** `AfriSpeech/afrigloss-nllb-600m` (BLEU: **28.78**, chrF2: **53.26**; Raw: **34.77 / 57.49**)
- 🥉 **Rank 3:** `AfriSpeech/afrigloss-opus-mt-fr-tw` (BLEU: **28.14**, chrF2: **52.59**; Raw: **33.58 / 56.77**)
- *Recommendation:* **NLLB-1.3B** delivers the highest accuracy and fluency, outperforming OPUS by **+3.35 BLEU**.

#### 2. French $\rightarrow$ Swahili (`sw`)
- 🥇 **Rank 1:** `AfriSpeech/afrigloss-nllb-1.3b` (BLEU: **32.81**, chrF2: **57.77**; Raw: **33.43 / 57.97**)
- 🥈 **Rank 2:** `AfriSpeech/afrigloss-opus-mt-fr-sw` (BLEU: **29.98**, chrF2: **56.16**; Raw: **30.78 / 56.95**)
- 🥉 **Rank 3:** `AfriSpeech/afrigloss-nllb-600m` (BLEU: **29.51**, chrF2: **55.29**; Raw: **30.07 / 56.18**)
- *Recommendation:* **NLLB-1.3B** is the strongest model overall (+2.83 BLEU over OPUS), while **OPUS-MT** is a great lightweight alternative (~30 BLEU).

#### 3. French $\rightarrow$ Hausa (`ha`)
- 🥇 **Rank 1:** `AfriSpeech/afrigloss-nllb-1.3b` (BLEU: **30.53**, chrF2: **53.25**; Raw: **30.31 / 52.89**)
- 🥈 **Rank 2:** `AfriSpeech/afrigloss-nllb-600m` (BLEU: **28.17**, chrF2: **50.77**; Raw: **28.86 / 51.15**)
- 🥉 **Rank 3:** `AfriSpeech/afrigloss-opus-mt-fr-ha` (BLEU: **26.53**, chrF2: **49.51**; Raw: **27.24 / 50.33**)
- *Recommendation:* **NLLB-1.3B** excels with +4.00 BLEU over OPUS, producing natural Chadic translations.

#### 4. French $\rightarrow$ Ewe (`ee`)
- 🥇 **Rank 1:** `AfriSpeech/afrigloss-opus-mt-fr-ee` (BLEU: **29.15**, chrF2: **52.10**; Raw: **29.91 / 52.87**)
- 🥈 **Rank 2:** `AfriSpeech/afrigloss-nllb-1.3b` (BLEU: **27.44**, chrF2: **51.32**; Raw: **28.33 / 51.82**)
- 🥉 **Rank 3:** `AfriSpeech/afrigloss-nllb-600m` (BLEU: **24.91**, chrF2: **48.69**; Raw: **25.44 / 49.34**)
- *Recommendation:* **OPUS-MT** is the top choice on Ewe (+1.71 BLEU over NLLB-1.3B).

#### 5. French $\rightarrow$ Igbo (`ig`)
- 🥇 **Rank 1:** `AfriSpeech/afrigloss-nllb-1.3b` (BLEU: **27.90**, chrF2: **48.31**; Raw: **28.86 / 49.10**)
- 🥈 **Rank 2:** `AfriSpeech/afrigloss-opus-mt-fr-ig` (BLEU: **26.37**, chrF2: **47.80**; Raw: **27.12 / 48.25**)
- 🥉 **Rank 3:** `AfriSpeech/afrigloss-nllb-600m` (BLEU: **23.45**, chrF2: **45.10**; Raw: **24.10 / 45.63**)
- *Recommendation:* **NLLB-1.3B** provides the highest grammatical fidelity (+1.53 BLEU over OPUS).

#### 6. French $\rightarrow$ Shona (`sn`)
- 🥇 **Rank 1:** `AfriSpeech/afrigloss-nllb-1.3b` (BLEU: **27.85**, chrF2: **58.80**; Raw: **28.21 / 58.78**)
- 🥈 **Rank 2:** `AfriSpeech/afrigloss-opus-mt-fr-sn` (BLEU: **25.93**, chrF2: **56.47**; Raw: **26.64 / 56.99**)
- 🥉 **Rank 3:** `AfriSpeech/afrigloss-nllb-600m` (BLEU: **24.24**, chrF2: **55.55**; Raw: **25.28 / 56.36**)
- *Recommendation:* **NLLB-1.3B** handles Bantu noun classes and prefix concords with superior subword character metrics (58.80 chrF2).

#### 7. French $\rightarrow$ Yoruba (`yo`)
- 🥇 **Rank 1:** `AfriSpeech/afrigloss-opus-mt-fr-yo` (BLEU: **24.78**, chrF2: **43.60**; Raw: **26.29 / 44.93**)
- 🥈 **Rank 2:** `AfriSpeech/afrigloss-nllb-1.3b` (BLEU: **22.57**, chrF2: **40.95**; Raw: **23.57 / 41.92**)
- 🥉 **Rank 3:** `AfriSpeech/afrigloss-nllb-600m` (BLEU: **20.07**, chrF2: **38.77**; Raw: **21.11 / 39.60**)
- *Recommendation:* **OPUS-MT** is essential for Yoruba tonal integrity (+2.21 BLEU over NLLB-1.3B).

---

## 🛠️ Production Selection Matrix

| Architectural Family | Parameters | Model Size | Best Suited For | Top Languages |
| :--- | :---: | :---: | :--- | :--- |
| **Afri-Gloss NLLB-1.3B** | 1.3 Billion | ~1.1 GB (LoRA) / ~5.2 GB (Full) | Highest translation fluency, complex Bantu noun classes, enterprise APIs | Twi, Hausa, Shona, Ewe, Swahili |
| **Afri-Gloss NLLB-600M** | 600 Million | ~35 MB (LoRA) / ~2.4 GB (Full) | Balanced multilingual serving, multi-source translation (EN + FR) | Swahili, Hausa, Twi, Shona |
| **Afri-Gloss OPUS-MT** | 77 Million | **~280 MB** | Ultra-low latency, CPU/edge deployment, specialized tonal diacritics | **Yoruba**, Ewe, Twi |

---

## 🔗 Hugging Face Model Hub Links

All production models are available under the **CC-BY-NC 4.0** license:

- **Hugging Face Collection**: [AfriSpeech/afri-gloss-mt-models](https://huggingface.co/collections/AfriSpeech/afri-gloss-mt-models-6adce3a3c13c6a86f07b6e6d)
- **Multilingual Foundation Models**:
  - [`AfriSpeech/afrigloss-nllb-1.3b`](https://huggingface.co/AfriSpeech/afrigloss-nllb-1.3b)
  - [`AfriSpeech/afrigloss-nllb-600m`](https://huggingface.co/AfriSpeech/afrigloss-nllb-600m)
- **Dedicated OPUS-MT Bilingual Models**:
  - English $\rightarrow$ African: `afrigloss-opus-mt-en-tw`, `en-sw`, `en-yo`, `en-ha`, `en-ig`, `en-ee`, `en-sn`
  - French $\rightarrow$ African: `afrigloss-opus-mt-fr-tw`, `fr-sw`, `fr-yo`, `fr-ha`, `fr-ig`, `fr-ee`, `fr-sn`
