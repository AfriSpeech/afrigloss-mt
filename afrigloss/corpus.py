"""
Corpus ingestion and parallel text retrieval for Afri-Gloss-MT.

Integrates with AfriSpeech/africa-corpus (hosted on Hugging Face) and
AfriSpeech/africa-corpus-builder to retrieve parallel sentence pairs
for 693 African languages aligned with English, French, Spanish, Portuguese,
or directly between African languages.
"""

import csv
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

from huggingface_hub import hf_hub_download, list_repo_files

from afrigloss.registry import AFRICA_CORPUS_REPO, normalize_lang_code

# Max field size for reading long CSV lines
csv.field_size_limit(10_000_000)

CACHE_DIR = Path(os.environ.get("AFRIGLOSS_CACHE_DIR", Path.home() / ".cache" / "afrigloss" / "corpus"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class ParallelPair:
    key: str
    src: str
    tgt: str


class AfricaCorpusLoader:
    """Retrieves and aligns parallel texts from AfriSpeech/africa-corpus."""

    def __init__(self, repo_id: str = AFRICA_CORPUS_REPO, cache_dir: Optional[Path] = None):
        self.repo_id = repo_id
        self.cache_dir = Path(cache_dir) if cache_dir else CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._repo_files: Optional[List[str]] = None
        self._lang_to_file: Optional[Dict[str, str]] = None

    def _discover_files(self) -> Dict[str, str]:
        """Index all language CSV files available in the Hugging Face dataset repo."""
        if self._lang_to_file is not None:
            return self._lang_to_file

        files = list_repo_files(self.repo_id, repo_type="dataset")
        self._repo_files = files
        lang_to_file: Dict[str, str] = {}

        # Files are named like: Language_iso3_version.csv (e.g. Twi_twi_v1234.csv, Afrikaans_afr_v117.csv)
        pattern = re.compile(r"^.+?_([a-z]{3})_v\d+\.csv$", re.IGNORECASE)
        for f in files:
            m = pattern.match(f)
            if m:
                iso3 = m.group(1).lower()
                # Store first or preferred file for this language
                if iso3 not in lang_to_file:
                    lang_to_file[iso3] = f

        # Also map English cache if present
        if "english_cache.csv" in files:
            lang_to_file["eng"] = "english_cache.csv"

        self._lang_to_file = lang_to_file
        return lang_to_file

    def list_available_languages(self) -> List[str]:
        """Return all ISO 639-3 language codes available in the corpus."""
        mapping = self._discover_files()
        return sorted(list(mapping.keys()))

    def get_csv_path(self, lang_code: str) -> Path:
        """Download (cached) the CSV file for a given language code."""
        code = normalize_lang_code(lang_code)
        mapping = self._discover_files()

        if code not in mapping:
            # Special check for English
            if code in {"eng", "en"}:
                filename = "english_cache.csv"
            else:
                raise ValueError(
                    f"Language code '{lang_code}' (normalized '{code}') not found in {self.repo_id}."
                )
        else:
            filename = mapping[code]

        # Download from Hugging Face Hub (cached automatically)
        local_path = hf_hub_download(
            repo_id=self.repo_id,
            filename=filename,
            repo_type="dataset",
            cache_dir=self.cache_dir,
        )
        return Path(local_path)

    def load_verses(self, lang_code: str) -> Dict[str, str]:
        """Load verse_key -> verse_text dictionary for a language."""
        csv_path = self.get_csv_path(lang_code)
        verses: Dict[str, str] = {}

        with open(csv_path, mode="r", encoding="utf-8-sig", errors="replace") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if not header:
                return verses

            # Find columns
            key_idx = 0
            text_idx = 1
            for i, col in enumerate(header):
                cl = col.lower()
                if "key" in cl or "verse" in cl:
                    key_idx = i
                elif any(k in cl for k in ["local", "eng", "text", "sentence", "content"]):
                    text_idx = i
            # If still not found and last column is not key, prefer last column
            if text_idx == key_idx and len(header) > 1:
                text_idx = len(header) - 1
            elif text_idx == 1 and "version" in header[1].lower() and len(header) > 2:
                text_idx = 2

            # If not found by keyword, default to last column
            if text_idx == -1:
                text_idx = len(header) - 1

            for row in reader:
                if len(row) > max(key_idx, text_idx):
                    k = row[key_idx].strip()
                    t = clean_text(row[text_idx])
                    if k and t:
                        verses[k] = t

        return verses

    def get_parallel_pairs(
        self,
        src_lang: str,
        tgt_lang: str,
        max_samples: Optional[int] = None,
    ) -> List[ParallelPair]:
        """
        Extract aligned sentence pairs for src_lang <-> tgt_lang.
        Both languages are joined on their shared verse_key.
        """
        src_code = normalize_lang_code(src_lang)
        tgt_code = normalize_lang_code(tgt_lang)

        src_verses = self.load_verses(src_code)
        tgt_verses = self.load_verses(tgt_code)

        # Inner join on common keys
        common_keys = sorted(list(set(src_verses.keys()).intersection(tgt_verses.keys())))

        pairs: List[ParallelPair] = []
        for k in common_keys:
            s_text = src_verses[k]
            t_text = tgt_verses[k]
            if s_text and t_text and len(s_text.split()) >= 2 and len(t_text.split()) >= 2:
                pairs.append(ParallelPair(key=k, src=s_text, tgt=t_text))
                if max_samples and len(pairs) >= max_samples:
                    break

        return pairs


def clean_text(text: str) -> str:
    """Clean verse text by stripping verse markers and excess whitespace."""
    t = text.strip()
    # Remove leading verse number digits like "1 ", "12 ", etc.
    t = re.sub(r"^\d+\s+", "", t)
    # Normalize spaces
    t = re.sub(r"\s+", " ", t).strip()
    return t
