"""
Corpus ingestion and parallel text retrieval for Afri-Gloss-MT.

Integrates with:
  1. AfriSpeech/africa-corpus (693 African languages, multi-version Bible texts).
  2. AfriSpeech/african-corpus-jw (Large-scale parallel texts, 100k-250k sentences/lang).
  3. Custom User Datasets (.tsv, .csv, .jsonl, .parquet, or paired .txt files).

Allows users to build Gloss-MT datasets either completely zero-setup from online
African corpora, or by supplying their own parallel domain data.
"""

import csv
import gzip
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple, Union

from huggingface_hub import hf_hub_download, list_repo_files

from afrigloss.registry import AFRICA_CORPUS_REPO, normalize_lang_code

csv.field_size_limit(10_000_000)

CACHE_DIR = Path(os.environ.get("AFRIGLOSS_CACHE_DIR", Path.home() / ".cache" / "afrigloss" / "corpus"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)

AFRICAN_JW_REPO = "AfriSpeech/african-corpus-jw"


@dataclass
class ParallelPair:
    key: str
    src: str
    tgt: str


class AfricaCorpusLoader:
    """Retrieves and aligns parallel texts from African corpora and user files."""

    def __init__(
        self,
        repo_id: str = AFRICA_CORPUS_REPO,
        jw_repo_id: str = AFRICAN_JW_REPO,
        cache_dir: Optional[Path] = None,
    ):
        self.repo_id = repo_id
        self.jw_repo_id = jw_repo_id
        self.cache_dir = Path(cache_dir) if cache_dir else CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._lang_to_files: Optional[Dict[str, List[str]]] = None
        self._jw_files: Optional[Dict[str, str]] = None

    def _discover_files(self) -> Dict[str, List[str]]:
        """Index all language CSV files available in AfriSpeech/africa-corpus."""
        if self._lang_to_files is not None:
            return self._lang_to_files

        files = list_repo_files(self.repo_id, repo_type="dataset")
        lang_to_files: Dict[str, List[str]] = {}

        pattern = re.compile(r"^.+?_([a-z]{3})_v\d+\.csv$", re.IGNORECASE)
        sorted_files = sorted(files, key=lambda x: ("arabic_script" in x.lower(), x))

        for f in sorted_files:
            m = pattern.match(f)
            if m:
                iso3 = m.group(1).lower()
                if "arabic_script" in f.lower() and iso3 in lang_to_files:
                    continue
                lang_to_files.setdefault(iso3, []).append(f)

        if "english_cache.csv" in files:
            lang_to_files["eng"] = ["english_cache.csv"]

        self._lang_to_files = lang_to_files
        return lang_to_files

    def _discover_jw_files(self) -> Dict[str, str]:
        """Index available language CSV files in AfriSpeech/african-corpus-jw."""
        if self._jw_files is not None:
            return self._jw_files

        try:
            files = list_repo_files(self.jw_repo_id, repo_type="dataset")
        except Exception:
            self._jw_files = {}
            return self._jw_files

        jw_map: Dict[str, str] = {}
        # Format: e.g. Twi_tw_v1.csv or Swahili_sw_v1.csv
        pattern = re.compile(r"^.+?_([a-z]{2,3})_v\d+\.csv$", re.IGNORECASE)
        for f in files:
            m = pattern.match(f)
            if m:
                code = normalize_lang_code(m.group(1).lower())
                jw_map[code] = f

        if "english_cache.csv" in files:
            jw_map["eng"] = "english_cache.csv"
            jw_map["en"] = "english_cache.csv"

        self._jw_files = jw_map
        return self._jw_files

    def load_verses_from_file(self, filename: str, repo_id: str) -> Dict[str, str]:
        """Download and load verses from a specific CSV file in a Hugging Face repo."""
        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="dataset",
            cache_dir=self.cache_dir,
        )
        verses: Dict[str, str] = {}
        with open(local_path, mode="r", encoding="utf-8-sig", errors="replace") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if not header:
                return verses

            key_idx = 0
            text_idx = 1
            for i, col in enumerate(header):
                cl = col.lower()
                if "key" in cl or "verse" in cl:
                    key_idx = i
                elif any(k in cl for k in ["local", "eng", "text", "sentence", "content"]):
                    text_idx = i

            if text_idx == key_idx and len(header) > 1:
                text_idx = len(header) - 1
            elif text_idx == 1 and "version" in header[1].lower() and len(header) > 2:
                text_idx = 2

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
        include_all_versions: bool = True,
        include_jw: bool = True,
    ) -> List[ParallelPair]:
        """
        Extract aligned sentence pairs for src_lang <-> tgt_lang.
        Aggregates across multiple translation versions and JW300 to reach 100k+ samples.
        """
        src_code = normalize_lang_code(src_lang)
        tgt_code = normalize_lang_code(tgt_lang)

        file_mapping = self._discover_files()
        jw_mapping = self._discover_jw_files() if include_jw else {}

        src_files = file_mapping.get(src_code, ["english_cache.csv"] if src_code in ("eng", "en") else [])
        tgt_files = file_mapping.get(tgt_code, ["english_cache.csv"] if tgt_code in ("eng", "en") else [])

        if not src_files and src_code not in jw_mapping:
            raise ValueError(f"Source language '{src_lang}' not found in African corpora.")
        if not tgt_files and tgt_code not in jw_mapping:
            raise ValueError(f"Target language '{tgt_lang}' not found in African corpora.")

        pairs: List[ParallelPair] = []
        seen_keys: Set[Tuple[str, str]] = set()

        # 1. Primary Bible version join
        primary_src_file = src_files[0] if src_files else None
        primary_tgt_file = tgt_files[0] if tgt_files else None

        if primary_src_file and primary_tgt_file:
            src_verses = self.load_verses_from_file(primary_src_file, self.repo_id)
            tgt_verses = self.load_verses_from_file(primary_tgt_file, self.repo_id)

            common_keys = sorted(list(set(src_verses.keys()).intersection(tgt_verses.keys())))
            for k in common_keys:
                s = src_verses[k]
                t = tgt_verses[k]
                if s and t and len(s.split()) >= 2 and len(t.split()) >= 2:
                    if (s, t) not in seen_keys:
                        seen_keys.add((s, t))
                        pairs.append(ParallelPair(key=k, src=s, tgt=t))
                        if max_samples and len(pairs) >= max_samples:
                            return pairs

        # 2. Multi-version Bible iterations if more samples needed
        if include_all_versions and (not max_samples or len(pairs) < max_samples):
            # Iterate through additional version files
            for tgt_f in tgt_files[1:]:
                tgt_verses = self.load_verses_from_file(tgt_f, self.repo_id)
                common = sorted(list(set(src_verses.keys()).intersection(tgt_verses.keys())))
                for k in common:
                    s = src_verses[k]
                    t = tgt_verses[k]
                    if s and t and len(s.split()) >= 2 and len(t.split()) >= 2:
                        if (s, t) not in seen_keys:
                            seen_keys.add((s, t))
                            pairs.append(ParallelPair(key=f"{k}_{tgt_f}", src=s, tgt=t))
                            if max_samples and len(pairs) >= max_samples:
                                return pairs

        # 3. JW300 dataset join if more samples needed
        if include_jw and (not max_samples or len(pairs) < max_samples):
            jw_src_f = jw_mapping.get(src_code)
            jw_tgt_f = jw_mapping.get(tgt_code)

            if jw_src_f and jw_tgt_f:
                try:
                    jw_src = self.load_verses_from_file(jw_src_f, self.jw_repo_id)
                    jw_tgt = self.load_verses_from_file(jw_tgt_f, self.jw_repo_id)
                    common_jw = sorted(list(set(jw_src.keys()).intersection(jw_tgt.keys())))

                    for k in common_jw:
                        s = jw_src[k]
                        t = jw_tgt[k]
                        if s and t and len(s.split()) >= 2 and len(t.split()) >= 2:
                            if (s, t) not in seen_keys:
                                seen_keys.add((s, t))
                                pairs.append(ParallelPair(key=f"jw_{k}", src=s, tgt=t))
                                if max_samples and len(pairs) >= max_samples:
                                    return pairs
                except Exception as e:
                    pass

        return pairs


def load_custom_parallel_file(
    file_path: str | Path,
    src_lang: Optional[str] = None,
    tgt_lang: Optional[str] = None,
    max_samples: Optional[int] = None,
) -> List[ParallelPair]:
    """
    Load user-supplied parallel dataset from a file (.tsv, .csv, .jsonl, .parquet).

    Automatically identifies source and target columns or handles 2-column tabular data.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Custom parallel file not found: {path}")

    pairs: List[ParallelPair] = []
    suffix = path.suffix.lower()

    if suffix in (".tsv", ".csv"):
        delimiter = "\t" if suffix == ".tsv" else ","
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f, delimiter=delimiter)
            header = next(reader, None)
            if not header:
                return []

            # Check if header contains column names or is already data
            src_idx = 0
            tgt_idx = 1
            has_header = False

            for i, col in enumerate(header):
                cl = col.lower()
                if any(k in cl for k in ("src", "source", "en", "eng", "input")):
                    src_idx = i
                    has_header = True
                elif any(k in cl for k in ("tgt", "target", "trans", "output", "label")):
                    tgt_idx = i
                    has_header = True

            if not has_header:
                # Treat first row as data
                if len(header) >= 2:
                    pairs.append(ParallelPair(key="0", src=clean_text(header[0]), tgt=clean_text(header[1])))

            for idx, row in enumerate(reader, 1):
                if len(row) > max(src_idx, tgt_idx):
                    s = clean_text(row[src_idx])
                    t = clean_text(row[tgt_idx])
                    if s and t and len(s.split()) >= 2 and len(t.split()) >= 2:
                        pairs.append(ParallelPair(key=str(idx), src=s, tgt=t))
                        if max_samples and len(pairs) >= max_samples:
                            break

    elif suffix == ".jsonl":
        with open(path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f, 1):
                if line.strip():
                    item = json.loads(line)
                    s = item.get("src") or item.get("source") or item.get("input") or item.get("en")
                    t = item.get("tgt") or item.get("target") or item.get("translation")
                    if s and t:
                        pairs.append(ParallelPair(key=str(idx), src=clean_text(s), tgt=clean_text(t)))
                        if max_samples and len(pairs) >= max_samples:
                            break

    elif suffix == ".parquet":
        import pyarrow.parquet as pq
        table = pq.read_table(str(path))
        df = table.to_pydict()
        # Find columns
        src_col = next((c for c in df if any(k in c.lower() for k in ("src", "source", "en", "eng"))), list(df.keys())[0])
        tgt_col = next((c for c in df if any(k in c.lower() for k in ("tgt", "target", "trans"))), list(df.keys())[1])

        for idx, (s, t) in enumerate(zip(df[src_col], df[tgt_col]), 1):
            if s and t:
                pairs.append(ParallelPair(key=str(idx), src=clean_text(str(s)), tgt=clean_text(str(t))))
                if max_samples and len(pairs) >= max_samples:
                    break

    return pairs


def clean_text(text: str) -> str:
    """Clean verse text by stripping verse markers and excess whitespace."""
    t = text.strip()
    t = re.sub(r"^\d+\s+", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t
