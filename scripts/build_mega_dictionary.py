"""
Mega African Dictionary Builder for Afri-Gloss-MT.

Translates high-frequency English nouns from AfriSpeech/africa-nouns
(1.41 million African-domain English nouns) into African languages using Gemini.

Features:
  - Batched requests (50-100 nouns per API call) for ultra-fast throughput
  - Persistent SQLite caching (resumable, skips already-translated nouns)
  - Frequency tier filtering (e.g. top 10k, 50k, 100k nouns by corpus frequency)
  - Exports clean, sorted TSV dictionaries ready for Gloss-MT training and inference

Usage:
  python scripts/build_mega_dictionary.py --tgt twi --min-freq 1000 --limit 10000
  python scripts/build_mega_dictionary.py --tgt swh --min-freq 500
"""

import argparse
import json
import logging
import os
import sqlite3
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from afrigloss.registry import LANGUAGE_INFO, normalize_lang_code

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("mega_dict")

GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
DEFAULT_API_KEY = os.environ.get("GEMINI_API_KEY", "")

TARGET_LANGUAGE_PROMPTS = {
    "twi": "Asante Twi (Akan, Ghana)",
    "aka": "Asante Twi (Akan, Ghana)",
    "yor": "Yoruba (Nigeria)",
    "swh": "Swahili (East Africa)",
    "hau": "Hausa (Nigeria/Niger, written in Latin Boko script)",
    "ibo": "Igbo (Nigeria)",
    "ewe": "Ewe (Ghana/Togo)",
    "sna": "Shona (Zimbabwe)",
    "nya": "Chichewa / Nyanja (Malawi/Zambia)",
    "lug": "Luganda (Uganda)",
    "kin": "Kinyarwanda (Rwanda)",
    "lin": "Lingala (DRC/Congo)",
    "zul": "Zulu (South Africa)",
    "xho": "Xhosa (South Africa)",
    "som": "Somali (Somalia)",
    "amh": "Amharic (Ethiopia)",
}


class MegaDictionaryDB:
    """Persistent SQLite database for storing and caching translated nouns."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()
            cur.execute("""
                CREATE TABLE IF NOT EXISTS translations (
                    en_noun TEXT,
                    target_lang TEXT,
                    translation TEXT,
                    frequency REAL,
                    PRIMARY KEY (en_noun, target_lang)
                )
            """)
            cur.execute("PRAGMA journal_mode = WAL")
            conn.commit()

    def get_cached(self, nouns: List[str], target_lang: str) -> Dict[str, str]:
        """Fetch all cached translations for a list of nouns in a single query."""
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()
            placeholders = ",".join("?" for _ in nouns)
            cur.execute(
                f"SELECT en_noun, translation FROM translations WHERE target_lang = ? AND en_noun IN ({placeholders})",
                [target_lang] + nouns,
            )
            return dict(cur.fetchall())

    def save_batch(self, records: List[Tuple[str, str, str, float]]):
        """Save a batch of (en_noun, target_lang, translation, frequency) records."""
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()
            cur.executemany(
                "INSERT OR REPLACE INTO translations (en_noun, target_lang, translation, frequency) VALUES (?, ?, ?, ?)",
                records,
            )
            conn.commit()

    def export_tsv(self, target_lang: str, out_tsv_path: Path):
        """Export all cached translations for a language to a clean TSV dictionary."""
        out_tsv_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT en_noun, translation, frequency FROM translations WHERE target_lang = ? ORDER BY frequency DESC",
                (target_lang,),
            )
            rows = cur.fetchall()

        with open(out_tsv_path, "w", encoding="utf-8") as f:
            f.write("source_phrase\ttarget_phrase\tconfidence\tfrequency\n")
            for en, trans, freq in rows:
                if trans and trans.strip():
                    f.write(f"{en.strip()}\t{trans.strip()}\t0.9500\t{int(freq)}\n")

        logger.info(f"Exported {len(rows):,} entries for {target_lang} -> {out_tsv_path}")


def translate_noun_batch_gemini(
    nouns: List[str],
    target_lang_name: str,
    api_key: str,
    model: str = GEMINI_MODEL,
    max_retries: int = 3,
) -> Dict[str, str]:
    """Call Gemini to translate a batch of English nouns into the target African language."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"

    prompt = (
        f"Translate the following English noun phrases into natural, standard {target_lang_name}.\n"
        f"Return ONLY a valid JSON dictionary where the keys are the exact English nouns and values are their standard translations.\n"
        f"If a term is an untranslatable proper name or foreign noun, keep the standard local spelling.\n"
        f"English nouns:\n{json.dumps(nouns, ensure_ascii=False)}"
    )

    req_body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"response_mime_type": "application/json"},
    }

    req = urllib.request.Request(
        url,
        data=json.dumps(req_body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )

    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode())
                text = data["candidates"][0]["content"]["parts"][0]["text"]
                result = json.loads(text)
                if isinstance(result, dict):
                    return {k.strip().lower(): str(v).strip() for k, v in result.items() if v}
                return {}
        except Exception as e:
            if attempt == max_retries - 1:
                logger.warning(f"Batch failed after {max_retries} attempts: {e}")
                return {}
            time.sleep(1.0 + attempt * 2)

    return {}


def build_mega_dictionary(
    target_lang: str,
    min_freq: float = 1000.0,
    limit: Optional[int] = None,
    batch_size: int = 50,
    max_workers: int = 5,
    api_key: str = DEFAULT_API_KEY,
    db_path: Path = REPO_ROOT / "data" / "mega_dictionary.db",
    out_dir: Path = REPO_ROOT / "data" / "dictionaries",
):
    """Main pipeline for building the Mega African Dictionary from africa-nouns-en.parquet."""
    if not api_key:
        api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is required. Pass via --api-key or set GEMINI_API_KEY environment variable.")

    tgt_code = normalize_lang_code(target_lang)
    lang_name = TARGET_LANGUAGE_PROMPTS.get(tgt_code, LANGUAGE_INFO.get(tgt_code, {}).get("name", tgt_code.upper()))

    logger.info(f"Target language: {lang_name} ({tgt_code})")
    db = MegaDictionaryDB(db_path)

    # 1. Download/load africa-nouns-en.parquet
    logger.info("Loading AfriSpeech/africa-nouns/data/africa-nouns-en.parquet...")
    local_parquet = hf_hub_download(
        repo_id="AfriSpeech/africa-nouns",
        filename="data/africa-nouns-en.parquet",
        repo_type="dataset",
    )
    table = pq.read_table(local_parquet, columns=["phrase", "total_count"])
    df = table.to_pydict()
    all_phrases = df["phrase"]
    all_counts = df["total_count"]

    # Filter by frequency
    candidates = []
    for p, c in zip(all_phrases, all_counts):
        if c >= min_freq and len(p.strip()) >= 2:
            candidates.append((p.strip().lower(), c))

    logger.info(f"Found {len(candidates):,} nouns with total_count >= {min_freq:,.0f}")
    if limit and len(candidates) > limit:
        candidates = candidates[:limit]
        logger.info(f"Limiting to top {limit:,} nouns")

    # 2. Check cached nouns
    noun_strings = [p for p, _ in candidates]
    noun_freq_map = {p: c for p, c in candidates}
    cached_map = db.get_cached(noun_strings, tgt_code)
    logger.info(f"Already cached in database: {len(cached_map):,} / {len(candidates):,}")

    todo_nouns = [n for n in noun_strings if n not in cached_map]
    logger.info(f"Nouns to translate with Gemini: {len(todo_nouns):,}")

    if todo_nouns:
        # Create batches
        batches = [todo_nouns[i : i + batch_size] for i in range(0, len(todo_nouns), batch_size)]
        logger.info(f"Split into {len(batches):,} batches of ~{batch_size} nouns each (workers={max_workers})")

        completed = 0
        t0 = time.time()

        def process_batch(batch_nouns):
            trans_dict = translate_noun_batch_gemini(batch_nouns, lang_name, api_key)
            records = []
            for en_word in batch_nouns:
                trans = trans_dict.get(en_word)
                if trans:
                    records.append((en_word, tgt_code, trans, noun_freq_map.get(en_word, 1.0)))
            if records:
                db.save_batch(records)
            return len(records)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_batch, b) for b in batches]
            for f in as_completed(futures):
                count = f.result()
                completed += 1
                if completed % 10 == 0 or completed == len(batches):
                    speed = (completed * batch_size) / (time.time() - t0 + 1e-5)
                    logger.info(f"Progress: {completed}/{len(batches)} batches ({completed/len(batches)*100:.1f}%) | {speed:.1f} nouns/s")

    # 3. Export to clean TSV dictionary
    out_tsv = out_dir / f"mega_eng_{tgt_code}.tsv"
    db.export_tsv(tgt_code, out_tsv)
    logger.info(f"Mega Dictionary for {lang_name} successfully built: {out_tsv}")


def main():
    parser = argparse.ArgumentParser(description="Build Mega African Dictionary using Gemini & africa-nouns-en.parquet")
    parser.add_argument("--tgt", required=True, help="Target African language (e.g. tw, sw, yo, ha, ig)")
    parser.add_argument("--min-freq", type=float, default=1000.0, help="Minimum corpus frequency threshold (default: 1000)")
    parser.add_argument("--limit", type=int, default=None, help="Limit to top N nouns (e.g. 10000)")
    parser.add_argument("--batch-size", type=int, default=50, help="Batch size per Gemini API call")
    parser.add_argument("--workers", type=int, default=5, help="Concurrent worker threads")
    parser.add_argument("--api-key", default=DEFAULT_API_KEY, help="Gemini API Key")

    args = parser.parse_args()
    build_mega_dictionary(
        target_lang=args.tgt,
        min_freq=args.min_freq,
        limit=args.limit,
        batch_size=args.batch_size,
        max_workers=args.workers,
        api_key=args.api_key,
    )


if __name__ == "__main__":
    main()
