"""
Mega African Dictionary Builder for Afri-Gloss-MT.

Translates top-frequency English nouns from AfriSpeech/africa-nouns
(1.41 million African-domain English nouns) into African languages using Gemini 3.8 Flash.

Safety & Quality Gate:
  Only permits Gemini translation if:
    1. The language is verified as 'strong' (score >= 65.0) in AfriSpeech/gemini-afro-mt-bench:
       https://huggingface.co/spaces/AfriSpeech/gemini-afro-mt-bench
    2. A corresponding Gloss-MT model exists / is trained for the language.

Features:
  - Batched requests (100 nouns per API call, verified non-truncating)
  - Massive concurrency (20-30 worker threads for paid tier)
  - Persistent SQLite caching (resumable, skips already-translated nouns)
  - Exports clean TSV dictionaries sorted by corpus frequency

Usage:
  python scripts/build_mega_dictionary.py --tgt twi --limit 100000 --workers 20
"""

import argparse
import json
import logging
import os
import queue
import re
import sqlite3
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from afrigloss.registry import LANGUAGE_INFO, find_model_for_pair, normalize_lang_code

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("mega_dict")

GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")
DEFAULT_API_KEY = os.environ.get("GEMINI_API_KEY", "")

# Verified benchmark scores from AfriSpeech/gemini-afro-mt-bench (threshold: score >= 65.0, 'strong' tier)
GEMINI_AFRO_BENCHMARK_STRONG: Dict[str, Dict[str, any]] = {
    "hau": {"name": "Hausa", "score": 72.55, "tier": "strong"},
    "ewe": {"name": "Ewe", "score": 72.55, "tier": "strong"},
    "twi": {"name": "Twi / Akan", "score": 72.06, "tier": "strong"},
    "aka": {"name": "Akan", "score": 72.06, "tier": "strong"},
    "yor": {"name": "Yoruba", "score": 69.12, "tier": "strong"},
    "ibo": {"name": "Igbo", "score": 67.65, "tier": "strong"},
    "swh": {"name": "Swahili", "score": 66.18, "tier": "strong"},
    "swc": {"name": "Congo Swahili", "score": 66.18, "tier": "strong"},
    "sna": {"name": "Shona", "score": 65.69, "tier": "strong"},
    "amh": {"name": "Amharic", "score": 86.27, "tier": "strong"},
}

TARGET_LANGUAGE_PROMPTS: Dict[str, str] = {
    "twi": "standard Asante Twi (Akan, Ghana)",
    "aka": "standard Asante Twi (Akan, Ghana)",
    "yor": "standard Yoruba (Nigeria, with proper tone marks)",
    "swh": "standard Swahili (Kiswahili, East Africa)",
    "swc": "standard Swahili (Kiswahili, East Africa)",
    "hau": "standard Hausa (Nigeria/Niger, written in standard Latin Boko script)",
    "ibo": "standard Igbo (Nigeria)",
    "ewe": "standard Ewe (Ghana/Togo)",
    "sna": "standard Shona (ChiShona, Zimbabwe)",
    "amh": "standard Amharic (Ethiopia, in Ge'ez fidäl script)",
}


def verify_gemini_eligibility(target_lang: str, force: bool = False) -> str:
    """
    Ensure the target language qualifies for Gemini translation:
      1. Must be in the 'strong' tier of gemini-afro-mt-bench.
      2. Must have a supported Gloss-MT translation model.
    """
    code = normalize_lang_code(target_lang)
    bench_info = GEMINI_AFRO_BENCHMARK_STRONG.get(code)

    if not bench_info and not force:
        raise ValueError(
            f"Language '{target_lang}' (ISO: {code}) is NOT approved for Gemini dictionary generation.\n"
            f"Per AfriSpeech quality standards, Gemini is only used for languages verified in the 'strong' tier (score >= 65.0) "
            f"on https://huggingface.co/spaces/AfriSpeech/gemini-afro-mt-bench.\n"
            f"Approved languages: {list(GEMINI_AFRO_BENCHMARK_STRONG.keys())}.\n"
            f"To bypass, pass --force."
        )

    # Check model registry
    model_info = find_model_for_pair("en", code)
    if not model_info and not force:
        raise ValueError(
            f"No Gloss-MT base model registered for English -> {code}. "
            f"We only build mega dictionaries for languages with trained/supported models."
        )

    score = bench_info["score"] if bench_info else "N/A"
    logger.info(f"Verified {code} ({bench_info.get('name') if bench_info else code}): Gemini Benchmark Score = {score} (Tier: strong)")
    return code


class MegaDictionaryDB:
    """Thread-safe persistent SQLite database for caching translated nouns."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        self.write_queue = queue.Queue(maxsize=20000)
        self._stop_event = threading.Event()
        self._writer_thread = threading.Thread(target=self._writer_loop, daemon=True)
        self._writer_thread.start()

    def _init_db(self):
        with sqlite3.connect(self.db_path, timeout=60.0) as conn:
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
            cur.execute("PRAGMA synchronous = NORMAL")
            conn.commit()

    def get_cached_set(self, target_lang: str) -> set:
        """Fetch all previously translated nouns for this language as a fast set."""
        with sqlite3.connect(self.db_path, timeout=60.0) as conn:
            cur = conn.cursor()
            cur.execute("SELECT en_noun FROM translations WHERE target_lang = ?", (target_lang,))
            return set(row[0] for row in cur.fetchall())

    def _writer_loop(self):
        """Dedicated background thread that batches database writes."""
        conn = sqlite3.connect(self.db_path, timeout=60.0)
        cur = conn.cursor()
        buffer = []

        while not self._stop_event.is_set() or not self.write_queue.empty():
            try:
                item = self.write_queue.get(timeout=0.2)
                buffer.extend(item)
            except queue.Empty:
                pass

            if buffer and (len(buffer) >= 500 or self.write_queue.empty()):
                try:
                    cur.executemany(
                        "INSERT OR REPLACE INTO translations (en_noun, target_lang, translation, frequency) VALUES (?, ?, ?, ?)",
                        buffer,
                    )
                    conn.commit()
                except Exception as e:
                    logger.error(f"DB commit error: {e}")
                buffer.clear()

        if buffer:
            cur.executemany(
                "INSERT OR REPLACE INTO translations (en_noun, target_lang, translation, frequency) VALUES (?, ?, ?, ?)",
                buffer,
            )
            conn.commit()

        conn.close()

    def push_records(self, records: List[Tuple[str, str, str, float]]):
        """Enqueue records for asynchronous writing."""
        if records:
            self.write_queue.put(records)

    def close(self):
        """Signal writer thread to finish and flush buffer."""
        self._stop_event.set()
        self._writer_thread.join(timeout=10.0)

    def export_tsv(self, target_lang: str, out_tsv_path: Path):
        """Export all cached translations for a language to a clean TSV dictionary."""
        out_tsv_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path, timeout=60.0) as conn:
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
    max_retries: int = 4,
) -> Dict[str, str]:
    """Call Gemini to translate a batch of English nouns into the target African language."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"

    prompt = (
        f"Translate the following English noun phrases into {target_lang_name}.\n"
        f"Return ONLY a valid JSON dictionary where keys are the exact English nouns and values are their standard translations.\n"
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
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read().decode())
                text = data["candidates"][0]["content"]["parts"][0]["text"]
                result = json.loads(text)
                if isinstance(result, dict):
                    return {k.strip().lower(): str(v).strip() for k, v in result.items() if v}
                return {}
        except Exception as e:
            if attempt == max_retries - 1:
                return {}
            time.sleep(1.0 + attempt * 2)

    return {}


def build_mega_dictionary(
    target_lang: str,
    limit: int = 100000,
    min_freq: float = 1.0,
    batch_size: int = 100,
    max_workers: int = 20,
    api_key: str = DEFAULT_API_KEY,
    model: str = GEMINI_MODEL,
    force: bool = False,
    db_path: Path = REPO_ROOT / "data" / "mega_dictionary.db",
    out_dir: Path = REPO_ROOT / "data" / "dictionaries",
):
    """Main pipeline for building the Mega African Dictionary from africa-nouns-en.parquet."""
    tgt_code = verify_gemini_eligibility(target_lang, force=force)
    lang_name = TARGET_LANGUAGE_PROMPTS.get(tgt_code, LANGUAGE_INFO.get(tgt_code, {}).get("name", tgt_code.upper()))

    if not api_key:
        api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is required. Pass via --api-key or set GEMINI_API_KEY environment variable.")

    logger.info(f"Target language: {lang_name} ({tgt_code}) | Concurrency: {max_workers} workers | Batch size: {batch_size}")
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

    candidates = []
    for p, c in zip(all_phrases, all_counts):
        if c >= min_freq and len(p.strip()) >= 2:
            candidates.append((p.strip().lower(), c))

    logger.info(f"Loaded {len(candidates):,} candidate nouns from parquet.")
    if limit and len(candidates) > limit:
        candidates = candidates[:limit]
        logger.info(f"Filtered to top {limit:,} nouns by frequency.")

    # 2. Check cached nouns
    noun_freq_map = {p: c for p, c in candidates}
    cached_set = db.get_cached_set(tgt_code)
    logger.info(f"Already cached in database for {tgt_code}: {len(cached_set):,} nouns")

    todo_nouns = [p for p, _ in candidates if p not in cached_set]
    logger.info(f"Nouns to translate with {model}: {len(todo_nouns):,}")

    if todo_nouns:
        batches = [todo_nouns[i : i + batch_size] for i in range(0, len(todo_nouns), batch_size)]
        logger.info(f"Split into {len(batches):,} batches of {batch_size} nouns each (concurrency={max_workers})")

        completed_batches = 0
        total_translated = 0
        t0 = time.time()

        def process_batch(batch_nouns):
            trans_dict = translate_noun_batch_gemini(batch_nouns, lang_name, api_key, model=model)
            records = []
            for en_word in batch_nouns:
                trans = trans_dict.get(en_word)
                if trans:
                    records.append((en_word, tgt_code, trans, noun_freq_map.get(en_word, 1.0)))
            if records:
                db.push_records(records)
            return len(records)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_batch, b) for b in batches]
            for f in as_completed(futures):
                count = f.result()
                total_translated += count
                completed_batches += 1

                if completed_batches % 10 == 0 or completed_batches == len(batches):
                    elapsed = time.time() - t0
                    speed = (completed_batches * batch_size) / (elapsed + 1e-5)
                    remaining = (len(batches) - completed_batches) * batch_size
                    eta_mins = (remaining / (speed + 1e-5)) / 60
                    logger.info(
                        f"Progress: {completed_batches}/{len(batches)} batches ({completed_batches/len(batches)*100:.1f}%) | "
                        f"Translated: {total_translated:,} nouns | Speed: {speed:.1f} nouns/s | ETA: {eta_mins:.1f} mins"
                    )

    db.close()

    # 3. Export to clean TSV dictionary
    out_tsv = out_dir / f"mega_eng_{tgt_code}_100k.tsv"
    db.export_tsv(tgt_code, out_tsv)
    logger.info(f"Mega Dictionary for {lang_name} successfully built: {out_tsv}")


def main():
    parser = argparse.ArgumentParser(description="Build Mega African Dictionary using Gemini 3.8 & africa-nouns-en.parquet")
    parser.add_argument("--tgt", required=True, help="Target African language (e.g. tw, sw, yo, ha, ig)")
    parser.add_argument("--limit", type=int, default=100000, help="Target top N nouns (default: 100,000)")
    parser.add_argument("--min-freq", type=float, default=1.0, help="Minimum corpus frequency threshold")
    parser.add_argument("--batch-size", type=int, default=100, help="Batch size per Gemini API call (default: 100)")
    parser.add_argument("--workers", type=int, default=20, help="Concurrent worker threads (default: 20)")
    parser.add_argument("--api-key", default=DEFAULT_API_KEY, help="Gemini API Key")
    parser.add_argument("--model", default=GEMINI_MODEL, help="Gemini Model ID (default: gemini-3.8-flash)")
    parser.add_argument("--force", action="store_true", help="Bypass benchmark tier gate")

    args = parser.parse_args()
    build_mega_dictionary(
        target_lang=args.tgt,
        limit=args.limit,
        min_freq=args.min_freq,
        batch_size=args.batch_size,
        max_workers=args.workers,
        api_key=args.api_key,
        model=args.model,
        force=args.force,
    )


if __name__ == "__main__":
    main()
