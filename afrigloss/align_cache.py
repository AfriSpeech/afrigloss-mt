"""
Persistent Alignment Cache for Afri-Gloss-MT.

Caches word/phrase alignments for parallel sentence pairs in JSONL format.
When re-running dataset generation or scaling from 10k to 50k or 100k samples,
already-aligned sentence pairs are loaded instantly from the cache,
so only new pairs are aligned.
"""

import hashlib
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def sentence_pair_hash(src: str, tgt: str) -> str:
    """Compute a fast deterministic MD5 hash for a parallel sentence pair."""
    normalized = f"{src.strip()}|||{tgt.strip()}"
    return hashlib.md5(normalized.encode("utf-8")).hexdigest()


class AlignmentCache:
    """
    On-disk resumable JSONL cache for sentence-level word alignments.
    """

    def __init__(self, cache_file: str | Path):
        self.cache_path = Path(cache_file)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._cache: Dict[str, List[Dict[str, any]]] = {}
        self._file_handle = None
        self._load()

    def _load(self):
        """Load existing cached alignments into memory."""
        if not self.cache_path.exists():
            return

        with open(self.cache_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        record = json.loads(line)
                        k = record.get("key") or sentence_pair_hash(record["src"], record["tgt"])
                        self._cache[k] = record.get("alignments", [])
                    except Exception:
                        continue

    def __len__(self) -> int:
        return len(self._cache)

    def contains(self, src: str, tgt: str) -> bool:
        """Check if an alignment for this sentence pair is already cached."""
        k = sentence_pair_hash(src, tgt)
        return k in self._cache

    def get(self, src: str, tgt: str) -> Optional[List[Dict[str, any]]]:
        """Retrieve cached alignments for (src, tgt) if present."""
        k = sentence_pair_hash(src, tgt)
        return self._cache.get(k)

    def set(self, src: str, tgt: str, alignments: List[Dict[str, any]]):
        """Store an alignment in memory and append to on-disk JSONL cache."""
        k = sentence_pair_hash(src, tgt)
        self._cache[k] = alignments

        # Open in append mode
        if self._file_handle is None:
            self._file_handle = open(self.cache_path, "a", encoding="utf-8")

        record = {
            "key": k,
            "src": src,
            "tgt": tgt,
            "alignments": alignments,
        }
        self._file_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._file_handle.flush()

    def close(self):
        """Close cache file handle if open."""
        if self._file_handle is not None:
            self._file_handle.close()
            self._file_handle = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
