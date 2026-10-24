"""
Afri-Gloss-MT: African Glossary-Constrained Machine Translation Toolkit.

Enables zero-shot and fine-tuned glossary control for 693 African languages,
leveraging 440 Helsinki-NLP OPUS-MT translation models and AfriSpeech/africa-corpus.
"""

from .aligner import StatisticalWordAligner, WordAlignment
from .corpus import AfricaCorpusLoader, ParallelPair
from .dataset import GlossMTExample, build_gloss_pair, create_gloss_dataset, save_dataset_splits
from .dictionary import AfricanGlossary, GlossaryDictionary, build_dictionary_from_aligner
from .inference import AfriGlossTranslator
from .registry import (
    COLLECTION_SLUG,
    COLLECTION_URL,
    LANGUAGE_INFO,
    find_model_for_pair,
    normalize_lang_code,
)
from .trainer import GlossMTTrainer, TrainConfig, clean_output

__version__ = "0.1.0"

__all__ = [
    "AfricaCorpusLoader",
    "ParallelPair",
    "StatisticalWordAligner",
    "WordAlignment",
    "GlossaryDictionary",
    "AfricanGlossary",
    "build_dictionary_from_aligner",
    "GlossMTExample",
    "build_gloss_pair",
    "create_gloss_dataset",
    "save_dataset_splits",
    "GlossMTTrainer",
    "TrainConfig",
    "clean_output",
    "AfriGlossTranslator",
    "find_model_for_pair",
    "normalize_lang_code",
    "COLLECTION_SLUG",
    "COLLECTION_URL",
    "LANGUAGE_INFO",
]
