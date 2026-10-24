"""
Command-Line Interface for Afri-Gloss-MT.

Provides subcommands:
  - list-models: Inspect models in the 440-model AfriSpeech Helsinki-NLP collection
  - align:       Run word alignment with confidence scoring on parallel text
  - build-dict:  Extract high-confidence bilingual dictionary from parallel corpus
  - build-data:  Build Gloss-MT pre-translated dataset splits for training
  - train:       Fine-tune OPUS-MT MarianMT model on Gloss-MT dataset
  - translate:   Execute glossary-constrained translation (one-shot or interactive)
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

from afrigloss.aligner import StatisticalWordAligner, tokenize_words
from afrigloss.corpus import AfricaCorpusLoader
from afrigloss.dataset import create_gloss_dataset, save_dataset_splits
from afrigloss.dictionary import GlossaryDictionary, build_dictionary_from_aligner
from afrigloss.registry import (
    COLLECTION_URL,
    LANGUAGE_INFO,
    find_model_for_pair,
    normalize_lang_code,
)


def cmd_list_models(args):
    """List available models in the AfriSpeech collection."""
    print("=" * 70)
    print("AfriSpeech Helsinki-NLP African Translation Models")
    print(f"Collection: {COLLECTION_URL}")
    print("=" * 70)

    target_lang = args.lang.strip().lower() if args.lang else None

    count = 0
    for code, info in sorted(LANGUAGE_INFO.items()):
        if target_lang and target_lang not in (code, info["alpha2"], info["name"].lower()):
            continue

        model_info_en_to = find_model_for_pair("en", code)
        model_info_to_en = find_model_for_pair(code, "en")

        print(f"\n[{info['name']}] (ISO: {code} / {info['alpha2']}) - {info['region']}")
        if model_info_en_to:
            prefix = f" (prefix: '{model_info_en_to['target_token']}')" if model_info_en_to.get("target_token") else ""
            print(f"  En -> {info['alpha2']}: {model_info_en_to['model_id']}{prefix}")
        if model_info_to_en:
            print(f"  {info['alpha2']} -> En: {model_info_to_en['model_id']}")
        count += 1

    print(f"\nListed {count} languages.")


def cmd_build_dict(args):
    """Build a high-confidence bilingual dictionary from parallel text."""
    src = normalize_lang_code(args.src)
    tgt = normalize_lang_code(args.tgt)

    print(f"[AfriGloss] Loading parallel corpus for {src} <-> {tgt} from Africa Corpus...")
    loader = AfricaCorpusLoader()
    pairs = loader.get_parallel_pairs(src, tgt, max_samples=args.max_samples)
    print(f"[AfriGloss] Loaded {len(pairs):,} parallel sentence pairs.")

    if not pairs:
        print("[AfriGloss] No parallel pairs found for this pair.")
        sys.exit(1)

    print(f"[AfriGloss] Training statistical word aligner (min_pair_freq={args.min_freq})...")
    bitext = [(p.src, p.tgt) for p in pairs]
    aligner = StatisticalWordAligner(min_pair_freq=args.min_freq, min_confidence=args.min_confidence)
    aligner.train(bitext)

    print(f"[AfriGloss] Extracting lexicon with confidence >= {args.min_confidence:.2f}...")
    lexicon = aligner.extract_bilingual_lexicon(min_confidence=args.min_confidence)
    dictionary = build_dictionary_from_aligner(lexicon, min_confidence=args.min_confidence)

    out_file = Path(args.out)
    dictionary.save_tsv(out_file)
    print(f"[AfriGloss] Successfully extracted {len(dictionary):,} high-confidence dictionary entries!")
    print(f"[AfriGloss] Saved to: {out_file}")


def cmd_merge_dicts(args):
    """Merge two bilingual dictionaries into a unified dictionary."""
    from afrigloss.dictionary import GlossaryDictionary, merge_dictionaries

    print(f"[AfriGloss] Loading base dictionary: {args.base}...")
    d_a = GlossaryDictionary.load_tsv(args.base)
    print(f"[AfriGloss] Base dictionary entries: {len(d_a):,}")

    print(f"[AfriGloss] Loading new dictionary: {args.new}...")
    d_b = GlossaryDictionary.load_tsv(args.new)
    print(f"[AfriGloss] New dictionary entries: {len(d_b):,}")

    merged = merge_dictionaries(d_a, d_b)
    out_file = Path(args.out)
    merged.save_tsv(out_file)
    print(f"[AfriGloss] Successfully merged {len(merged):,} entries -> {out_file}")


def cmd_build_data(args):
    """Build a Gloss-MT pre-inserted dataset for fine-tuning."""
    src = normalize_lang_code(args.src)
    tgt = normalize_lang_code(args.tgt)

    dictionary = None
    if args.dictionary and Path(args.dictionary).exists():
        print(f"[AfriGloss] Loading dictionary from {args.dictionary}...")
        dictionary = GlossaryDictionary.load_tsv(args.dictionary)
        print(f"[AfriGloss] Dictionary entries: {len(dictionary):,}")

    print(f"[AfriGloss] Loading parallel corpus for {src} <-> {tgt} from Africa Corpus...")
    loader = AfricaCorpusLoader()
    pairs = loader.get_parallel_pairs(src, tgt, max_samples=None) # Load full corpus to train aligner
    print(f"[AfriGloss] Loaded {len(pairs):,} parallel verses.")

    # Train statistical aligner across the full corpus
    print(f"[AfriGloss] Training aligner across full {len(pairs):,} verses for dense lexical coverage...")
    bitext = [(p.src, p.tgt) for p in pairs]
    aligner = StatisticalWordAligner(min_pair_freq=2, min_confidence=args.min_confidence)
    aligner.train(bitext)

    target_count = args.target_samples or 10_000
    print(f"[AfriGloss] Extracting {target_count:,} examples with guaranteed high-confidence noun-chunk insertions...")

    splits = create_gloss_dataset(
        bitext,
        dictionary=dictionary,
        aligner=aligner,
        target_count=target_count,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        min_confidence=args.min_confidence,
    )

    out_dir = Path(args.out_dir)
    save_dataset_splits(splits, out_dir)
    print(f"[AfriGloss] Dataset splits saved to {out_dir}:")
    for name, s in splits.items():
        print(f"  - {name}: {len(s):,} samples (100% with verified terminology constraints)")


def cmd_build_custom(args):
    """Build Gloss-MT dataset and dictionary from user-supplied parallel file."""
    from afrigloss.corpus import load_custom_parallel_file

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[AfriGloss Custom] Loading user parallel file: {args.parallel_file}...")
    pairs = load_custom_parallel_file(args.parallel_file)
    print(f"[AfriGloss Custom] Loaded {len(pairs):,} parallel sentence pairs.")

    if not pairs:
        print("[AfriGloss Custom] Error: No valid parallel pairs found.")
        sys.exit(1)

    if len(pairs) < 5000:
        print(f"\n[Note on Dataset Scale] You provided {len(pairs):,} sentence pairs.")
        print("  - Even a small dataset (1,000 - 5,000 sentences) will teach the model the Gloss-MT in-place constraint copying rule!")
        print("  - For maximum vocabulary coverage and fluency improvements, 10k to 50k+ sentences are recommended.")
        print("  - Proceeding with training set generation with what is available...\n")

    print("[AfriGloss Custom] Aligning parallel sentences to extract dictionary...")
    bitext = [(p.src, p.tgt) for p in pairs]
    aligner = StatisticalWordAligner(min_pair_freq=2, min_confidence=args.min_confidence)
    aligner.train(bitext)

    lexicon = aligner.extract_bilingual_lexicon(min_confidence=args.min_confidence)
    dictionary = build_dictionary_from_aligner(lexicon, min_confidence=args.min_confidence)
    dict_file = out_dir / f"dictionary_{args.src}_{args.tgt}.tsv"
    dictionary.save_tsv(dict_file)
    print(f"[AfriGloss Custom] Extracted {len(dictionary):,} high-confidence dictionary entries -> {dict_file}")

    target_count = min(len(pairs), args.target_samples or 100_000)
    print(f"[AfriGloss Custom] Building pre-inserted Gloss-MT splits (target: {target_count:,})...")
    splits = create_gloss_dataset(
        bitext,
        dictionary=dictionary,
        aligner=aligner,
        target_count=target_count,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        min_confidence=args.min_confidence,
    )

    save_dataset_splits(splits, out_dir)
    print(f"[AfriGloss Custom] Done! Splits saved to {out_dir}:")
    for name, s in splits.items():
        print(f"  - {name}: {len(s):,} samples (100% with verified terminology constraints)")


def cmd_train(args):
    """Fine-tune OPUS-MT on a Gloss-MT dataset."""
    from afrigloss.trainer import GlossMTTrainer, TrainConfig

    src = normalize_lang_code(args.src)
    tgt = normalize_lang_code(args.tgt)

    if not args.model:
        model_info = find_model_for_pair(src, tgt)
        if not model_info:
            print(f"[AfriGloss] No default model found for {src} -> {tgt}. Please specify --model.")
            sys.exit(1)
        model_name = model_info["model_id"]
    else:
        model_name = args.model

    cfg = TrainConfig(
        model_name_or_path=model_name,
        output_dir=args.out,
        train_file=args.train_file,
        val_file=args.val_file,
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        max_steps=args.max_steps,
        bf16=not args.no_bf16,
    )

    trainer = GlossMTTrainer(cfg)
    trainer.train()


def cmd_translate(args):
    """Run constrained translation using dictionary."""
    from afrigloss.inference import AfriGlossTranslator

    src = normalize_lang_code(args.src)
    tgt = normalize_lang_code(args.tgt)

    model_info = find_model_for_pair(src, tgt)
    model_name = args.model or (model_info["model_id"] if model_info else None)
    target_prefix = model_info.get("target_token") if model_info else None

    if not model_name:
        print(f"[AfriGloss] Could not resolve model for {src} -> {tgt}. Pass --model explicitly.")
        sys.exit(1)

    translator = AfriGlossTranslator(
        model_name_or_path=model_name,
        dictionary=args.dictionary,
        target_prefix=target_prefix,
    )

    # Parse manual glossary overrides if passed: e.g. "covenant:apam,God:Onyankopɔn"
    custom_gloss: Dict[str, str] = {}
    if args.gloss:
        for pair in args.gloss.split(","):
            if ":" in pair:
                k, v = pair.split(":", 1)
                custom_gloss[k.strip()] = v.strip()

    if args.interactive:
        print("\n" + "=" * 50)
        print(f"Afri-Gloss-MT Interactive Mode: {src.upper()} -> {tgt.upper()}")
        print("Type text to translate, or ':gloss word:translation' to add a term. Ctrl+C to exit.")
        print("=" * 50 + "\n")
        try:
            while True:
                line = input("src> ").strip()
                if not line:
                    continue
                if line.startswith(":gloss "):
                    rule = line[len(":gloss "):].strip()
                    if ":" in rule:
                        k, v = rule.split(":", 1)
                        custom_gloss[k.strip()] = v.strip()
                        print(f"Added glossary override: '{k.strip()}' -> '{v.strip()}'")
                    continue

                res = translator.translate(line, custom_glossary=custom_gloss)
                print(f"tgt> {res['translation']}")
                if res["applied_terms"]:
                    print(f"     [Injected: {res['applied_terms']} | Preserved: {res['constraints_honored']}]")
        except (KeyboardInterrupt, EOFError):
            print("\nExiting interactive mode.")
            return

    if not args.text:
        print("Please provide --text '...' to translate, or use --interactive.")
        return

    res = translator.translate(args.text, custom_glossary=custom_gloss)
    print("\n--- Translation Result ---")
    print(f"Input:       {args.text}")
    print(f"Pre-injected:{res['pretranslated_input']}")
    print(f"Translation: {res['translation']}")
    if res["applied_terms"]:
        print(f"Gloss Terms: {res['applied_terms']}")
        print(f"Preserved:   {res['constraints_honored']}")


def main():
    parser = argparse.ArgumentParser(
        prog="afrigloss",
        description="Afri-Gloss-MT: African Machine Translation with Pre-Inserted Terminology Constraints",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # list-models
    p_list = subparsers.add_parser("list-models", help="List African models in AfriSpeech collection")
    p_list.add_argument("--lang", help="Filter by language code or name (e.g. twi, yor, sw)")
    p_list.set_defaults(func=cmd_list_models)

    # build-dict
    p_dict = subparsers.add_parser("build-dict", help="Extract high-confidence dictionary from parallel text")
    p_dict.add_argument("--src", required=True, help="Source language (e.g. en)")
    p_dict.add_argument("--tgt", required=True, help="Target language (e.g. tw, yo, sw)")
    p_dict.add_argument("--out", default="data/dictionary.tsv", help="Output TSV path")
    p_dict.add_argument("--min-confidence", type=float, default=0.70, help="Min alignment confidence (0.0 to 1.0)")
    p_dict.add_argument("--min-freq", type=int, default=3, help="Min co-occurrence frequency")
    p_dict.add_argument("--max-samples", type=int, default=None, help="Limit number of verses processed")
    p_dict.set_defaults(func=cmd_build_dict)

    # merge-dicts
    p_merge = subparsers.add_parser("merge-dicts", help="Merge two bilingual dictionaries into a unified dictionary")
    p_merge.add_argument("--base", required=True, help="Path to base TSV dictionary")
    p_merge.add_argument("--new", required=True, help="Path to new TSV dictionary to merge in")
    p_merge.add_argument("--out", required=True, help="Output merged TSV path")
    p_merge.set_defaults(func=cmd_merge_dicts)

    # build-data
    p_data = subparsers.add_parser("build-data", help="Build pre-inserted Gloss-MT training dataset")
    p_data.add_argument("--src", required=True, help="Source language (e.g. en)")
    p_data.add_argument("--tgt", required=True, help="Target language (e.g. tw, yo, sw)")
    p_data.add_argument("--dictionary", help="Path to high-confidence dictionary TSV (optional)")
    p_data.add_argument("--out-dir", default="data/gloss_dataset", help="Output directory for splits")
    p_data.add_argument("--target-samples", type=int, default=10000, help="Target dataset samples with guaranteed insertions (default: 10000)")
    p_data.add_argument("--val-ratio", type=float, default=0.05, help="Validation split ratio")
    p_data.add_argument("--test-ratio", type=float, default=0.05, help="Test split ratio")
    p_data.add_argument("--min-confidence", type=float, default=0.50, help="Min term confidence threshold")
    p_data.set_defaults(func=cmd_build_data)

    # build-custom
    p_custom = subparsers.add_parser("build-custom", help="Build Gloss-MT dataset & dictionary from user's own parallel file")
    p_custom.add_argument("--parallel-file", required=True, help="Path to custom parallel file (.tsv, .csv, .jsonl, .parquet)")
    p_custom.add_argument("--src", default="en", help="Source language code (default: en)")
    p_custom.add_argument("--tgt", required=True, help="Target language code (e.g. tw, yo, sw)")
    p_custom.add_argument("--out-dir", required=True, help="Output directory for generated dataset splits & dictionary")
    p_custom.add_argument("--target-samples", type=int, default=100000, help="Max samples to include (default: 100,000)")
    p_custom.add_argument("--val-ratio", type=float, default=0.05, help="Validation split ratio")
    p_custom.add_argument("--test-ratio", type=float, default=0.05, help="Test split ratio")
    p_custom.add_argument("--min-confidence", type=float, default=0.50, help="Min alignment confidence threshold")
    p_custom.set_defaults(func=cmd_build_custom)

    # train
    p_train = subparsers.add_parser("train", help="Fine-tune OPUS-MT on Gloss-MT dataset")
    p_train.add_argument("--src", required=True, help="Source language code")
    p_train.add_argument("--tgt", required=True, help="Target language code")
    p_train.add_argument("--model", help="Hugging Face model ID or checkpoint directory (optional)")
    p_train.add_argument("--train-file", required=True, help="Path to train.jsonl")
    p_train.add_argument("--val-file", help="Path to validation.jsonl")
    p_train.add_argument("--out", default="checkpoints/afrigloss_model", help="Output checkpoint directory")
    p_train.add_argument("--batch-size", type=int, default=8, help="Batch size")
    p_train.add_argument("--lr", type=float, default=3e-5, help="Learning rate")
    p_train.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    p_train.add_argument("--max-steps", type=int, default=None, help="Max steps for pilot run")
    p_train.add_argument("--no-bf16", action="store_true", help="Disable bfloat16")
    p_train.set_defaults(func=cmd_train)

    # translate
    p_trans = subparsers.add_parser("translate", help="Translate with dictionary terminology constraints")
    p_trans.add_argument("--src", required=True, help="Source language code")
    p_trans.add_argument("--tgt", required=True, help="Target language code")
    p_trans.add_argument("--model", help="Model ID or checkpoint path")
    p_trans.add_argument("--dictionary", help="Path to dictionary TSV file")
    p_trans.add_argument("--text", help="Text to translate")
    p_trans.add_argument("--gloss", help="Manual terminology overrides (e.g. 'word:trans,word2:trans2')")
    p_trans.add_argument("--interactive", action="store_true", help="Launch interactive REPL")
    p_trans.set_defaults(func=cmd_translate)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
