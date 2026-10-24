"""
Inference Engine and Constrained Translation Runtime for Afri-Gloss-MT.

Translates input sentences while strictly honoring glossary and dictionary terms:
  1. Identifies nominal phrases in source text using spaCy noun chunks (or user overrides).
  2. Replaces noun phrases IN PLACE with target translations in '<...>' brackets.
  3. Executes MarianMT generation.
  4. Post-processes output to extract clean translation while preserving terminology constraints.
"""

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch

from afrigloss.dictionary import GlossaryDictionary
from afrigloss.extractor import extract_noun_phrases
from afrigloss.registry import find_model_for_pair
from afrigloss.trainer import clean_output


class AfriGlossTranslator:
    """
    Gloss-MT Translation Pipeline with terminology injection and constraints.
    """

    def __init__(
        self,
        model_name_or_path: str,
        dictionary: Optional[Union[GlossaryDictionary, str, Path]] = None,
        target_prefix: Optional[str] = None,
        device: Optional[str] = None,
    ):
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[AfriGloss] Loading model '{model_name_or_path}' on {self.device}...")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path).to(self.device)
        self.model.eval()

        self.target_prefix = target_prefix

        # Load dictionary if provided
        if isinstance(dictionary, (str, Path)):
            self.dictionary = GlossaryDictionary.load_tsv(dictionary)
        elif isinstance(dictionary, GlossaryDictionary):
            self.dictionary = dictionary
        else:
            self.dictionary = GlossaryDictionary()

    def inject_glossary(
        self,
        text: str,
        custom_glossary: Optional[Dict[str, str]] = None,
    ) -> Tuple[str, List[Dict[str, str]]]:
        """
        Identify noun phrases in the source text and replace them in place as '<target_term>'.

        Returns:
            (pretranslated_text, list_of_applied_terms)
        """
        active_dict = self.dictionary
        if custom_glossary:
            active_dict = GlossaryDictionary(dict(self.dictionary.entries))
            for k, v in custom_glossary.items():
                active_dict.add(k, v, confidence=1.0)

        # 1. Extract noun phrases from source text
        noun_chunks = extract_noun_phrases(text)
        insertions: List[Tuple[int, int, str, str]] = []

        for start, end, phrase in noun_chunks:
            p_clean = phrase.lower().strip()
            tgt_trans = active_dict.lookup(p_clean)
            if tgt_trans:
                insertions.append((start, end, phrase, tgt_trans))

        # 2. Also check if custom glossary has direct matches not captured by noun chunks
        if custom_glossary:
            for s_term, t_term in custom_glossary.items():
                for m in re.finditer(rf"\b{re.escape(s_term)}\b", text, re.IGNORECASE):
                    start, end = m.start(), m.end()
                    if not any(max(start, is_s) < min(end, is_e) for is_s, is_e, _, _ in insertions):
                        insertions.append((start, end, m.group(0), t_term))

        if not insertions:
            return text, []

        # Sort matches in reverse order so in-place replacements do not shift string indices
        insertions.sort(key=lambda m: m[0], reverse=True)

        pretranslated = text
        applied_terms = []

        for start, end, src_phrase, tgt_trans in insertions:
            # Match casing
            if src_phrase.isupper():
                trans_repr = tgt_trans.upper()
            elif src_phrase[0].isupper():
                trans_repr = tgt_trans[0].upper() + tgt_trans[1:]
            else:
                trans_repr = tgt_trans.lower()

            pretranslated = pretranslated[:start] + f"<{trans_repr}>" + pretranslated[end:]
            applied_terms.append({"src": src_phrase, "tgt": tgt_trans})

        applied_terms.reverse()
        return pretranslated, applied_terms

    def translate(
        self,
        text: str,
        custom_glossary: Optional[Dict[str, str]] = None,
        max_length: int = 256,
        num_beams: int = 4,
        return_brackets: bool = False,
    ) -> Dict[str, Union[str, List[Dict[str, str]], bool]]:
        """
        Translate a single sentence with terminology constraint.

        Returns:
            Dict containing:
              - 'translation': final clean translated text
              - 'pretranslated_input': source text with pre-inserted target terms
              - 'applied_terms': terms injected from the dictionary
              - 'constraints_honored': whether all injected terms appear in the output
        """
        pretranslated, applied_terms = self.inject_glossary(text, custom_glossary)

        # Prepend multilingual prefix if required (e.g. >>yor<< for en-alv)
        prompt = pretranslated
        if self.target_prefix:
            prompt = f"{self.target_prefix} {prompt}"

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length).to(self.device)

        with torch.no_grad():
            generated_tokens = self.model.generate(
                **inputs,
                max_length=max_length,
                num_beams=num_beams,
                early_stopping=True,
            )

        raw_output = self.tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
        final_translation = clean_output(raw_output)

        # Verify terminology retention
        honored = True
        for term in applied_terms:
            tgt_term = term["tgt"]
            if not re.search(rf"\b{re.escape(tgt_term)}\b", final_translation, re.IGNORECASE):
                honored = False
                break

        return {
            "translation": raw_output if return_brackets else final_translation,
            "raw_translation": raw_output,
            "pretranslated_input": pretranslated,
            "applied_terms": applied_terms,
            "constraints_honored": honored,
        }

    def translate_batch(
        self,
        texts: List[str],
        custom_glossary: Optional[Dict[str, str]] = None,
        batch_size: int = 8,
        max_length: int = 256,
    ) -> List[Dict]:
        """Translate a batch of sentences."""
        results = []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i : i + batch_size]
            for t in chunk:
                results.append(self.translate(t, custom_glossary=custom_glossary, max_length=max_length))
        return results
