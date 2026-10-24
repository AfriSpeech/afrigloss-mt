"""
Language & Model Registry for Afri-Gloss-MT.

Connects the 440 Helsinki-NLP African translation models in the AfriSpeech collection:
https://huggingface.co/collections/AfriSpeech/helsinki-nlp-african-translation-models-6adc5ed6acc7c9c71098e79a
with the 693 African languages in AfriSpeech/africa-corpus.
"""

from typing import Dict, List, Optional, Set, Tuple

COLLECTION_SLUG = "AfriSpeech/helsinki-nlp-african-translation-models-6adc5ed6acc7c9c71098e79a"
COLLECTION_URL = f"https://huggingface.co/collections/{COLLECTION_SLUG}"
AFRICA_CORPUS_REPO = "AfriSpeech/africa-corpus"

# ─────────────────────────────────────────────────────────────
# LANGUAGE CODE CANONICAL MAPPINGS
# (ISO 639-3 -> ISO 639-1 / 639-2, Name, Region)
# ─────────────────────────────────────────────────────────────

LANGUAGE_INFO: Dict[str, Dict[str, str]] = {
    # West Africa
    "twi": {"alpha2": "tw", "name": "Twi / Akan", "family": "alv", "region": "West Africa"},
    "aka": {"alpha2": "ak", "name": "Akan", "family": "alv", "region": "West Africa"},
    "yor": {"alpha2": "yo", "name": "Yoruba", "family": "alv", "region": "West Africa"},
    "ibo": {"alpha2": "ig", "name": "Igbo", "family": "alv", "region": "West Africa"},
    "hau": {"alpha2": "ha", "name": "Hausa", "family": "afa", "region": "West Africa"},
    "ewe": {"alpha2": "ee", "name": "Ewe", "family": "alv", "region": "West Africa"},
    "gaa": {"alpha2": "gaa", "name": "Ga", "family": "alv", "region": "West Africa"},
    "efi": {"alpha2": "efi", "name": "Efik", "family": "alv", "region": "West Africa"},
    "guw": {"alpha2": "guw", "name": "Gun", "family": "alv", "region": "West Africa"},
    "iso": {"alpha2": "iso", "name": "Isoko", "family": "alv", "region": "West Africa"},
    "tiv": {"alpha2": "tiv", "name": "Tiv", "family": "alv", "region": "West Africa"},
    "wol": {"alpha2": "wo", "name": "Wolof", "family": "alv", "region": "West Africa"},
    "mos": {"alpha2": "mos", "name": "Mossi (Mooré)", "family": "alv", "region": "West Africa"},
    "dop": {"alpha2": "dop", "name": "Lukpa", "family": "alv", "region": "West Africa"},
    "fon": {"alpha2": "fon", "name": "Fon", "family": "alv", "region": "West Africa"},
    "fuv": {"alpha2": "fuv", "name": "Fulfulde", "family": "alv", "region": "West Africa"},
    "dag": {"alpha2": "dag", "name": "Dagbani", "family": "alv", "region": "West Africa"},

    # East & Horn of Africa
    "swh": {"alpha2": "sw", "name": "Swahili", "family": "bnt", "region": "East Africa"},
    "swc": {"alpha2": "swc", "name": "Congo Swahili", "family": "bnt", "region": "Central/East Africa"},
    "amh": {"alpha2": "am", "name": "Amharic", "family": "sem", "region": "Horn of Africa"},
    "som": {"alpha2": "so", "name": "Somali", "family": "cus", "region": "Horn of Africa"},
    "orm": {"alpha2": "om", "name": "Oromo", "family": "cus", "region": "Horn of Africa"},
    "tir": {"alpha2": "ti", "name": "Tigrinya", "family": "sem", "region": "Horn of Africa"},
    "lug": {"alpha2": "lg", "name": "Luganda (Ganda)", "family": "bnt", "region": "East Africa"},
    "kin": {"alpha2": "rw", "name": "Kinyarwanda", "family": "bnt", "region": "East Africa"},
    "run": {"alpha2": "rn", "name": "Kirundi (Rundi)", "family": "bnt", "region": "East Africa"},
    "luo": {"alpha2": "luo", "name": "Luo (Dholuo)", "family": "nil", "region": "East Africa"},
    "wal": {"alpha2": "wal", "name": "Wolaytta", "family": "omt", "region": "Horn of Africa"},

    # Central Africa
    "lin": {"alpha2": "ln", "name": "Lingala", "family": "bnt", "region": "Central Africa"},
    "kon": {"alpha2": "kg", "name": "Kikongo (Kongo)", "family": "bnt", "region": "Central Africa"},
    "kwy": {"alpha2": "kwy", "name": "San Salvador Kongo", "family": "bnt", "region": "Central Africa"},
    "sag": {"alpha2": "sg", "name": "Sango", "family": "alv", "region": "Central Africa"},
    "lua": {"alpha2": "lua", "name": "Luba-Kasai (Tshiluba)", "family": "bnt", "region": "Central Africa"},
    "lub": {"alpha2": "lu", "name": "Luba-Katanga (Kiluba)", "family": "bnt", "region": "Central Africa"},
    "tll": {"alpha2": "tll", "name": "Tetela", "family": "bnt", "region": "Central Africa"},
    "rnd": {"alpha2": "rnd", "name": "Ruund", "family": "bnt", "region": "Central Africa"},
    "zne": {"alpha2": "zne", "name": "Zande", "family": "alv", "region": "Central Africa"},

    # Southern Africa
    "sna": {"alpha2": "sn", "name": "Shona", "family": "bnt", "region": "Southern Africa"},
    "nya": {"alpha2": "ny", "name": "Chichewa (Nyanja)", "family": "bnt", "region": "Southern Africa"},
    "bem": {"alpha2": "bem", "name": "Bemba", "family": "bnt", "region": "Southern Africa"},
    "xho": {"alpha2": "xh", "name": "Xhosa", "family": "bnt", "region": "Southern Africa"},
    "zul": {"alpha2": "zu", "name": "Zulu", "family": "bnt", "region": "Southern Africa"},
    "nso": {"alpha2": "nso", "name": "Northern Sotho (Sepedi)", "family": "bnt", "region": "Southern Africa"},
    "sot": {"alpha2": "st", "name": "Southern Sotho (Sesotho)", "family": "bnt", "region": "Southern Africa"},
    "tsn": {"alpha2": "tn", "name": "Tswana (Setswana)", "family": "bnt", "region": "Southern Africa"},
    "tso": {"alpha2": "ts", "name": "Tsonga (Xitsonga)", "family": "bnt", "region": "Southern Africa"},
    "ven": {"alpha2": "ve", "name": "Venda (Tshivenda)", "family": "bnt", "region": "Southern Africa"},
    "ssw": {"alpha2": "ss", "name": "Swati (Siswati)", "family": "bnt", "region": "Southern Africa"},
    "loz": {"alpha2": "loz", "name": "Lozi (Silozi)", "family": "bnt", "region": "Southern Africa"},
    "tum": {"alpha2": "tum", "name": "Tumbuka", "family": "bnt", "region": "Southern Africa"},
    "toi": {"alpha2": "toi", "name": "Tonga (Zambia)", "family": "bnt", "region": "Southern Africa"},
    "kqn": {"alpha2": "kqn", "name": "Kaonde", "family": "bnt", "region": "Southern Africa"},
    "lue": {"alpha2": "lue", "name": "Luvale", "family": "bnt", "region": "Southern Africa"},
    "lun": {"alpha2": "lun", "name": "Lunda", "family": "bnt", "region": "Southern Africa"},
    "umb": {"alpha2": "umb", "name": "Umbundu", "family": "bnt", "region": "Southern Africa"},
    "kua": {"alpha2": "kj", "name": "Kuanyama", "family": "bnt", "region": "Southern Africa"},
    "ndo": {"alpha2": "ng", "name": "Ndonga", "family": "bnt", "region": "Southern Africa"},
    "kwn": {"alpha2": "kwn", "name": "Kwangali", "family": "bnt", "region": "Southern Africa"},
    "nyk": {"alpha2": "nyk", "name": "Nyaneka", "family": "bnt", "region": "Southern Africa"},
    "afr": {"alpha2": "af", "name": "Afrikaans", "family": "gmw", "region": "Southern Africa"},

    # North Africa & Islands
    "kab": {"alpha2": "kab", "name": "Kabyle (Berber)", "family": "ber", "region": "North Africa"},
    "cop": {"alpha2": "cop", "name": "Coptic", "family": "egx", "region": "North Africa"},
    "mlg": {"alpha2": "mg", "name": "Malagasy", "family": "poz", "region": "East African Islands"},
    "mfe": {"alpha2": "mfe", "name": "Mauritian Creole", "family": "cpf", "region": "East African Islands"},
    "crs": {"alpha2": "crs", "name": "Seychellois Creole", "family": "cpf", "region": "East African Islands"},
}

# Reverse lookup by alpha2 and common lowercase name
_CODE_LOOKUP: Dict[str, str] = {}
for iso3, info in LANGUAGE_INFO.items():
    _CODE_LOOKUP[iso3] = iso3
    _CODE_LOOKUP[info["alpha2"]] = iso3
    clean_name = info["name"].lower().split()[0].replace("/", "").strip()
    _CODE_LOOKUP[clean_name] = iso3
# Common synonyms
_CODE_LOOKUP.update({
    "twi": "twi",
    "akan": "twi",
    "tw": "twi",
    "ak": "twi",
    "yoruba": "yor",
    "yo": "yor",
    "igbo": "ibo",
    "ig": "ibo",
    "hausa": "hau",
    "ha": "hau",
    "swahili": "swh",
    "sw": "swh",
    "shona": "sna",
    "sn": "sna",
    "zulu": "zul",
    "zu": "zul",
    "xhosa": "xho",
    "xh": "xho",
    "lingala": "lin",
    "ln": "lin",
    "luganda": "lug",
    "lg": "lug",
    "kinyarwanda": "kin",
    "rw": "kin",
    "chichewa": "nya",
    "ny": "nya",
    "amharic": "amh",
    "am": "amh",
    "somali": "som",
    "so": "som",
    "tigrinya": "tir",
    "ti": "tir",
    "ewe": "ewe",
    "ee": "ewe",
    "ga": "gaa",
    "afrikaans": "afr",
    "af": "afr",
    "en": "eng",
    "eng": "eng",
    "english": "eng",
    "fr": "fra",
    "fra": "fra",
    "french": "fra",
    "es": "spa",
    "spa": "spa",
    "spanish": "spa",
    "pt": "por",
    "por": "por",
    "portuguese": "por",
    "de": "deu",
    "deu": "deu",
    "german": "deu",
    "sv": "swe",
    "swe": "swe",
    "swedish": "swe",
    "fi": "fin",
    "fin": "fin",
    "finnish": "fin",
})

# NLLB-200 Language Codes for African Languages
NLLB_LANG_CODES: Dict[str, str] = {
    "twi": "aka_Latn",
    "aka": "aka_Latn",
    "tw": "aka_Latn",
    "ak": "aka_Latn",
    "yor": "yor_Latn",
    "yo": "yor_Latn",
    "swh": "swh_Latn",
    "sw": "swh_Latn",
    "hau": "hau_Latn",
    "ha": "hau_Latn",
    "ibo": "ibo_Latn",
    "ig": "ibo_Latn",
    "ewe": "ewe_Latn",
    "ee": "ewe_Latn",
    "sna": "sna_Latn",
    "sn": "sna_Latn",
    "nya": "nya_Latn",
    "ny": "nya_Latn",
    "lug": "lug_Latn",
    "lg": "lug_Latn",
    "kin": "kin_Latn",
    "rw": "kin_Latn",
    "lin": "lin_Latn",
    "ln": "lin_Latn",
    "run": "run_Latn",
    "rn": "run_Latn",
    "xho": "xho_Latn",
    "xh": "xho_Latn",
    "zul": "zul_Latn",
    "zu": "zul_Latn",
    "som": "som_Latn",
    "so": "som_Latn",
    "amh": "amh_Ethi",
    "am": "amh_Ethi",
    "orm": "gaz_Latn",
    "om": "gaz_Latn",
    "tir": "tir_Ethi",
    "ti": "tir_Ethi",
    "afr": "afr_Latn",
    "af": "afr_Latn",
    "wol": "wol_Latn",
    "wo": "wol_Latn",
    "mos": "mos_Latn",
    "sag": "sag_Latn",
    "sg": "sag_Latn",
    "tsn": "tsn_Latn",
    "tn": "tsn_Latn",
    "tso": "tso_Latn",
    "ts": "tso_Latn",
    "sot": "sot_Latn",
    "st": "sot_Latn",
    "ssw": "ssw_Latn",
    "ss": "ssw_Latn",
    "nso": "nso_Latn",
    "umb": "umb_Latn",
    "bem": "bem_Latn",
    "eng": "eng_Latn",
    "en": "eng_Latn",
    "fra": "fra_Latn",
    "fr": "fra_Latn",
    "spa": "spa_Latn",
    "es": "spa_Latn",
    "por": "por_Latn",
    "pt": "por_Latn",
    "deu": "deu_Latn",
    "de": "deu_Latn",
}


def get_nllb_lang(code: str) -> Optional[str]:
    """Resolve an ISO code to NLLB-200 4-character script code (e.g. aka_Latn, yor_Latn)."""
    clean = code.strip().lower()
    return NLLB_LANG_CODES.get(clean, NLLB_LANG_CODES.get(normalize_lang_code(clean)))


def normalize_lang_code(token: str) -> str:
    """Normalize any language name, 2-letter code, or 3-letter code to canonical 3-letter code."""
    token = token.strip().lower()
    return _CODE_LOOKUP.get(token, token)


def get_alpha2_code(token: str) -> str:
    """Get the 2-letter or short code used by MarianMT for this language."""
    iso3 = normalize_lang_code(token)
    if iso3 in LANGUAGE_INFO:
        return LANGUAGE_INFO[iso3]["alpha2"]
    if iso3 in {"eng", "en"}:
        return "en"
    if iso3 in {"fra", "fr"}:
        return "fr"
    if iso3 in {"spa", "es"}:
        return "es"
    if iso3 in {"por", "pt"}:
        return "pt"
    if iso3 in {"deu", "de"}:
        return "de"
    if iso3 in {"swe", "sv"}:
        return "sv"
    if iso3 in {"fin", "fi"}:
        return "fi"
    return iso3


# ─────────────────────────────────────────────────────────────
# MULTILINGUAL FAMILY TARGET TOKENS
# ─────────────────────────────────────────────────────────────

FAMILY_TARGET_TOKENS: Dict[str, Dict[str, str]] = {
    "alv": {
        "ewe": ">>ewe<<",
        "ibo": ">>ibo<<",
        "kin": ">>kin<<",
        "lin": ">>lin<<",
        "lug": ">>lug<<",
        "nya": ">>nya<<",
        "run": ">>run<<",
        "sag": ">>sag<<",
        "sna": ">>sna<<",
        "toi": ">>toi_Latn<<",
        "tso": ">>tso<<",
        "umb": ">>umb<<",
        "wol": ">>wol<<",
        "xho": ">>xho<<",
        "yor": ">>yor<<",
        "zul": ">>zul<<",
    },
    "bnt": {
        "kin": ">>kin<<",
        "lin": ">>lin<<",
        "lug": ">>lug<<",
        "nya": ">>nya<<",
        "run": ">>run<<",
        "sna": ">>sna<<",
        "toi": ">>toi_Latn<<",
        "tso": ">>tso<<",
        "umb": ">>umb<<",
        "xho": ">>xho<<",
        "zul": ">>zul<<",
    },
    "nic": {
        "bam": ">>bam_Latn<<",
        "ewe": ">>ewe<<",
        "ibo": ">>ibo<<",
        "kin": ">>kin<<",
        "lin": ">>lin<<",
        "lug": ">>lug<<",
        "nya": ">>nya<<",
        "run": ">>run<<",
        "sag": ">>sag<<",
        "sna": ">>sna<<",
        "toi": ">>toi_Latn<<",
        "tso": ">>tso<<",
        "umb": ">>umb<<",
        "wol": ">>wol<<",
        "xho": ">>xho<<",
        "yor": ">>yor<<",
        "zul": ">>zul<<",
    },
    "afa": {
        "amh": ">>amh<<",
        "som": ">>som<<",
        "hau": ">>hau_Latn<<",
        "tir": ">>tir<<",
        "kab": ">>kab<<",
    },
}


def find_model_for_pair(src: str, tgt: str) -> Optional[Dict[str, str]]:
    """
    Find the best pretrained Helsinki-NLP OPUS-MT model for a language pair.

    Returns a dict with:
      - 'model_id': Hugging Face repo ID (e.g. 'Helsinki-NLP/opus-mt-en-tw')
      - 'type': 'direct' or 'multilingual_target' or 'multilingual_source'
      - 'target_token': prefix token needed (e.g. '>>yor<<') or None
    """
    src_3 = normalize_lang_code(src)
    tgt_3 = normalize_lang_code(tgt)
    src_2 = get_alpha2_code(src_3)
    tgt_2 = get_alpha2_code(tgt_3)

    # 1. Check direct pair with alpha2 codes
    candidate_direct = f"Helsinki-NLP/opus-mt-{src_2}-{tgt_2}"

    # Common known direct models
    direct_models = {
        ("en", "tw"): "Helsinki-NLP/opus-mt-en-tw",
        ("es", "tw"): "Helsinki-NLP/opus-mt-es-tw",
        ("fr", "tw"): "Helsinki-NLP/opus-mt-fr-tw",
        ("fi", "tw"): "Helsinki-NLP/opus-mt-fi-tw",
        ("sv", "tw"): "Helsinki-NLP/opus-mt-sv-tw",
        ("tw", "es"): "Helsinki-NLP/opus-mt-tw-es",
        ("tw", "fr"): "Helsinki-NLP/opus-mt-tw-fr",
        ("tw", "fi"): "Helsinki-NLP/opus-mt-tw-fi",
        ("tw", "sv"): "Helsinki-NLP/opus-mt-tw-sv",
        ("en", "yo"): "Helsinki-NLP/opus-mt-en-yo",
        ("yo", "en"): "Helsinki-NLP/opus-mt-yo-en",
        ("en", "ig"): "Helsinki-NLP/opus-mt-en-ig",
        ("ig", "en"): "Helsinki-NLP/opus-mt-ig-en",
        ("en", "ha"): "Helsinki-NLP/opus-mt-en-ha",
        ("ha", "en"): "Helsinki-NLP/opus-mt-ha-en",
        ("en", "sw"): "Helsinki-NLP/opus-mt-en-sw",
        ("sw", "en"): "Helsinki-NLP/opus-mt-sw-en",
        ("en", "ee"): "Helsinki-NLP/opus-mt-en-ee",
        ("ee", "en"): "Helsinki-NLP/opus-mt-ee-en",
        ("en", "gaa"): "Helsinki-NLP/opus-mt-en-gaa",
        ("gaa", "en"): "Helsinki-NLP/opus-mt-gaa-en",
        ("en", "sn"): "Helsinki-NLP/opus-mt-en-sn",
        ("sn", "en"): "Helsinki-NLP/opus-mt-sn-en",
        ("en", "ny"): "Helsinki-NLP/opus-mt-en-ny",
        ("ny", "en"): "Helsinki-NLP/opus-mt-ny-en",
        ("en", "rw"): "Helsinki-NLP/opus-mt-en-rw",
        ("rw", "en"): "Helsinki-NLP/opus-mt-rw-en",
        ("en", "rn"): "Helsinki-NLP/opus-mt-en-rn",
        ("rn", "en"): "Helsinki-NLP/opus-mt-rn-en",
        ("en", "lg"): "Helsinki-NLP/opus-mt-en-lg",
        ("lg", "en"): "Helsinki-NLP/opus-mt-lg-en",
        ("en", "ln"): "Helsinki-NLP/opus-mt-en-ln",
        ("ln", "en"): "Helsinki-NLP/opus-mt-ln-en",
        ("en", "xh"): "Helsinki-NLP/opus-mt-en-xh",
        ("xh", "en"): "Helsinki-NLP/opus-mt-xh-en",
        ("en", "ts"): "Helsinki-NLP/opus-mt-en-ts",
        ("ts", "en"): "Helsinki-NLP/opus-mt-ts-en",
        ("en", "st"): "Helsinki-NLP/opus-mt-en-st",
        ("st", "en"): "Helsinki-NLP/opus-mt-st-en",
        ("en", "nso"): "Helsinki-NLP/opus-mt-en-nso",
        ("nso", "en"): "Helsinki-NLP/opus-mt-nso-en",
        ("en", "tn"): "Helsinki-NLP/opus-mt-en-tn",
        ("tn", "en"): "Helsinki-NLP/opus-mt-tn-en",
        ("en", "ve"): "Helsinki-NLP/opus-mt-en-ve",
        ("ve", "en"): "Helsinki-NLP/opus-mt-ve-en",
        ("en", "ss"): "Helsinki-NLP/opus-mt-en-ss",
        ("ss", "en"): "Helsinki-NLP/opus-mt-ss-en",
        ("en", "sg"): "Helsinki-NLP/opus-mt-en-sg",
        ("sg", "en"): "Helsinki-NLP/opus-mt-sg-en",
        ("en", "wo"): "Helsinki-NLP/opus-mt-en-wo",
        ("wo", "en"): "Helsinki-NLP/opus-mt-wo-en",
        ("en", "mos"): "Helsinki-NLP/opus-mt-en-mos",
        ("mos", "en"): "Helsinki-NLP/opus-mt-mos-en",
        ("en", "lua"): "Helsinki-NLP/opus-mt-en-lua",
        ("lua", "en"): "Helsinki-NLP/opus-mt-lua-en",
        ("en", "lu"): "Helsinki-NLP/opus-mt-en-lu",
        ("lu", "en"): "Helsinki-NLP/opus-mt-lu-en",
        ("en", "kg"): "Helsinki-NLP/opus-mt-en-kg",
        ("kg", "en"): "Helsinki-NLP/opus-mt-kg-en",
        ("en", "bem"): "Helsinki-NLP/opus-mt-en-bem",
        ("bem", "en"): "Helsinki-NLP/opus-mt-bem-en",
        ("en", "tiv"): "Helsinki-NLP/opus-mt-en-tiv",
        ("tiv", "en"): "Helsinki-NLP/opus-mt-tiv-en",
        ("en", "efi"): "Helsinki-NLP/opus-mt-en-efi",
        ("efi", "en"): "Helsinki-NLP/opus-mt-efi-en",
        ("en", "guw"): "Helsinki-NLP/opus-mt-en-guw",
        ("guw", "en"): "Helsinki-NLP/opus-mt-guw-en",
        ("en", "iso"): "Helsinki-NLP/opus-mt-en-iso",
        ("iso", "en"): "Helsinki-NLP/opus-mt-iso-en",
        ("en", "am"): "Helsinki-NLP/opus-mt-en-am",
        ("am", "en"): "Helsinki-NLP/opus-mt-am-en",
        ("en", "so"): "Helsinki-NLP/opus-mt-en-so",
        ("so", "en"): "Helsinki-NLP/opus-mt-so-en",
        ("en", "ti"): "Helsinki-NLP/opus-mt-en-ti",
        ("ti", "en"): "Helsinki-NLP/opus-mt-ti-en",
        ("en", "om"): "Helsinki-NLP/opus-mt-en-om",
        ("om", "en"): "Helsinki-NLP/opus-mt-om-en",
        ("en", "af"): "Helsinki-NLP/opus-mt-en-af",
        ("af", "en"): "Helsinki-NLP/opus-mt-af-en",
        ("en", "crs"): "Helsinki-NLP/opus-mt-en-crs",
        ("crs", "en"): "Helsinki-NLP/opus-mt-crs-en",
        ("en", "mfe"): "Helsinki-NLP/opus-mt-en-mfe",
        ("mfe", "en"): "Helsinki-NLP/opus-mt-mfe-en",
        ("en", "mg"): "Helsinki-NLP/opus-mt-en-mg",
        ("mg", "en"): "Helsinki-NLP/opus-mt-mg-en",
    }

    if (src_2, tgt_2) in direct_models:
        return {
            "model_id": direct_models[(src_2, tgt_2)],
            "type": "direct",
            "target_token": None,
        }

    # 2. Check Multilingual Atlantic-Congo / Bantu / Niger-Congo models
    if src_2 == "en":
        for group in ["alv", "bnt", "nic", "afa"]:
            if tgt_3 in FAMILY_TARGET_TOKENS.get(group, {}):
                return {
                    "model_id": f"Helsinki-NLP/opus-mt-en-{group}",
                    "type": "multilingual_target",
                    "target_token": FAMILY_TARGET_TOKENS[group][tgt_3],
                }

    if tgt_2 == "en":
        for group in ["alv", "bnt", "nic"]:
            if tgt_3 in FAMILY_TARGET_TOKENS.get(group, {}):
                return {
                    "model_id": f"Helsinki-NLP/opus-mt-{group}-en",
                    "type": "multilingual_source",
                    "target_token": None,
                }

    return {
        "model_id": candidate_direct,
        "type": "direct_candidate",
        "target_token": None,
    }
