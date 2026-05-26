"""
Comprehensive benchmark using FLORES-101 dataset.

1012 sentences per language, 12+ languages.
Compares RotorSubword against modern multilingual tokenizers.
"""

import os
from gatoken import RotorSubwordTokenizer, StandardTokenizer, compute_metrics
from transformers import AutoTokenizer

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "flores101_dataset", "devtest")

# FLORES-101 language mapping
LANGUAGES = {
    "en": ("eng", "English"),
    "id": ("ind", "Indonesian"),
    "zh": ("zho_simpl", "Chinese (Simplified)"),
    "ms": ("msa", "Malay"),
    "vi": ("vie", "Vietnamese"),
    "th": ("tha", "Thai"),
    "tl": ("tgl", "Tagalog"),
    "jv": ("jav", "Javanese"),
    "ko": ("kor", "Korean"),
    "ja": ("jpn", "Japanese"),
    "ar": ("ara", "Arabic"),
    "hi": ("hin", "Hindi"),
}

BASELINES = [
    ("GPT-2", "gpt2"),
    ("Qwen-2.5", "Qwen/Qwen2.5-1.5B"),
    ("XLM-R", "xlm-roberta-base"),
    ("mGPT", "ai-forever/mGPT"),
    ("BLOOM", "bigscience/bloom-560m"),
    ("Mistral", "mistralai/Mistral-7B-v0.1"),
    ("BERT-multilingual", "bert-base-multilingual-cased"),
]


def load_flores_lang(lang_code):
    """Load FLORES-101 data for a language."""
    fname = f"{lang_code}.devtest"
    fpath = os.path.join(DATA_DIR, fname)
    if not os.path.exists(fpath):
        return []
    with open(fpath, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def load_all_flores():
    """Load all FLORES-101 languages."""
    corpora = {}
    for lang, (code, name) in LANGUAGES.items():
        texts = load_flores_lang(code)
        if texts:
            corpora[lang] = texts
            print(f"  {lang} ({name}): {len(texts)} sentences")
    return corpora


def train_ga(corpora, max_vocab=500):
    """Train RotorSubword on a subset of the data."""
    # Use first 100 sentences per language for training
    train_texts = []
    for lang, texts in corpora.items():
        train_texts.extend(texts[:100])

    tok = RotorSubwordTokenizer(max_vocab_size=max_vocab)
    tok.train(train_texts)
    return tok


def evaluate_hf(model_name, corpora, eval_langs):
    """Evaluate a HuggingFace tokenizer."""
    try:
        hf_tok = AutoTokenizer.from_pretrained(model_name)
        tok = StandardTokenizer(hf_tok)
    except Exception as e:
        print(f"  Could not load {model_name}: {e}")
        return None

    results = {}
    for lang in eval_langs:
        if lang in corpora and corpora[lang]:
            # Use all sentences for evaluation
            metrics = compute_metrics(tok, corpora[lang], lang)
            results[lang] = metrics
    return results


def main():
    print("Loading FLORES-101 data...")
    corpora = load_all_flores()

    if not corpora:
        print("ERROR: No FLORES data found. Run download_flores.py first.")
        return

    # Train Ga tokenizer
    print("\nTraining RotorSubword on FLORES-101 (100 sentences per language)...")
    ga_tok = train_ga(corpora, max_vocab=500)
    print(f"  Vocab: {ga_tok.vocab_size}, Merges: {len(ga_tok.merges)}")

    eval_langs = sorted(corpora.keys())

    # Evaluate GA tokenizer
    print("\nEvaluating RotorSubword...")
    ga_results = {}
    en_fert_ga = None
    for lang in eval_langs:
        # Use sentences 100-200 for evaluation (unseen)
        eval_texts = corpora[lang][100:200]
        if eval_texts:
            m = compute_metrics(ga_tok, eval_texts, lang)
            ga_results[lang] = m
            if lang == "en":
                en_fert_ga = m.fertility

    # Evaluate baselines
    print("\nEvaluating baselines...")
    all_results = {"RotorSubword": ga_results}
    for name, model_id in BASELINES:
        print(f"  {name}...")
        results = evaluate_hf(model_id, corpora, eval_langs)
        if results:
            all_results[name] = results

    # Print comparison table
    print("\n" + "=" * 100)
    print("FLORES-101 CROSS-LINGUISTIC TOKENIZATION BENCHMARK")
    print(f"  Training: 100 sentences per language, vocab=500")
    print(f"  Evaluation: 100 sentences per language (unseen)")
    print("=" * 100)

    # Header
    header = f"{'Tokenizer':<20}"
    for lang in eval_langs:
        header += f" {lang:>5}"
    header += f" {'Parity':>8}"
    print(header)
    print("-" * len(header))

    for tok_name, lang_results in all_results.items():
        row = f"{tok_name:<20}"
        en_fert = lang_results.get("en").fertility if "en" in lang_results else 1.0
        ferts = []
        for lang in eval_langs:
            if lang in lang_results:
                m = lang_results[lang]
                ratio = m.fertility / en_fert if en_fert > 0 else 0
                row += f" {ratio:>5.2f}"
                ferts.append(m.fertility)
            else:
                row += f" {'N/A':>5}"

        parity = min(ferts) / max(ferts) if ferts else 0
        row += f" {parity:>8.3f}"
        print(row)

    # Print raw fertility table
    print("\n" + "=" * 100)
    print("RAW FERTILITY (tokens per word)")
    print("=" * 100)

    header = f"{'Tokenizer':<20}"
    for lang in eval_langs:
        header += f" {lang:>8}"
    print(header)
    print("-" * len(header))

    for tok_name, lang_results in all_results.items():
        row = f"{tok_name:<20}"
        for lang in eval_langs:
            if lang in lang_results:
                m = lang_results[lang]
                row += f" {m.fertility:>8.2f}"
            else:
                row += f" {'N/A':>8}"
        print(row)

    # Print parity for each pair vs English
    print("\n" + "=" * 100)
    print("RATIO VS ENGLISH (lower = more fair, 1.0 = perfect parity)")
    print("=" * 100)

    header = f"{'Tokenizer':<20}"
    for lang in eval_langs:
        if lang != "en":
            header += f" {lang+'/EN':>7}"
    header += f" {'Parity':>8}"
    print(header)
    print("-" * len(header))

    for tok_name, lang_results in all_results.items():
        en_fert = lang_results.get("en").fertility if "en" in lang_results else 1.0
        row = f"{tok_name:<20}"
        ferts = []
        for lang in eval_langs:
            if lang in lang_results:
                ferts.append(lang_results[lang].fertility)
                if lang != "en":
                    ratio = lang_results[lang].fertility / en_fert if en_fert > 0 else 0
                    row += f" {ratio:>7.2f}"

        parity = min(ferts) / max(ferts) if ferts else 0
        row += f" {parity:>8.3f}"
        print(row)


if __name__ == "__main__":
    main()