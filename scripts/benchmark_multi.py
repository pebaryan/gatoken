"""
Comprehensive benchmark: GA tokenizer vs modern multilingual tokenizers.

Compares RotorSubwordTokenizer against:
- GPT-2 (50k vocab, English-centric)
- Qwen-2.5 (151k vocab, multilingual)
- XLM-R (250k vocab, multilingual)
- mGPT (100k vocab, multilingual)
- BLOOM (250k vocab, multilingual)
- Mistral (32k vocab)
- BERT-multilingual (119k vocab)
- BigCode (49k vocab, code-focused)
"""

from gatoken import RotorSubwordTokenizer, StandardTokenizer, compute_metrics
from scripts.corpus import get_all_corpora
from transformers import AutoTokenizer

# Tokenizers to benchmark (name, model_id)
BASELINES = [
    ("GPT-2", "gpt2"),
    ("Qwen-2.5", "Qwen/Qwen2.5-1.5B"),
    ("XLM-R", "xlm-roberta-base"),
    ("mGPT", "ai-forever/mGPT"),
    ("BLOOM", "bigscience/bloom-560m"),
    ("Mistral", "mistralai/Mistral-7B-v0.1"),
    ("BERT-multilingual", "bert-base-multilingual-cased"),
    ("BigCode", "bigcode/starcoder2-3b"),
]

LANGUAGES = ["en", "id", "zh", "ms", "vi"]


def evaluate_tokenizer(name, model_id, corpora):
    """Evaluate a HuggingFace tokenizer across all languages."""
    try:
        hf_tok = AutoTokenizer.from_pretrained(model_id)
        tok = StandardTokenizer(hf_tok)
    except Exception as e:
        print(f"  Could not load {name}: {e}")
        return None

    results = {}
    for lang in LANGUAGES:
        if lang in corpora and corpora[lang]:
            metrics = compute_metrics(tok, corpora[lang], lang)
            results[lang] = metrics
    
    return results


def main():
    corpora = get_all_corpora()
    print(f"Corpus sizes: {', '.join(f'{lang}={len(texts)}' for lang, texts in corpora.items())}")

    # GA tokenizer
    print("\nTraining RotorSubwordTokenizer...")
    all_texts = []
    for lang, texts in corpora.items():
        all_texts.extend(texts[:10])

    ga_tok = RotorSubwordTokenizer(max_vocab_size=200)
    ga_tok.train(all_texts)
    print(f"  Vocab: {ga_tok.vocab_size}, Merges: {len(ga_tok.merges)}")

    results = {}

    # GA results
    ga_results = {}
    for lang in LANGUAGES:
        if lang in corpora and corpora[lang]:
            ga_results[lang] = compute_metrics(ga_tok, corpora[lang], lang)
    results["RotorSubword"] = (None, ga_results)

    # Baseline results
    for name, model_id in BASELINES:
        print(f"\nEvaluating {name} ({model_id})...")
        baseline_results = evaluate_tokenizer(name, model_id, corpora)
        if baseline_results:
            results[name] = (model_id, baseline_results)

    # Print comparison table
    print("\n" + "=" * 90)
    print("CROSS-LINGUISTIC TOKENIZATION BENCHMARK")
    print("=" * 90)

    # Header
    header = f"{'Tokenizer':<20} {'Vocab':>8}"
    for lang in LANGUAGES:
        header += f" {lang+' fert':>9} {lang+' R/EN':>8}"
    header += f" {'Parity':>8}"
    print(header)
    print("-" * len(header))

    for name, (model_id, lang_results) in results.items():
        row = f"{name:<20}"
        if model_id:
            try:
                hf_tok = AutoTokenizer.from_pretrained(model_id)
                row += f" {hf_tok.vocab_size:>8}"
            except:
                row += f" {'?':>8}"
        else:
            row += f" {ga_tok.vocab_size:>8}"

        en_fert = lang_results.get("en", None)
        en_fert_val = en_fert.fertility if en_fert else 1.0
        
        ferts = []
        for lang in LANGUAGES:
            if lang in lang_results:
                m = lang_results[lang]
                ferts.append(m.fertility)
                ratio = m.fertility / en_fert_val if en_fert_val > 0 else 0
                row += f" {m.fertility:>9.2f} {ratio:>8.2f}"
            else:
                row += f" {'N/A':>9} {'N/A':>8}"

        parity = min(ferts) / max(ferts) if ferts else 0
        row += f" {parity:>8.3f}"
        print(row)

    # Summary: fairness comparison
    print("\n" + "=" * 90)
    print("FAIRNESS SUMMARY (Ratio vs English, lower = more fair)")
    print("=" * 90)
    print(f"{'Tokenizer':<20}", end="")
    for lang in LANGUAGES:
        if lang != "en":
            print(f" {lang+'/EN':>8}", end="")
    print(f" {'Parity':>8}")
    print("-" * 60)

    for name, (model_id, lang_results) in results.items():
        row = f"{name:<20}"
        en_fert = lang_results.get("en", None)
        en_fert_val = en_fert.fertility if en_fert else 1.0
        
        ferts = []
        for lang in LANGUAGES:
            if lang in lang_results:
                ferts.append(lang_results[lang].fertility)
                if lang != "en":
                    ratio = lang_results[lang].fertility / en_fert_val if en_fert_val > 0 else 0
                    row += f" {ratio:>8.2f}"

        parity = min(ferts) / max(ferts) if ferts else 0
        row += f" {parity:>8.3f}"
        print(row)


if __name__ == "__main__":
    main()