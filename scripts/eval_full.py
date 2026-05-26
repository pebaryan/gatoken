"""
Full evaluation: GA tokenizer vs baselines on multilingual corpus.

Compares RotorSubwordTokenizer (iterative BPE) against GPT-2 and other
HuggingFace tokenizers across 5 languages.
"""

import argparse
from gatoken import RotorSubwordTokenizer, StandardTokenizer, compute_metrics
from scripts.corpus import get_all_corpora

try:
    from transformers import AutoTokenizer
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False


def evaluate_ga(max_vocab=200):
    """Train and evaluate RotorSubwordTokenizer."""
    corpora = get_all_corpora()
    all_texts = []
    for lang, texts in corpora.items():
        all_texts.extend(texts)

    print(f"Training RotorSubwordTokenizer (max_vocab={max_vocab})...")
    print(f"  Training on {len(all_texts)} sentences from {len(corpora)} languages")
    
    tok = RotorSubwordTokenizer(max_vocab_size=max_vocab)
    tok.train(all_texts)
    print(f"  Vocab: {tok.vocab_size}, Merges: {len(tok.merges)}")

    results = {}
    for lang, texts in corpora.items():
        metrics = compute_metrics(tok, texts, lang)
        results[lang] = metrics

    return results


def evaluate_hf(model_name, corpora):
    """Evaluate a HuggingFace tokenizer."""
    if not HF_AVAILABLE:
        return None
    try:
        hf_tok = AutoTokenizer.from_pretrained(model_name)
        tok = StandardTokenizer(hf_tok)
        results = {}
        for lang, texts in corpora.items():
            metrics = compute_metrics(tok, texts, lang)
            results[lang] = metrics
        return results
    except Exception as e:
        print(f"  Error: {e}")
        return None


def print_results(name, results, en_fertility=None):
    """Print evaluation results in a formatted table."""
    if not results:
        return
    
    en_fert = en_fertility or results.get("en", None)
    en_fert_val = en_fert.fertility if en_fert else results["en"].fertility

    print(f"\n{'='*70}")
    print(f"  {name}")
    print(f"{'='*70}")
    print(f"  {'Lang':<6} {'Fertility':>10} {'Tok/Char':>10} {'Ratio/EN':>10}")
    print(f"  {'-'*40}")
    
    for lang in ["en", "id", "zh", "ms", "vi"]:
        if lang in results:
            m = results[lang]
            ratio = m.fertility / en_fert_val if en_fert_val > 0 else 0
            print(f"  {lang:<6} {m.fertility:>10.3f} {m.tokens_per_char:>10.4f} {ratio:>10.3f}")

    # Compute parity
    ferts = [results[lang].fertility for lang in results]
    if ferts:
        parity = min(ferts) / max(ferts)
        print(f"\n  Parity score: {parity:.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ga-vocab", type=int, default=200,
                        help="Max vocab size for GA tokenizer")
    parser.add_argument("--skip-hf", action="store_true",
                        help="Skip HuggingFace baselines")
    args = parser.parse_args()

    corpora = get_all_corpora()
    print(f"Corpus sizes: {', '.join(f'{lang}={len(texts)}' for lang, texts in corpora.items())}")

    # GA tokenizer
    ga_results = evaluate_ga(args.ga_vocab)
    print_results("RotorSubwordTokenizer (GA)", ga_results)

    # Baselines
    if not args.skip_hf:
        print("\n" + "="*70)
        print("  Baseline Tokenizers")
        print("="*70)
        
        baselines = {
            "GPT-2": "gpt2",
            "Llama-3": "meta-llama/Meta-Llama-3-8B",
            "Qwen-2.5": "Qwen/Qwen2.5-0.5B",
        }
        
        for name, model_name in baselines.items():
            print(f"\nLoading {name} ({model_name})...")
            results = evaluate_hf(model_name, corpora)
            if results:
                print_results(name, results)
            else:
                print(f"  Could not load {name}")


if __name__ == "__main__":
    main()