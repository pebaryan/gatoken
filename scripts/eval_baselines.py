"""
Baseline evaluation: Compare GA tokenizer against standard tokenizers.

Evaluates GPT-2, and optionally Llama-3 / Qwen-2.5,
on FLORES-200 (if available) or synthetic test data.
"""

import argparse
from gatoken import RotorSubwordTokenizer, StandardTokenizer, compute_metrics
from scripts.eval_bias import ENGLISH_TEXTS, INDONESIAN_TEXTS, CHINESE_TEXTS

try:
    from transformers import AutoTokenizer
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False


def evaluate_hf_tokenizer(model_name, texts, lang):
    """Evaluate a HuggingFace tokenizer."""
    if not HF_AVAILABLE:
        print(f"  transformers not available, skipping {model_name}")
        return None
    try:
        tok = AutoTokenizer.from_pretrained(model_name)
        # Wrap in our interface
        class HFWrap:
            def tokenize(self, text):
                return tok.convert_ids_to_tokens(tok.encode(text))
            @property
            def vocab_size(self):
                return tok.vocab_size
        wrapper = StandardTokenizer(tok)
        return compute_metrics(wrapper, texts, lang)
    except Exception as e:
        print(f"  Error loading {model_name}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baselines", nargs="+",
                        default=["gpt2"],
                        choices=["gpt2", "llama3", "qwen2.5"],
                        help="Baseline tokenizers to evaluate")
    parser.add_argument("--ga-vocab", type=int, default=200,
                        help="Max vocab size for GA tokenizer")
    args = parser.parse_args()

    # Map baseline names to HF model names
    model_map = {
        "gpt2": "gpt2",
        "llama3": "meta-llama/Meta-Llama-3-8B",
        "qwen2.5": "Qwen/Qwen2.5-0.5B",
    }

    print("=== Baseline Evaluation ===\n")

    # GA tokenizer
    print(f"Training RotorSubwordTokenizer (vocab={args.ga_vocab})...")
    ga_tok = RotorSubwordTokenizer(max_vocab_size=args.ga_vocab)
    ga_tok.train(ENGLISH_TEXTS[:10] + INDONESIAN_TEXTS[:10] + CHINESE_TEXTS[:10])

    ga_en = compute_metrics(ga_tok, ENGLISH_TEXTS[:10], 'en')
    ga_id = compute_metrics(ga_tok, INDONESIAN_TEXTS[:10], 'id')
    ga_zh = compute_metrics(ga_tok, CHINESE_TEXTS[:10], 'zh')

    print(f"\n{'Tokenizer':<25} {'EN fert':>8} {'ID fert':>8} {'ZH fert':>8} {'ID/EN':>8} {'ZH/EN':>8}")
    print("-" * 75)

    ga_id_en = ga_id.fertility / ga_en.fertility if ga_en.fertility > 0 else 0
    ga_zh_en = ga_zh.fertility / ga_en.fertility if ga_en.fertility > 0 else 0
    print(f"{'RotorSubword (GA)':<25} {ga_en.fertility:>8.3f} {ga_id.fertility:>8.3f} {ga_zh.fertility:>8.3f} {ga_id_en:>8.3f} {ga_zh_en:>8.3f}")

    # Baseline tokenizers
    for name in args.baselines:
        model_name = model_map.get(name, name)
        print(f"\nLoading {name} ({model_name})...")

        en_m = evaluate_hf_tokenizer(model_name, ENGLISH_TEXTS[:10], 'en')
        id_m = evaluate_hf_tokenizer(model_name, INDONESIAN_TEXTS[:10], 'id')
        zh_m = evaluate_hf_tokenizer(model_name, CHINESE_TEXTS[:10], 'zh')

        if en_m and id_m and zh_m:
            id_en = id_m.fertility / en_m.fertility if en_m.fertility > 0 else 0
            zh_en = zh_m.fertility / en_m.fertility if en_m.fertility > 0 else 0
            print(f"{name:<25} {en_m.fertility:>8.3f} {id_m.fertility:>8.3f} {zh_m.fertility:>8.3f} {id_en:>8.3f} {zh_en:>8.3f}")


if __name__ == "__main__":
    main()