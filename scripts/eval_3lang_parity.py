"""
3-Language Parity Evaluation (English, Indonesian, Chinese)

Compares tokenization fairness across three languages using the
RotorSubwordTokenizer with rotor-guided merging and grade-aware scoring.
"""

from gatoken import RotorSubwordTokenizer, compute_metrics
from scripts.eval_bias import ENGLISH_TEXTS, INDONESIAN_TEXTS, CHINESE_TEXTS


def main():
    print("Training RotorSubwordTokenizer on combined data...")
    train_texts = ENGLISH_TEXTS + INDONESIAN_TEXTS + CHINESE_TEXTS
    tokenizer = RotorSubwordTokenizer(max_vocab_size=5000)
    tokenizer.train(train_texts)

    print(f"Vocab size: {tokenizer.vocab_size}")
    print(f"Merges learned: {len(tokenizer.merges)}\n")

    en = compute_metrics(tokenizer, ENGLISH_TEXTS, "en")
    id_ = compute_metrics(tokenizer, INDONESIAN_TEXTS, "id")
    zh = compute_metrics(tokenizer, CHINESE_TEXTS, "zh")

    print("=== Fertility & Efficiency ===")
    print(f"English    : fertility={en.fertility:.3f} | tokens/char={en.tokens_per_char:.4f}")
    print(f"Indonesian : fertility={id_.fertility:.3f} | tokens/char={id_.tokens_per_char:.4f}")
    print(f"Chinese    : fertility={zh.fertility:.3f} | tokens/char={zh.tokens_per_char:.4f}")
    print()

    print("=== Pairwise Ratios (lower is better) ===")
    print(f"Indonesian / English : {id_.fertility / en.fertility:.3f}")
    print(f"Chinese    / English : {zh.fertility / en.fertility:.3f}")
    print(f"Chinese    / Indonesian : {zh.fertility / id_.fertility:.3f}")
    print()

    ratios = [
        id_.fertility / en.fertility,
        zh.fertility / en.fertility,
        zh.fertility / id_.fertility
    ]
    avg_dev = sum(abs(r - 1.0) for r in ratios) / len(ratios)
    print(f"Average deviation from perfect parity: {avg_dev:.3f}")


if __name__ == "__main__":
    main()
