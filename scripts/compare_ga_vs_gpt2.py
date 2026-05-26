"""
Compare RotorSubwordTokenizer v2 (with rotor merging) vs GPT-2
"""

from transformers import AutoTokenizer
from gatoken import StandardTokenizer, RotorSubwordTokenizer, compute_metrics, compare_languages

# Training data (same as evaluation for simplicity in prototype)
TRAIN_TEXTS = [
    "The quick brown fox jumps over the lazy dog.",
    "I really like eating spicy food at the night market.",
    "My grandmother makes the best fried rice in the village.",
    "Rubah cokelat yang gesit melompati anjing malas.",
    "Saya sangat suka makan makanan pedas di pasar malam.",
    "Nenek saya membuat nasi goreng terbaik di desa.",
]

ENGLISH = [
    "The quick brown fox jumps over the lazy dog.",
    "I really like eating spicy food at the night market.",
    "My grandmother makes the best fried rice in the village.",
]

INDONESIAN = [
    "Rubah cokelat yang gesit melompati anjing malas.",
    "Saya sangat suka makan makanan pedas di pasar malam.",
    "Nenek saya membuat nasi goreng terbaik di desa.",
]


def evaluate(name, tokenizer):
    en = compute_metrics(tokenizer, ENGLISH, "en")
    id_ = compute_metrics(tokenizer, INDONESIAN, "id")
    comp = compare_languages(en, id_)
    print(f"\n=== {name} ===")
    print(f"English fertility : {en.fertility}")
    print(f"Indonesian fertility: {id_.fertility}")
    print(f"Fertility ratio (id/en): {comp['fertility_ratio (id/en)']}")
    print(f"Parity score: {comp['parity_score']}")


if __name__ == "__main__":
    # GPT-2 baseline
    hf = AutoTokenizer.from_pretrained("gpt2")
    gpt2 = StandardTokenizer(hf)
    evaluate("GPT-2", gpt2)

    # GA Tokenizer with rotor merging
    ga_tok = RotorSubwordTokenizer(max_vocab_size=800)
    ga_tok.train(TRAIN_TEXTS)
    evaluate("RotorSubwordTokenizer v2 (rotor merging)", ga_tok)
    print(f"\nGA Vocab size: {ga_tok.vocab_size}")
    print(f"Number of learned merges: {len(ga_tok.merges)}")
