"""
Evaluate hybrid scoring (frequency + geometric) across different blend weights.

Sweeps freq_weight from 0.0 (pure geometric) to 1.0 (pure BPE)
on the FLORES-101 benchmark.
"""

import os
from gatoken import RotorSubwordTokenizer, StandardTokenizer, compute_metrics
from transformers import AutoTokenizer

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "flores101_dataset", "devtest")

LANGUAGES = {
    "en": "eng",
    "id": "ind",
    "zh": "zho_simpl",
    "ms": "msa",
    "vi": "vie",
    "th": "tha",
    "ko": "kor",
    "ja": "jpn",
    "ar": "ara",
    "hi": "hin",
}

BASELINES = [
    ("GPT-2", "gpt2"),
    ("XLM-R", "xlm-roberta-base"),
]

WEIGHTS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def load_flores_lang(lang_code):
    fname = f"{lang_code}.devtest"
    fpath = os.path.join(DATA_DIR, fname)
    if not os.path.exists(fpath):
        return []
    with open(fpath, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def main():
    # Load data
    corpora = {}
    for lang, code in LANGUAGES.items():
        texts = load_flores_lang(code)
        if texts:
            corpora[lang] = texts
            print(f"  {lang}: {len(texts)} sentences")

    train_texts = []
    for lang, texts in corpora.items():
        train_texts.extend(texts[:100])

    eval_langs = sorted(corpora.keys())

    # Sweep freq_weight
    print(f"\n{'weight':>8}  ", end="")
    for lang in eval_langs:
        print(f"{lang:>7}", end="")
    print(f"  {'Parity':>8}  {'EN_fert':>8}")

    print("-" * (10 + 7 * len(eval_langs) + 20))

    results = {}
    for w in WEIGHTS:
        tok = RotorSubwordTokenizer(max_vocab_size=500, freq_weight=w)
        tok.train(train_texts)

        ferts = {}
        for lang in eval_langs:
            eval_texts = corpora[lang][100:200]
            m = compute_metrics(tok, eval_texts, lang)
            ferts[lang] = m.fertility

        en_fert = ferts.get("en", 1.0)
        parity = min(ferts.values()) / max(ferts.values())

        row = f"{w:>8.1f}  "
        for lang in eval_langs:
            ratio = ferts[lang] / en_fert if en_fert > 0 else 0
            row += f"{ratio:>7.2f}"
        row += f"  {parity:>8.3f}  {en_fert:>8.2f}"
        print(row)

        results[w] = {
            "ferts": ferts,
            "parity": parity,
            "en_fert": en_fert,
            "vocab": tok.vocab_size,
        }

    # Find best weight for parity
    best_w = max(results.keys(), key=lambda w: results[w]["parity"])
    print(f"\nBest freq_weight for parity: {best_w:.1f} (parity={results[best_w]['parity']:.3f}, en_fert={results[best_w]['en_fert']:.2f})")

    # Find best weight for compression (lowest en_fert with parity > 0.05)
    valid = {w: r for w, r in results.items() if r["parity"] > 0.05}
    best_comp = min(valid.keys(), key=lambda w: valid[w]["en_fert"])
    print(f"Best freq_weight for compression (parity > 0.05): {best_comp:.1f} (en_fert={valid[best_comp]['en_fert']:.2f}, parity={valid[best_comp]['parity']:.3f})")

    # Baselines
    print("\n--- Baselines ---")
    for name, model_id in BASELINES:
        try:
            hf_tok = AutoTokenizer.from_pretrained(model_id)
            tok = StandardTokenizer(hf_tok)
            ferts = {}
            for lang in eval_langs:
                m = compute_metrics(tok, corpora[lang][100:200], lang)
                ferts[lang] = m.fertility
            en_fert = ferts.get("en", 1.0)
            parity = min(ferts.values()) / max(ferts.values())
            print(f"  {name}: en_fert={en_fert:.2f}, parity={parity:.3f}")
        except Exception as e:
            print(f"  {name}: Failed - {e}")


if __name__ == "__main__":
    main()