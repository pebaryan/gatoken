"""
Comprehensive benchmark: Fertility vs Characters-Per-Token metrics.

Compares RotorSubword (50k vocab) against GPT-2 using both
whitespace-fertility and characters-per-token (script-fair).
"""

import os
import json
from gatoken import RotorSubwordTokenizer, StandardTokenizer, compute_metrics
from transformers import AutoTokenizer

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "flores101_dataset", "devtest")

LANGUAGES = {
    "en": "eng", "id": "ind", "zh": "zho_simpl", "ms": "msa", "vi": "vie",
    "th": "tha", "tl": "tgl", "jv": "jav", "ko": "kor", "ja": "jpn",
    "ar": "ara", "hi": "hin",
}


def load_flores():
    corpora = {}
    for lang, code in LANGUAGES.items():
        fpath = os.path.join(DATA_DIR, f"{code}.devtest")
        if os.path.exists(fpath):
            with open(fpath, "r", encoding="utf-8") as f:
                corpora[lang] = [l.strip() for l in f if l.strip()]
    return corpora


def evaluate_ga(corpora, vocab_size=50000, batch_size=100):
    import time
    train_texts = []
    for lang, texts in corpora.items():
        train_texts.extend(texts[:50])

    print(f"Training RotorSubword (vocab={vocab_size}, batch={batch_size})...")
    t0 = time.time()
    tok = RotorSubwordTokenizer(max_vocab_size=vocab_size, batch_size=batch_size)
    tok.train(train_texts)
    elapsed = time.time() - t0
    print(f"  Vocab: {tok.vocab_size}, Merges: {len(tok.merges)}, Time: {elapsed:.0f}s")

    results = {}
    for lang in sorted(corpora):
        m = compute_metrics(tok, corpora[lang][200:300], lang)
        results[lang] = m
    return results


def evaluate_hf(model_name, corpora):
    try:
        hf_tok = AutoTokenizer.from_pretrained(model_name)
        tok = StandardTokenizer(hf_tok)
    except Exception as e:
        print(f"  Could not load {model_name}: {e}")
        return None

    results = {}
    for lang in sorted(corpora):
        m = compute_metrics(tok, corpora[lang][200:300], lang)
        results[lang] = m
    return results


def print_table(name, results, metric="fertility"):
    """Print a comparison table. metric can be 'fertility' or 'chars_per_token'."""
    en_val = results.get("en")
    en_fert = en_val.fertility if en_val else 1.0
    en_cpt = en_val.chars_per_token if en_val else 1.0

    print(f"\n{'='*80}")
    print(f"  {name} — {metric}")
    print(f"{'='*80}")

    if metric == "fertility":
        print(f"  {'Lang':<6} {'Fert':>8} {'R/EN':>8}")
        print(f"  {'-'*24}")
        ferts = []
        for lang in sorted(results):
            m = results[lang]
            ferts.append(m.fertility)
            ratio = m.fertility / en_fert if en_fert > 0 else 0
            print(f"  {lang:<6} {m.fertility:>8.2f} {ratio:>8.2f}")
        parity = min(ferts) / max(ferts) if ferts else 0
        print(f"\n  Parity: {parity:.4f}")

    elif metric == "chars_per_token":
        print(f"  {'Lang':<6} {'CPT':>8} {'R/EN':>8}")
        print(f"  {'-'*24}")
        cpts = []
        for lang in sorted(results):
            m = results[lang]
            cpts.append(m.chars_per_token)
            # For CPT: higher is better (more characters per token = more efficient)
            # Ratio: other/en — closer to 1.0 is more fair
            ratio = m.chars_per_token / en_cpt if en_cpt > 0 else 0
            print(f"  {lang:<6} {m.chars_per_token:>8.3f} {ratio:>8.3f}")
        # For CPT parity: we want the ratio close to 1.0 for all languages
        # "Fair" CPT parity: languages should have similar chars-per-token
        # Lower CPT means the language is penalized (fewer chars per token)
        ratios = [results[lang].chars_per_token / en_cpt for lang in sorted(results)]
        # Parity for CPT: min_ratio / max_ratio (how close to 1.0 are the ratios)
        cpt_parity = min(ratios) / max(ratios) if ratios else 0
        print(f"\n  CPT Parity (min_ratio/max_ratio): {cpt_parity:.4f}")
        # Also: are ratios closer to 1.0 than fertility ratios?
        fert_ratios = [results[lang].fertility / en_fert for lang in sorted(results)]
        fert_parity = min(fert_ratios) / max(fert_ratios) if fert_ratios else 0
        print(f"  Fertility Parity: {fert_parity:.4f}")


def main():
    corpora = load_flores()
    print(f"Loaded {len(corpora)} languages")
    for lang, texts in sorted(corpora.items()):
        print(f"  {lang}: {len(texts)} sentences")

    # GA tokenizer (50k vocab)
    ga_results = evaluate_ga(corpora)

    # GPT-2 baseline
    print("\nEvaluating GPT-2...")
    gpt2_results = evaluate_hf("gpt2", corpora)

    # Print all tables
    print("\n" + "=" * 80)
    print("COMPREHENSIVE BENCHMARK: FERTILITY vs CHARACTERS-PER-TOKEN")
    print("=" * 80)

    print_table("RotorSubword (50k)", ga_results, metric="fertility")
    print_table("RotorSubword (50k)", ga_results, metric="chars_per_token")

    if gpt2_results:
        print_table("GPT-2 (50k)", gpt2_results, metric="fertility")
        print_table("GPT-2 (50k)", gpt2_results, metric="chars_per_token")

    # Side-by-side comparison
    if gpt2_results:
        print("\n" + "=" * 80)
        print("  SIDE-BY-SIDE: RotorSubword vs GPT-2")
        print("=" * 80)
        print(f"  {'Lang':<6} {'GA_fert':>8} {'GA_R':>6} {'GPT_fert':>8} {'GPT_R':>6} | {'GA_cpt':>7} {'GA_cR':>6} {'GPT_cpt':>7} {'GPT_cR':>6}")
        print(f"  {'-'*80}")

        en_ga = ga_results.get("en")
        en_gpt = gpt2_results.get("en")

        for lang in sorted(set(ga_results) & set(gpt2_results)):
            ga_m = ga_results[lang]
            gpt_m = gpt2_results[lang]

            ga_fert_r = ga_m.fertility / en_ga.fertility if en_ga.fertility > 0 else 0
            gpt_fert_r = gpt_m.fertility / en_gpt.fertility if en_gpt.fertility > 0 else 0
            ga_cpt_r = ga_m.chars_per_token / en_ga.chars_per_token if en_ga.chars_per_token > 0 else 0
            gpt_cpt_r = gpt_m.chars_per_token / en_gpt.chars_per_token if en_gpt.chars_per_token > 0 else 0

            print(f"  {lang:<6} {ga_m.fertility:>8.2f} {ga_fert_r:>6.2f} {gpt_m.fertility:>8.2f} {gpt_fert_r:>6.2f} | {ga_m.chars_per_token:>7.3f} {ga_cpt_r:>6.3f} {gpt_m.chars_per_token:>7.3f} {gpt_cpt_r:>6.3f}")

        # Summary
        ga_ferts = [ga_results[l].fertility for l in sorted(ga_results)]
        gpt_ferts = [gpt2_results[l].fertility for l in sorted(gpt2_results)]
        ga_fert_parity = min(ga_ferts) / max(ga_ferts)
        gpt_fert_parity = min(gpt_ferts) / max(gpt_ferts)

        ga_cpts = [ga_results[l].chars_per_token for l in sorted(ga_results)]
        gpt_cpts = [gpt2_results[l].chars_per_token for l in sorted(gpt2_results)]

        en_ga_cpt = ga_results["en"].chars_per_token
        en_gpt_cpt = gpt2_results["en"].chars_per_token
        ga_cpt_ratios = [ga_results[l].chars_per_token / en_ga_cpt for l in sorted(ga_results)]
        gpt_cpt_ratios = [gpt2_results[l].chars_per_token / en_gpt_cpt for l in sorted(gpt2_results)]
        ga_cpt_parity = min(ga_cpt_ratios) / max(ga_cpt_ratios)
        gpt_cpt_parity = min(gpt_cpt_ratios) / max(gpt_cpt_ratios)

        print(f"\n  Fertility Parity:  GA={ga_fert_parity:.4f}  GPT-2={gpt_fert_parity:.4f}  (higher=better)")
        print(f"  CPT Parity:        GA={ga_cpt_parity:.4f}  GPT-2={gpt_cpt_parity:.4f}  (higher=better)")

    # Save results
    all_results = {}
    for name, res in [("RotorSubword_50k", ga_results), ("GPT-2", gpt2_results)]:
        if res:
            all_results[name] = {
                lang: {
                    "fertility": m.fertility,
                    "chars_per_token": m.chars_per_token,
                    "tokens_per_char": m.tokens_per_char,
                    "total_tokens": m.total_tokens,
                    "total_chars": m.total_chars,
                }
                for lang, m in res.items()
            }

    with open(os.path.join(os.path.dirname(__file__), "..", "results_50k_cpt.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to results_50k_cpt.json")


if __name__ == "__main__":
    main()