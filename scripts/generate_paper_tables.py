"""
Generate LaTeX table fragments from the rerun benchmark JSONs.
Used to patch paper/PAPER.tex with bug-fix-era numbers.
"""

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
# Paper artifacts live outside this repo; override with $GATOKEN_PAPER_DIR.
PAPER = os.environ.get(
    "GATOKEN_PAPER_DIR",
    os.path.join(ROOT, "paper-out"),
)

LANGS = ["ar", "en", "hi", "id", "ja", "jv", "ko", "ms", "th", "tl", "vi", "zh"]
NON_EN = [l for l in LANGS if l != "en"]


def load(name):
    with open(os.path.join(PAPER, name), encoding="utf-8") as f:
        return json.load(f)


def parity(values):
    return min(values) / max(values) if values else 0.0


def ratios_vs_en(results, key="fertility"):
    en = results["en"][key]
    return {lang: results[lang][key] / en for lang in LANGS}


def fmt(x, fmt_str="{:.2f}"):
    return fmt_str.format(x) if x is not None else "--"


def table1_ratio_vs_english():
    """Table 1: ratio vs English fertility across 8 tokenizers, small-vocab eval slice."""
    rotor500 = load("results_500.json")["results"]
    rotor3k = load("results_3k.json")["results"]
    baselines = load("results_baselines_small.json")

    rows = []
    rows.append(("RotorSubword (500, char-level)", rotor500))
    rows.append(("RotorSubword (3k merges)", rotor3k))
    for name in ["GPT-2", "Qwen-2.5", "XLM-R", "mGPT", "BLOOM", "Mistral", "BERT-multilingual"]:
        if name in baselines and "error" not in baselines[name]:
            rows.append((name, baselines[name]["results"]))

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        r"\caption{Ratio vs English fertility (small-vocab eval, FLORES-101 devtest sentences 100--199, 100 sentences/lang). RotorSubword rows are produced by the bug-fixed implementation.}",
        r"\label{tab:ratio}",
        r"\begin{tabular}{lrrrrrrrrrrr}",
        r"\toprule",
        "Tokenizer & " + " & ".join(NON_EN) + r" & Parity \\",
        r"\midrule",
    ]
    for name, results in rows:
        r = ratios_vs_en(results, "fertility")
        ferts = [results[l]["fertility"] for l in LANGS]
        p = parity(ferts)
        row = [name]
        row += [fmt(r[l]) for l in NON_EN]
        row.append(fmt(p, "{:.3f}"))
        lines.append(" & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def table2_raw_fertility():
    """Table 2: raw fertility for RotorSubword + three representative baselines."""
    rotor500 = load("results_500.json")["results"]
    rotor3k = load("results_3k.json")["results"]
    baselines = load("results_baselines_small.json")

    rows = []
    rows.append(("RotorSubword (500, char-level)", rotor500))
    rows.append(("RotorSubword (3k merges)", rotor3k))
    for name in ["GPT-2", "Qwen-2.5", "XLM-R"]:
        if name in baselines:
            rows.append((name, baselines[name]["results"]))

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        r"\caption{Raw fertility (tokens per whitespace-delimited word) for RotorSubword and representative baselines, small-vocab eval.}",
        r"\label{tab:fertility}",
        r"\begin{tabular}{lrrrrrrrrrrrr}",
        r"\toprule",
        "Tokenizer & " + " & ".join(LANGS) + r" \\",
        r"\midrule",
    ]
    for name, results in rows:
        row = [name] + [fmt(results[l]["fertility"]) for l in LANGS]
        lines.append(" & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def table3_matched():
    """Table 3: matched-vocab fertility, RotorSubword vs GPT-2."""
    rotor = load("results_50k_rerun.json")
    rotor_results = rotor["results"]
    baselines = load("results_baselines_matched.json")
    gpt2 = baselines["GPT-2"]["results"]

    r_en = rotor_results["en"]["fertility"]
    g_en = gpt2["en"]["fertility"]

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Matched-vocabulary comparison: RotorSubword (vocab~" +
        f"{rotor['vocab']}, {rotor['merges']} merges, 50 sentences/lang training)" +
        r" vs GPT-2 (vocab 50,257). Eval on FLORES-101 devtest sentences 200--299.}",
        r"\label{tab:matched}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Language & RotorSubword & R/EN & GPT-2 & R/EN \\",
        r"\midrule",
    ]
    LANG_NAMES = {"ar":"Arabic","en":"English","hi":"Hindi","id":"Indonesian",
                  "ja":"Japanese","jv":"Javanese","ko":"Korean","ms":"Malay",
                  "th":"Thai","tl":"Tagalog","vi":"Vietnamese","zh":"Chinese"}
    for lang in LANGS:
        rf = rotor_results[lang]["fertility"]
        gf = gpt2[lang]["fertility"]
        lines.append(f"{LANG_NAMES[lang]} & {rf:.2f} & {rf/r_en:.2f} & {gf:.2f} & {gf/g_en:.2f} " + r"\\")

    r_pa = parity([rotor_results[l]["fertility"] for l in LANGS])
    g_pa = parity([gpt2[l]["fertility"] for l in LANGS])
    lines.append(r"\textbf{Parity} & & \textbf{" + f"{r_pa:.3f}" + r"} & & " + f"{g_pa:.3f}" + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def table4_cpt():
    """Table 4: matched-vocab characters-per-token."""
    rotor = load("results_50k_rerun.json")
    rotor_results = rotor["results"]
    baselines = load("results_baselines_matched.json")
    gpt2 = baselines["GPT-2"]["results"]

    r_en_cpt = rotor_results["en"]["chars_per_token"]
    g_en_cpt = gpt2["en"]["chars_per_token"]

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Characters-per-token and CPT ratio vs English at matched vocabulary. Higher CPT is more efficient; closer-to-1 ratios are fairer.}",
        r"\label{tab:cpt}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Language & Rotor CPT & R/EN & GPT-2 CPT & GPT-2 R/EN \\",
        r"\midrule",
    ]
    LANG_NAMES = {"ar":"Arabic","en":"English","hi":"Hindi","id":"Indonesian",
                  "ja":"Japanese","jv":"Javanese","ko":"Korean","ms":"Malay",
                  "th":"Thai","tl":"Tagalog","vi":"Vietnamese","zh":"Chinese"}
    for lang in LANGS:
        rc = rotor_results[lang]["chars_per_token"]
        gc = gpt2[lang]["chars_per_token"]
        lines.append(f"{LANG_NAMES[lang]} & {rc:.3f} & {rc/r_en_cpt:.3f} & {gc:.3f} & {gc/g_en_cpt:.3f} " + r"\\")

    # CPT parity using ratio-vs-EN range
    r_ratios = [rotor_results[l]["chars_per_token"]/r_en_cpt for l in LANGS]
    g_ratios = [gpt2[l]["chars_per_token"]/g_en_cpt for l in LANGS]
    r_pa = parity(r_ratios)
    g_pa = parity(g_ratios)
    lines.append(r"\textbf{CPT Parity} & & \textbf{" + f"{r_pa:.3f}" + r"} & & " + f"{g_pa:.3f}" + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def table_sqrt_bpe_ablation():
    """Apples-to-apples ablation: RotorSubword vs sqrt-n BPE trained on same
    FLORES data, same merge loop, same eval slice. Isolates the GA prior."""
    ga_3k = load("results_3k.json")
    sb_3k = load("results_sqrt_bpe_3k.json")
    ga_m = load("results_50k_rerun.json")
    sb_m = load("results_sqrt_bpe_matched.json")

    def row(name, payload):
        r = payload["results"]
        fs = [r[l]["fertility"] for l in LANGS]
        cs = [r[l]["chars_per_token"] for l in LANGS]
        en_cpt = r["en"]["chars_per_token"]
        ratios = [c / en_cpt for c in cs]
        fp = parity(fs)
        cp = parity(ratios)
        en_fert = r["en"]["fertility"]
        return (f"{name} & {payload['vocab']:,} & {payload['merges']:,} & "
                f"{en_fert:.2f} & {fp:.3f} & {cp:.3f} \\\\")

    lines = [
        r"\begin{table*}[htbp]",
        r"\centering",
        r"\caption{Apples-to-apples ablation isolating the rotor-alignment prior. RotorSubword and $\sqrt{n}$-BPE are trained on the same FLORES-101 sentences, with the same merge loop, batching rule, and evaluation split (devtest sentences 100--199 for the 3k row; 200--299 for the matched-vocab row). The only difference is whether the bigram score multiplies $\sqrt{n}$ by the rotor alignment term or by 1.}",
        r"\label{tab:sqrt-bpe-ablation}",
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        r"Setting & Tokenizer & Vocab & Merges & en fert & Fert parity & CPT parity \\",
        r"\midrule",
        f"\\multirow{{2}}{{*}}{{3k vocab, 100 sent/lang}} & RotorSubword & {ga_3k['vocab']:,} & {ga_3k['merges']:,} & {ga_3k['results']['en']['fertility']:.2f} & {parity([ga_3k['results'][l]['fertility'] for l in LANGS]):.3f} & {parity([ga_3k['results'][l]['chars_per_token']/ga_3k['results']['en']['chars_per_token'] for l in LANGS]):.3f} \\\\",
        f"& $\\sqrt{{n}}$-BPE & {sb_3k['vocab']:,} & {sb_3k['merges']:,} & {sb_3k['results']['en']['fertility']:.2f} & {parity([sb_3k['results'][l]['fertility'] for l in LANGS]):.3f} & {parity([sb_3k['results'][l]['chars_per_token']/sb_3k['results']['en']['chars_per_token'] for l in LANGS]):.3f} \\\\",
        r"\midrule",
        f"\\multirow{{2}}{{*}}{{matched, 50 sent/lang}} & RotorSubword & {ga_m['vocab']:,} & {ga_m['merges']:,} & {ga_m['results']['en']['fertility']:.2f} & {parity([ga_m['results'][l]['fertility'] for l in LANGS]):.3f} & {parity([ga_m['results'][l]['chars_per_token']/ga_m['results']['en']['chars_per_token'] for l in LANGS]):.3f} \\\\",
        f"& $\\sqrt{{n}}$-BPE & {sb_m['vocab']:,} & {sb_m['merges']:,} & {sb_m['results']['en']['fertility']:.2f} & {parity([sb_m['results'][l]['fertility'] for l in LANGS]):.3f} & {parity([sb_m['results'][l]['chars_per_token']/sb_m['results']['en']['chars_per_token'] for l in LANGS]):.3f} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ]
    return "\n".join(lines)


def main():
    out_path = os.path.join(PAPER, "tables.tex")
    parts = [
        "% Auto-generated by scripts/generate_paper_tables.py — do not hand-edit.",
        "",
        "% === Table 1 ===",
        table1_ratio_vs_english(),
        "",
        "% === Table 2 ===",
        table2_raw_fertility(),
        "",
        "% === Table 3 ===",
        table3_matched(),
        "",
        "% === Table 4 ===",
        table4_cpt(),
        "",
        "% === Table sqrt-BPE ablation ===",
        table_sqrt_bpe_ablation(),
        "",
    ]
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))
    print(f"Wrote {out_path}")

    # Also write the ablation table to its own file for \input{} from the paper
    abl_path = os.path.join(PAPER, "table_sqrt_bpe_ablation.tex")
    with open(abl_path, "w", encoding="utf-8") as f:
        f.write(table_sqrt_bpe_ablation() + "\n")
    print(f"Wrote {abl_path}")

    # Also print headline summary to stdout
    rotor3k = load("results_3k.json")["results"]
    rotor500 = load("results_500.json")["results"]
    rotor50k = load("results_50k_rerun.json")
    rotor50k_r = rotor50k["results"]
    baselines_s = load("results_baselines_small.json")
    baselines_m = load("results_baselines_matched.json")

    print("\nHeadline parity numbers (re-run with fixed implementation):")
    print(f"  500-vocab (0 merges, char-level): "
          f"fert parity={parity([rotor500[l]['fertility'] for l in LANGS]):.3f}")
    print(f"  3k-vocab  (750 merges):           "
          f"fert parity={parity([rotor3k[l]['fertility'] for l in LANGS]):.3f}")
    print(f"  ~35k vocab (32,935 merges, matched-ish): "
          f"fert parity={parity([rotor50k_r[l]['fertility'] for l in LANGS]):.3f}, "
          f"CPT parity={parity([rotor50k_r[l]['chars_per_token']/rotor50k_r['en']['chars_per_token'] for l in LANGS]):.3f}")
    print()
    print("Best baseline parity (small slice):")
    for name in ["GPT-2", "Qwen-2.5", "XLM-R", "mGPT", "BLOOM", "Mistral", "BERT-multilingual"]:
        if name in baselines_s and "error" not in baselines_s[name]:
            r = baselines_s[name]["results"]
            print(f"  {name:<20} fert parity={parity([r[l]['fertility'] for l in LANGS]):.3f}")
    print()
    print("Baseline parity (matched slice, used in Table 3/4 comparison):")
    for name in ["GPT-2", "Qwen-2.5", "XLM-R", "mGPT", "BLOOM", "Mistral", "BERT-multilingual"]:
        if name in baselines_m and "error" not in baselines_m[name]:
            r = baselines_m[name]["results"]
            print(f"  {name:<20} fert parity={parity([r[l]['fertility'] for l in LANGS]):.3f}")


if __name__ == "__main__":
    main()
