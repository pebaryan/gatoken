"""
Re-run all benchmarks referenced in paper/PAPER.tex with the bug-fixed
RotorSubword implementation. Writes JSON snapshots that downstream
LaTeX generation reads, so no number in the paper is hand-copied.
"""

import argparse
import json
import os
import time
from typing import Dict, List

import torch

from gatoken import (
    RotorSubwordTokenizer,
    StandardTokenizer,
    compute_metrics,
)
from gatoken.metrics import TokenizerMetrics


HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FLORES_DIR = os.path.join(ROOT, "data", "flores101_dataset", "devtest")
# Paper artifacts (JSONs, logs, LaTeX) live outside this repo. Override with
# $GATOKEN_PAPER_DIR; default to ./paper-out so we don't leak the writer's
# private path into the public code repo.
OUT_DIR = os.environ.get(
    "GATOKEN_PAPER_DIR",
    os.path.join(ROOT, "paper-out"),
)
os.makedirs(OUT_DIR, exist_ok=True)

LANGUAGES = {
    "en": ("eng", "English"),
    "id": ("ind", "Indonesian"),
    "zh": ("zho_simpl", "Chinese"),
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


def load_flores() -> Dict[str, List[str]]:
    corpora = {}
    for lang, (code, _name) in LANGUAGES.items():
        fpath = os.path.join(FLORES_DIR, f"{code}.devtest")
        if os.path.exists(fpath):
            with open(fpath, "r", encoding="utf-8") as f:
                corpora[lang] = [line.strip() for line in f if line.strip()]
    return corpora


def metrics_dict(m: TokenizerMetrics) -> dict:
    return {
        "fertility": m.fertility,
        "chars_per_token": m.chars_per_token,
        "tokens_per_char": m.tokens_per_char,
        "total_tokens": m.total_tokens,
        "total_chars": m.total_chars,
        "total_words": m.total_words,
    }


def eval_rotor(corpora, train_n, eval_slice, max_vocab, batch_size, tag):
    print(f"\n=== Training RotorSubword [{tag}]: "
          f"train={train_n}/lang, vocab<={max_vocab}, batch={batch_size} ===")
    train_texts = []
    for lang in sorted(corpora):
        train_texts.extend(corpora[lang][:train_n])
    t0 = time.time()
    tok = RotorSubwordTokenizer(
        max_vocab_size=max_vocab,
        freq_weight=0.0,
        batch_size=batch_size,
    )
    tok.train(train_texts)
    elapsed = time.time() - t0
    print(f"   trained: vocab={tok.vocab_size}, merges={len(tok.merges)}, "
          f"time={elapsed:.0f}s")

    a, b = eval_slice
    results = {}
    for lang in sorted(corpora):
        eval_texts = corpora[lang][a:b]
        m = compute_metrics(tok, eval_texts, lang)
        results[lang] = metrics_dict(m)
    return {
        "tag": tag,
        "vocab": tok.vocab_size,
        "merges": len(tok.merges),
        "train_sentences_per_lang": train_n,
        "eval_slice": list(eval_slice),
        "elapsed_seconds": elapsed,
        "results": results,
    }


def eval_baseline(model_id, corpora, eval_slice):
    from transformers import AutoTokenizer
    try:
        hf = AutoTokenizer.from_pretrained(model_id)
    except Exception as e:
        return {"error": str(e)}
    tok = StandardTokenizer(hf)
    a, b = eval_slice
    results = {}
    for lang in sorted(corpora):
        eval_texts = corpora[lang][a:b]
        m = compute_metrics(tok, eval_texts, lang)
        results[lang] = metrics_dict(m)
    return {"vocab": hf.vocab_size, "results": results}


class SqrtBPETokenizer:
    """Frequency-only BPE with score = sqrt(count). Used as the ablation baseline
    against RotorSubword, since the buggy implementation of RotorSubword was
    behaviorally identical to this. Same training and evaluation pipeline as
    RotorSubword (batch merging, same eval) so the comparison is apples-to-apples."""

    UNK_TOKEN = "<unk>"

    def __init__(self, max_vocab_size, batch_size=1):
        from collections import defaultdict as _dd
        self.max_vocab_size = max_vocab_size
        self.batch_size = max(1, batch_size)
        self.vocab = {self.UNK_TOKEN: 0}
        self.merges = []
        self._dd = _dd

    def train(self, texts):
        chars = sorted(set("".join(texts)))
        for c in chars:
            if c not in self.vocab:
                self.vocab[c] = len(self.vocab)
        corpus = [list(t) for t in texts]
        remaining = self.max_vocab_size - len(self.vocab)
        step = 0
        while remaining > 0:
            bg = self._dd(int)
            for d in corpus:
                for i in range(len(d) - 1):
                    bg[(d[i], d[i+1])] += 1
            if not bg:
                break
            scored = sorted(((c**0.5, a, b) for (a, b), c in bg.items()), reverse=True)
            accepted, used = [], set()
            for s, a, b in scored:
                if len(accepted) >= min(self.batch_size, remaining):
                    break
                if a in used or b in used:
                    continue
                accepted.append((s, a, b))
                used.add(a); used.add(b)
            if not accepted:
                accepted = [scored[0]]
            for _, a, b in accepted:
                merged = a + b
                if merged not in self.vocab:
                    self.vocab[merged] = len(self.vocab)
                    self.merges.append((a, b))
                    remaining -= 1
                new_corpus = []
                for d in corpus:
                    new = []
                    i = 0
                    while i < len(d):
                        if i + 1 < len(d) and d[i] == a and d[i+1] == b:
                            new.append(merged); i += 2
                        else:
                            new.append(d[i]); i += 1
                    new_corpus.append(new)
                corpus = new_corpus
                if remaining <= 0:
                    break
            step += 1
            if step % 50 == 0:
                print(f"  sqrt-bpe step {step}: {len(self.merges)} merges, "
                      f"vocab {len(self.vocab)}", flush=True)

    def tokenize(self, text):
        tokens = list(text)
        for a, b in self.merges:
            merged = a + b
            new = []
            i = 0
            while i < len(tokens):
                if i + 1 < len(tokens) and tokens[i] == a and tokens[i+1] == b:
                    new.append(merged); i += 2
                else:
                    new.append(tokens[i]); i += 1
            tokens = new
        return tokens

    @property
    def vocab_size(self):
        return len(self.vocab)


def eval_sqrt_bpe(corpora, train_n, eval_slice, max_vocab, batch_size, tag):
    print(f"\n=== Training sqrt-BPE [{tag}]: "
          f"train={train_n}/lang, vocab<={max_vocab}, batch={batch_size} ===")
    train_texts = []
    for lang in sorted(corpora):
        train_texts.extend(corpora[lang][:train_n])
    t0 = time.time()
    tok = SqrtBPETokenizer(max_vocab_size=max_vocab, batch_size=batch_size)
    tok.train(train_texts)
    elapsed = time.time() - t0
    print(f"   trained: vocab={tok.vocab_size}, merges={len(tok.merges)}, "
          f"time={elapsed:.0f}s")

    a, b = eval_slice
    results = {}
    for lang in sorted(corpora):
        m = compute_metrics(tok, corpora[lang][a:b], lang)
        results[lang] = metrics_dict(m)
    return {
        "tag": tag,
        "vocab": tok.vocab_size,
        "merges": len(tok.merges),
        "train_sentences_per_lang": train_n,
        "eval_slice": list(eval_slice),
        "elapsed_seconds": elapsed,
        "results": results,
    }


def run_small_vocab(corpora):
    """Faithful reproduction of paper's small-vocab setting (max_vocab=500).
    Note: 12-lang base char set is ~2249, so 500 produces 0 merges (char-level).
    """
    out = eval_rotor(
        corpora,
        train_n=100,
        eval_slice=(100, 200),
        max_vocab=500,
        batch_size=1,
        tag="small_500",
    )
    path = os.path.join(OUT_DIR, "results_500.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"   saved {path}")
    return out


def run_small_vocab_3k(corpora):
    """Honest small-vocab run: max_vocab=3000 so RotorSubword actually
    performs merges (~750 over a ~2249-char base)."""
    out = eval_rotor(
        corpora,
        train_n=100,
        eval_slice=(100, 200),
        max_vocab=3000,
        batch_size=10,
        tag="small_3k",
    )
    path = os.path.join(OUT_DIR, "results_3k.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"   saved {path}")
    return out


def run_matched_vocab(corpora):
    out = eval_rotor(
        corpora,
        train_n=50,
        eval_slice=(200, 300),
        max_vocab=50000,
        batch_size=100,
        tag="matched_50k",
    )
    path = os.path.join(OUT_DIR, "results_50k_rerun.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"   saved {path}")
    return out


def run_baselines(corpora, eval_slice, out_file):
    out = {}
    for name, mid in BASELINES:
        print(f"\n=== Baseline: {name} ({mid}) eval slice {eval_slice} ===")
        out[name] = eval_baseline(mid, corpora, eval_slice)
        if "error" in out[name]:
            print(f"   FAILED: {out[name]['error']}")
        else:
            print(f"   vocab={out[name]['vocab']}, "
                  f"en fert={out[name]['results']['en']['fertility']}")
    path = os.path.join(OUT_DIR, out_file)
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"   saved {path}")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--small", action="store_true", help="run 500-vocab")
    parser.add_argument("--small3k", action="store_true", help="run 3000-vocab")
    parser.add_argument("--matched", action="store_true", help="run 50k-vocab")
    parser.add_argument("--baselines-small", action="store_true")
    parser.add_argument("--baselines-matched", action="store_true")
    parser.add_argument("--sqrt-bpe-3k", action="store_true",
                        help="sqrt-frequency BPE ablation at 3k vocab")
    parser.add_argument("--sqrt-bpe-matched", action="store_true",
                        help="sqrt-frequency BPE ablation at matched vocab")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()
    if args.all:
        args.small = args.small3k = args.matched = True
        args.baselines_small = args.baselines_matched = True
        args.sqrt_bpe_3k = args.sqrt_bpe_matched = True

    torch.manual_seed(0)
    print("Loading FLORES-101...")
    corpora = load_flores()
    print(f"   {len(corpora)} languages, "
          f"{sum(len(v) for v in corpora.values())} total sentences")

    if args.small:
        run_small_vocab(corpora)
    if args.small3k:
        run_small_vocab_3k(corpora)
    if args.baselines_small:
        run_baselines(corpora, (100, 200), "results_baselines_small.json")
    if args.matched:
        run_matched_vocab(corpora)
    if args.baselines_matched:
        run_baselines(corpora, (200, 300), "results_baselines_matched.json")
    if args.sqrt_bpe_3k:
        out = eval_sqrt_bpe(corpora, 100, (100, 200), 3000, 10, "sqrt_bpe_3k")
        with open(os.path.join(OUT_DIR, "results_sqrt_bpe_3k.json"), "w") as f:
            json.dump(out, f, indent=2)
        print(f"   saved {os.path.join(OUT_DIR, 'results_sqrt_bpe_3k.json')}")
    if args.sqrt_bpe_matched:
        out = eval_sqrt_bpe(corpora, 50, (200, 300), 50000, 100, "sqrt_bpe_matched")
        with open(os.path.join(OUT_DIR, "results_sqrt_bpe_matched.json"), "w") as f:
            json.dump(out, f, indent=2)
        print(f"   saved {os.path.join(OUT_DIR, 'results_sqrt_bpe_matched.json')}")


if __name__ == "__main__":
    main()
