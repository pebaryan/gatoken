"""
Evaluation metrics for measuring language bias in tokenization.

Provides both whitespace-fertility and characters-per-token metrics.
Characters-per-token is more script-fair because it doesn't depend
on whitespace word segmentation (which doesn't exist in CJK/Thai).
"""

from typing import Dict, List
from dataclasses import dataclass


@dataclass
class TokenizerMetrics:
    language: str
    num_sentences: int
    total_tokens: int
    total_chars: int
    total_words: int
    fertility: float          # tokens per whitespace-delimited word
    chars_per_token: float    # characters per token (script-fair)
    tokens_per_char: float   # tokens per character
    compression_ratio: float # chars per token (same as chars_per_token)


def compute_metrics(tokenizer, texts: List[str], language: str) -> TokenizerMetrics:
    """Compute bias metrics for a list of texts."""
    total_tokens = 0
    total_chars = 0
    total_words = 0

    for text in texts:
        tokens = tokenizer.tokenize(text)
        total_tokens += len(tokens)
        total_chars += len(text)
        total_words += len(text.split())

    num_sentences = len(texts)
    fertility = total_tokens / max(total_words, 1)
    tokens_per_char = total_tokens / max(total_chars, 1)
    chars_per_token = total_chars / max(total_tokens, 1)

    return TokenizerMetrics(
        language=language,
        num_sentences=num_sentences,
        total_tokens=total_tokens,
        total_chars=total_chars,
        total_words=total_words,
        fertility=round(fertility, 3),
        chars_per_token=round(chars_per_token, 3),
        tokens_per_char=round(tokens_per_char, 4),
        compression_ratio=round(chars_per_token, 3),
    )


def compare_languages(en_metrics: TokenizerMetrics, other_metrics: TokenizerMetrics) -> Dict:
    """Compare English vs another language's tokenization fairness."""
    fertility_ratio = other_metrics.fertility / max(en_metrics.fertility, 1e-6)
    cpt_ratio = other_metrics.chars_per_token / max(en_metrics.chars_per_token, 1e-6)

    return {
        "fertility_ratio": round(fertility_ratio, 3),
        "chars_per_token_ratio": round(cpt_ratio, 3),
        "parity_fertility": round(min(1.0, 1.0/fertility_ratio), 3) if fertility_ratio >= 1 else round(fertility_ratio, 3),
        "parity_cpt": round(min(en_metrics.chars_per_token, other_metrics.chars_per_token) / max(en_metrics.chars_per_token, other_metrics.chars_per_token), 3),
    }