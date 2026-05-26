"""
Evaluation metrics for measuring language bias in tokenization.
Focused on Indonesian (id) vs English (en) comparison.
"""

from typing import Dict, List
from dataclasses import dataclass


@dataclass
class TokenizerMetrics:
    language: str
    num_sentences: int
    total_tokens: int
    total_chars: int
    fertility: float          # tokens per word (approx)
    tokens_per_char: float
    compression_ratio: float  # chars per token


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
    compression = total_chars / max(total_tokens, 1)

    return TokenizerMetrics(
        language=language,
        num_sentences=num_sentences,
        total_tokens=total_tokens,
        total_chars=total_chars,
        fertility=round(fertility, 3),
        tokens_per_char=round(tokens_per_char, 4),
        compression_ratio=round(compression, 2),
    )


def compare_languages(en_metrics: TokenizerMetrics, id_metrics: TokenizerMetrics) -> Dict:
    """Compare English vs Indonesian tokenization fairness."""
    fertility_ratio = id_metrics.fertility / en_metrics.fertility
    tokens_per_char_ratio = id_metrics.tokens_per_char / en_metrics.tokens_per_char

    return {
        "fertility_ratio (id/en)": round(fertility_ratio, 3),
        "tokens_per_char_ratio (id/en)": round(tokens_per_char_ratio, 3),
        "indonesian_is_worse": fertility_ratio > 1.15,  # >15% more tokens
        "parity_score": round(1.0 / max(fertility_ratio, 1e-6), 3),
    }
