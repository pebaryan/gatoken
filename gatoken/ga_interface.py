"""
GA Tokenizer Interface

This defines the contract for any Geometric Algebra-aware tokenizer.
Future GA-based implementations should subclass GATokenizer.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any
import torch


class GATokenizer(ABC):
    """
    Abstract interface for GA-aware tokenizers.

    The goal is to reduce language bias (e.g. Indonesian vs English)
    by using geometric operations instead of pure statistical merging.
    """

    @abstractmethod
    def encode(self, text: str, **kwargs) -> List[int]:
        """Encode text into token IDs."""
        pass

    @abstractmethod
    def decode(self, token_ids: List[int], **kwargs) -> str:
        """Decode token IDs back to text."""
        pass

    @abstractmethod
    def tokenize(self, text: str) -> List[str]:
        """Return the list of token strings (for inspection)."""
        pass

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        pass

    def get_metrics(self, text: str) -> Dict[str, Any]:
        """Basic metrics for a single text."""
        tokens = self.tokenize(text)
        return {
            "num_tokens": len(tokens),
            "num_chars": len(text),
            "tokens_per_char": len(tokens) / max(len(text), 1),
            "tokens": tokens,
        }

    def __call__(self, text: str, **kwargs):
        return self.encode(text, **kwargs)


class StandardTokenizer(GATokenizer):
    """
    Wrapper around a standard HuggingFace tokenizer.
    Used as baseline for comparison.
    """

    def __init__(self, hf_tokenizer):
        self.hf_tokenizer = hf_tokenizer

    def encode(self, text: str, **kwargs) -> List[int]:
        return self.hf_tokenizer.encode(text, **kwargs)

    def decode(self, token_ids: List[int], **kwargs) -> str:
        return self.hf_tokenizer.decode(token_ids, **kwargs)

    def tokenize(self, text: str) -> List[str]:
        return self.hf_tokenizer.tokenize(text)

    @property
    def vocab_size(self) -> int:
        return self.hf_tokenizer.vocab_size
