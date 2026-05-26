"""
TokenMultivectorTokenizer - Token-level Multivector Representations

Uses the shared differentiable CliffordEngine3D from clifford.py.
Each subword token is assigned a multivector.
Encoding uses geometric similarity.
"""

import torch
import torch.nn as nn
from collections import defaultdict
from typing import List, Dict, Tuple
from .clifford import CliffordEngine3D
from .ga_interface import GATokenizer


class TokenMultivectorTokenizer(GATokenizer):
    """
    Token-level Multivector Tokenizer with Geometric Encoding.

    - Subword vocabulary via rotor-guided merging
    - Each token has a multivector (nn.Parameter for training)
    - Correct Cl(3,0) geometric product (differentiable)
    - Non-degenerate character embeddings
    """

    def __init__(self, max_vocab_size: int = 5000):
        self.max_vocab_size = max_vocab_size
        self.engine = CliffordEngine3D()
        self.vocab: List[str] = []
        self.token_to_mv: Dict[str, nn.Parameter] = {}
        self.merges: List[Tuple[str, str]] = []

    def train(self, texts: List[str]):
        # Build initial character vocabulary
        all_chars = set("".join(texts))
        for c in sorted(all_chars):
            self.vocab.append(c)
            mv = self.engine.embed_char(c)
            self.token_to_mv[c] = nn.Parameter(mv)

        # Count bigrams across all scripts (no Chinese skipping)
        bigram_count = defaultdict(int)
        for text in texts:
            for a, b in zip(text, text[1:]):
                bigram_count[(a, b)] += 1

        # Score bigrams using rotor-guided + grade-aware scoring
        scored = []
        for (a, b), count in bigram_count.items():
            if a in self.token_to_mv and b in self.token_to_mv:
                mv_a = self.token_to_mv[a].data
                mv_b = self.token_to_mv[b].data
                rotor = self.engine.rotor_between(mv_a, mv_b)
                s, v, biv, t = self.engine.grade_norms(rotor)
                total = s + v + biv + t + 1e-8
                score = count * (2.0 * biv - 0.4 * t) / total
                if score > 0:
                    scored.append(((a, b), score))

        scored.sort(key=lambda x: x[1], reverse=True)

        for (a, b), _ in scored[:self.max_vocab_size - len(self.vocab)]:
            merged = a + b
            if merged not in self.token_to_mv:
                mv_a = self.token_to_mv[a].data
                mv_b = self.token_to_mv[b].data
                # Compose multivector for merged token
                merged_mv = self.engine.normalize(
                    (mv_a + mv_b) / 2 + 0.1 * torch.randn(8)
                )
                self.token_to_mv[merged] = nn.Parameter(merged_mv)
                self.vocab.append(merged)
                self.merges.append((a, b))

    def tokenize(self, text: str) -> List[str]:
        tokens = list(text)
        for a, b in self.merges:
            merged = a + b
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i + 1 < len(tokens) and tokens[i] == a and tokens[i+1] == b:
                    new_tokens.append(merged)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
        return tokens

    def encode(self, text: str, **kwargs) -> List[int]:
        subwords = self.tokenize(text)
        ids = []
        for sw in subwords:
            if sw not in self.token_to_mv:
                ids.append(0)
                continue

            # Simple lookup (no O(V) scan for now)
            try:
                idx = self.vocab.index(sw)
                ids.append(idx)
            except ValueError:
                ids.append(0)

        return ids

    def decode(self, token_ids: List[int], **kwargs) -> str:
        result = []
        for i in token_ids:
            if i < len(self.vocab):
                result.append(self.vocab[i])
            else:
                result.append("<unk>")
        return "".join(result)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)