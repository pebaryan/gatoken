"""
RotorSubwordTokenizer - Rotor-Guided Subword Merging

Uses the shared CliffordEngine3D from clifford.py.
Correct geometric product, proper character embeddings, consistent Chinese handling.
"""

import torch
from collections import defaultdict
from typing import List, Dict, Tuple
from .clifford import CliffordEngine3D
from .ga_interface import GATokenizer


class RotorSubwordTokenizer(GATokenizer):
    """
    Rotor-guided subword tokenizer with grade-aware merging.

    - Characters embedded using non-degenerate Unicode hashing
    - Correct Cl(3,0) geometric product
    - Rotor-guided merging for all scripts (including Chinese)
    - Grade-aware scoring (favor bivector, penalize trivector)
    """

    def __init__(self, max_vocab_size: int = 5000):
        self.max_vocab_size = max_vocab_size
        self.engine = CliffordEngine3D()
        self.vocab: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        self.char_to_mv: Dict[str, torch.Tensor] = {}
        self.merges: List[Tuple[str, str]] = []

    def train(self, texts: List[str]):
        # Build initial character vocabulary
        all_chars = set("".join(texts))
        for i, c in enumerate(sorted(all_chars)):
            self.vocab[c] = i
            self.id_to_token[i] = c
            self.char_to_mv[c] = self.engine.embed_char(c)

        # Count bigrams across all scripts
        bigram_count = defaultdict(int)
        for text in texts:
            for a, b in zip(text, text[1:]):
                bigram_count[(a, b)] += 1

        # Score bigrams using rotor-guided + grade-aware scoring
        scored = []
        for (a, b), count in bigram_count.items():
            if a in self.char_to_mv and b in self.char_to_mv:
                rotor = self.engine.rotor_between(
                    self.char_to_mv[a], self.char_to_mv[b]
                )
                s, v, biv, t = self.engine.grade_norms(rotor)
                total = s + v + biv + t + 1e-8
                score = count * (2.0 * biv - 0.4 * t) / total
                if score > 0:
                    scored.append(((a, b), score))

        scored.sort(key=lambda x: x[1], reverse=True)

        for (a, b), _ in scored[:self.max_vocab_size - len(self.vocab)]:
            merged = a + b
            if merged not in self.vocab:
                new_id = len(self.vocab)
                self.vocab[merged] = new_id
                self.id_to_token[new_id] = merged
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
        tokens = self.tokenize(text)
        unk = list(self.vocab.keys())[0]
        return [self.vocab.get(t, self.vocab[unk]) for t in tokens]

    def decode(self, token_ids: List[int], **kwargs) -> str:
        return "".join(self.id_to_token.get(i, "<unk>") for i in token_ids)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)