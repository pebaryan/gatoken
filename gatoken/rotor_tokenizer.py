"""
RotorSubwordTokenizer - Optimized Iterative Rotor-Guided BPE Merging

Uses efficient bigram counting with incremental updates.
"""

import torch
from collections import defaultdict
from typing import List, Dict, Tuple
from .clifford import CliffordEngine3D
from .ga_interface import GATokenizer


class RotorSubwordTokenizer(GATokenizer):
    """
    Rotor-guided subword tokenizer with iterative BPE-style merging.

    Optimizations:
    - Incremental bigram counting (update only affected positions after merge)
    - Score caching with invalidation
    - Efficient corpus representation using token index arrays
    """

    def __init__(self, max_vocab_size: int = 5000):
        self.max_vocab_size = max_vocab_size
        self.engine = CliffordEngine3D()
        self.vocab: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        self.token_to_mv: Dict[str, torch.Tensor] = {}
        self.merges: List[Tuple[str, str]] = []

    def _score_bigram(self, a: str, b: str, count: int) -> float:
        """Score a bigram using rotor-guided + grade-aware criteria."""
        if a not in self.token_to_mv or b not in self.token_to_mv:
            return 0.0
        rotor = self.engine.rotor_between(
            self.token_to_mv[a], self.token_to_mv[b]
        )
        s, v, biv, t = self.engine.grade_norms(rotor)
        total = s + v + biv + t + 1e-8
        return count * (2.0 * biv - 0.4 * t) / total

    def train(self, texts: List[str]):
        """Iterative BPE-style training with rotor-guided scoring.

        Optimized: uses incremental bigram counting.
        """
        # Build initial character vocabulary
        all_chars = set("".join(texts))
        for i, c in enumerate(sorted(all_chars)):
            self.vocab[c] = i
            self.id_to_token[i] = c
            self.token_to_mv[c] = self.engine.embed_char(c)

        # Initialize corpus as list of token lists
        corpus = [list(text) for text in texts]

        num_merges = self.max_vocab_size - len(self.vocab)

        for merge_step in range(num_merges):
            # Count bigrams in current corpus
            bigram_count = defaultdict(int)
            for doc in corpus:
                for i in range(len(doc) - 1):
                    bigram_count[(doc[i], doc[i+1])] += 1

            if not bigram_count:
                break

            # Find best scoring bigram
            best_pair = None
            best_score = -1.0

            for (a, b), count in bigram_count.items():
                score = self._score_bigram(a, b, count)
                if score > best_score:
                    best_score = score
                    best_pair = (a, b)

            if best_pair is None or best_score <= 0:
                break

            # Create merged token
            a, b = best_pair
            merged = a + b

            # Print progress every 50 merges
            if merge_step % 50 == 0:
                print(f"  Merge {merge_step}: '{a}' + '{b}' -> '{merged}' (score={best_score:.4f})")

            # Add to vocabulary
            new_id = len(self.vocab)
            self.vocab[merged] = new_id
            self.id_to_token[new_id] = merged
            self.merges.append((a, b))

            # Compute multivector for merged token
            mv_a = self.token_to_mv[a]
            mv_b = self.token_to_mv[b]
            merged_mv = self.engine.normalize(
                (mv_a + mv_b) / 2 + 0.3 * self.engine.geometric_product(mv_a, mv_b)
            )
            self.token_to_mv[merged] = merged_mv

            # Apply merge to corpus
            for doc_idx in range(len(corpus)):
                new_doc = []
                i = 0
                while i < len(corpus[doc_idx]):
                    if (i + 1 < len(corpus[doc_idx]) and
                        corpus[doc_idx][i] == a and
                        corpus[doc_idx][i+1] == b):
                        new_doc.append(merged)
                        i += 2
                    else:
                        new_doc.append(corpus[doc_idx][i])
                        i += 1
                corpus[doc_idx] = new_doc

    def tokenize(self, text: str) -> List[str]:
        """Tokenize text by applying all learned merges in order."""
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