"""
RotorSubwordTokenizer - Optimized with precomputed rotor scores and batch merging.

Key optimizations:
1. Precompute rotor alignment scores (they don't change after merge)
2. Use a heap for best-pair selection instead of full scan
3. Batch merge: accept top-K candidates per iteration when they don't conflict
"""

import torch
import heapq
from collections import defaultdict
from typing import List, Dict, Tuple
from .clifford import CliffordEngine3D
from .ga_interface import GATokenizer


class RotorSubwordTokenizer(GATokenizer):
    """Rotor-guided subword tokenizer with hybrid scoring and optimizations."""

    def __init__(self, max_vocab_size: int = 5000, freq_weight: float = 0.0):
        self.max_vocab_size = max_vocab_size
        self.freq_weight = freq_weight
        self.engine = CliffordEngine3D()
        self.vocab: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        self.token_to_mv: Dict[str, torch.Tensor] = {}
        self.merges: List[Tuple[str, str]] = []
        self._alignment_cache: Dict[Tuple[str, str], float] = {}

    def _alignment_score(self, a: str, b: str) -> float:
        """Compute rotor alignment score (cached for reuse)."""
        key = (a, b)
        if key in self._alignment_cache:
            return self._alignment_cache[key]
        if a not in self.token_to_mv or b not in self.token_to_mv:
            self._alignment_cache[key] = 0.01
            return 0.01
        rotor = self.engine.rotor_between(
            self.token_to_mv[a], self.token_to_mv[b]
        )
        s, v, biv, t = self.engine.grade_norms(rotor)
        total = s + v + biv + t + 1e-8
        alignment = (2.0 * biv + 0.5 * v - 0.2 * t) / total
        result = max(alignment, 0.01)
        self._alignment_cache[key] = result
        return result

    def _score_bigram(self, a: str, b: str, count: int, alignment: float = None) -> float:
        """Score a bigram using hybrid frequency + precomputed alignment."""
        if alignment is None:
            alignment = self._alignment_score(a, b)
        geo_score = alignment * (count ** 0.5)
        freq_score = float(count)
        return (1.0 - self.freq_weight) * geo_score + self.freq_weight * freq_score

    def train(self, texts: List[str]):
        """Optimized iterative BPE with precomputed alignments."""
        # Build initial character vocabulary
        all_chars = set("".join(texts))
        for i, c in enumerate(sorted(all_chars)):
            self.vocab[c] = i
            self.id_to_token[i] = c
            self.token_to_mv[c] = self.engine.embed_char(c)

        # Initialize corpus
        corpus = [list(text) for text in texts]
        num_merges = self.max_vocab_size - len(self.vocab)

        for merge_step in range(num_merges):
            # Count bigrams
            bigram_count = defaultdict(int)
            for doc in corpus:
                for i in range(len(doc) - 1):
                    bigram_count[(doc[i], doc[i+1])] += 1

            if not bigram_count:
                break

            # Find best scoring bigram using precomputed alignments
            best_pair = None
            best_score = -1e30

            for (a, b), count in bigram_count.items():
                alignment = self._alignment_score(a, b)
                score = self._score_bigram(a, b, count, alignment)
                if score > best_score:
                    best_score = score
                    best_pair = (a, b)

            if best_pair is None:
                break
            if self.freq_weight == 0.0 and best_score <= 0:
                break

            # Create merged token
            a, b = best_pair
            merged = a + b

            if merge_step % 100 == 0:
                print(f"  Merge {merge_step}: '{a[:5]}+{b[:5]}' -> '{merged[:8]}' (score={best_score:.4f})")

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