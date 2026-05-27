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

    UNK_TOKEN = "<unk>"

    def __init__(self, max_vocab_size: int = 5000, freq_weight: float = 0.0,
                 batch_size: int = 1):
        self.max_vocab_size = max_vocab_size
        self.freq_weight = freq_weight
        self.batch_size = max(1, batch_size)
        self.engine = CliffordEngine3D()
        self.vocab: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        self.token_to_mv: Dict[str, torch.Tensor] = {}
        self.merges: List[Tuple[str, str]] = []
        self._alignment_cache: Dict[Tuple[str, str], float] = {}
        # Reserve id 0 for an explicit unknown token so OOV characters don't
        # silently collide with whichever symbol happens to be first in vocab.
        self.vocab[self.UNK_TOKEN] = 0
        self.id_to_token[0] = self.UNK_TOKEN
        self.token_to_mv[self.UNK_TOKEN] = torch.zeros(self.engine.mv_dim)

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

    def _accept_merge(self, a: str, b: str) -> str:
        """Register a single (a,b) merge in vocab + merge list, return the merged token."""
        merged = a + b
        if merged in self.vocab:
            return merged
        new_id = len(self.vocab)
        self.vocab[merged] = new_id
        self.id_to_token[new_id] = merged
        self.merges.append((a, b))
        mv_a = self.token_to_mv[a]
        mv_b = self.token_to_mv[b]
        self.token_to_mv[merged] = self.engine.normalize(
            (mv_a + mv_b) / 2 + 0.3 * self.engine.geometric_product(mv_a, mv_b)
        )
        return merged

    @staticmethod
    def _apply_merge_to_corpus(corpus, a, b, merged):
        for doc_idx in range(len(corpus)):
            doc = corpus[doc_idx]
            new_doc = []
            i = 0
            while i < len(doc):
                if i + 1 < len(doc) and doc[i] == a and doc[i+1] == b:
                    new_doc.append(merged)
                    i += 2
                else:
                    new_doc.append(doc[i])
                    i += 1
            corpus[doc_idx] = new_doc

    def train(self, texts: List[str]):
        """Iterative rotor-guided BPE. Supports batch merging via self.batch_size."""
        # Build initial character vocabulary on top of the reserved UNK at id 0.
        all_chars = set("".join(texts))
        next_id = len(self.vocab)
        for c in sorted(all_chars):
            if c in self.vocab:
                continue
            self.vocab[c] = next_id
            self.id_to_token[next_id] = c
            self.token_to_mv[c] = self.engine.embed_char(c)
            next_id += 1

        corpus = [list(text) for text in texts]
        merges_remaining = self.max_vocab_size - len(self.vocab)
        step = 0

        while merges_remaining > 0:
            bigram_count = defaultdict(int)
            for doc in corpus:
                for i in range(len(doc) - 1):
                    bigram_count[(doc[i], doc[i+1])] += 1
            if not bigram_count:
                break

            # Score every candidate
            scored = []
            for (a, b), count in bigram_count.items():
                alignment = self._alignment_score(a, b)
                s = self._score_bigram(a, b, count, alignment)
                if self.freq_weight == 0.0 and s <= 0:
                    continue
                scored.append((s, a, b))
            if not scored:
                break
            scored.sort(reverse=True)

            # Pick up to batch_size non-conflicting pairs (no shared endpoint).
            accepted = []
            used = set()
            for s, a, b in scored:
                if len(accepted) >= min(self.batch_size, merges_remaining):
                    break
                if a in used or b in used:
                    continue
                accepted.append((s, a, b))
                used.add(a); used.add(b)
            if not accepted:
                # Fall back to a single best merge if everything conflicts.
                s, a, b = scored[0]
                accepted = [(s, a, b)]

            top_score = accepted[0][0]
            if step % 100 == 0 or self.batch_size > 1:
                lead_a, lead_b = accepted[0][1], accepted[0][2]
                # Force ASCII-safe output so Windows cp1252 stdout doesn't crash on CJK.
                safe_a = lead_a[:5].encode("ascii", "replace").decode("ascii")
                safe_b = lead_b[:5].encode("ascii", "replace").decode("ascii")
                print(f"  Step {step}: accepted {len(accepted)} merges, "
                      f"top '{safe_a}+{safe_b}' (score={top_score:.4f})",
                      flush=True)

            for _, a, b in accepted:
                merged = self._accept_merge(a, b)
                self._apply_merge_to_corpus(corpus, a, b, merged)
                merges_remaining -= 1
                if merges_remaining <= 0:
                    break
            step += 1

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
        unk_id = self.vocab[self.UNK_TOKEN]
        return [self.vocab.get(t, unk_id) for t in tokens]

    def decode(self, token_ids: List[int], **kwargs) -> str:
        return "".join(self.id_to_token.get(i, "<unk>") for i in token_ids)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)