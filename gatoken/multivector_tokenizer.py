"""
TokenMultivectorTokenizer v4 - Geometric Similarity Encoding

Subword tokens have multivectors.
Encoding now uses geometric similarity between input and token multivectors.
"""

import torch
import torch.nn as nn
from collections import defaultdict
from typing import List, Dict, Tuple
from .ga_interface import GATokenizer


GP_TABLE = {
    (0, 0): (0, 1), (0, 1): (1, 1), (0, 2): (2, 1), (0, 3): (3, 1),
    (0, 4): (4, 1), (0, 5): (5, 1), (0, 6): (6, 1), (0, 7): (7, 1),
    (1, 0): (1, 1), (1, 1): (0, 1), (1, 2): (4, 1), (1, 3): (5, 1),
    (1, 4): (2, 1), (1, 5): (3, 1), (1, 6): (7, 1), (1, 7): (6, 1),
    (2, 0): (2, 1), (2, 1): (4, -1), (2, 2): (0, 1), (2, 3): (6, 1),
    (2, 4): (1, -1), (2, 5): (7, 1), (2, 6): (3, 1), (2, 7): (5, -1),
    (3, 0): (3, 1), (3, 1): (5, -1), (3, 2): (6, -1), (3, 3): (0, 1),
    (3, 4): (7, 1), (3, 5): (1, -1), (3, 6): (2, -1), (3, 7): (4, -1),
    (4, 0): (4, 1), (4, 1): (2, -1), (4, 2): (1, 1), (4, 3): (7, -1),
    (4, 4): (0, -1), (4, 5): (6, 1), (4, 6): (5, -1), (4, 7): (3, -1),
    (5, 0): (5, 1), (5, 1): (3, -1), (5, 2): (7, -1), (5, 3): (1, 1),
    (5, 4): (6, -1), (5, 5): (0, -1), (5, 6): (4, 1), (5, 7): (2, 1),
    (6, 0): (6, 1), (6, 1): (7, -1), (6, 2): (3, -1), (6, 3): (2, 1),
    (6, 4): (5, 1), (6, 5): (4, -1), (6, 6): (0, -1), (6, 7): (1, -1),
    (7, 0): (7, 1), (7, 1): (6, 1), (7, 2): (5, 1), (7, 3): (4, 1),
    (7, 4): (3, 1), (7, 5): (2, -1), (7, 6): (1, 1), (7, 7): (0, 1),
}


class CliffordEngine3D:
    def __init__(self):
        self.mv_dim = 8

    def embed_char(self, char: str) -> torch.Tensor:
        mv = torch.zeros(self.mv_dim)
        idx = (ord(char) % 3) + 1
        mv[idx] = 1.0
        return mv

    def random_multivector(self) -> torch.Tensor:
        return torch.randn(self.mv_dim)

    def normalize(self, mv: torch.Tensor) -> torch.Tensor:
        norm = torch.norm(mv)
        if norm < 1e-8:
            return mv
        return mv / norm

    def geometric_product(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        result = torch.zeros(self.mv_dim, dtype=a.dtype)
        for i in range(self.mv_dim):
            if a[i] == 0: continue
            for j in range(self.mv_dim):
                if b[j] == 0: continue
                res_idx, sign = GP_TABLE.get((i, j), (0, 0))
                result[res_idx] += sign * a[i] * b[j]
        return result

    def grade_norms(self, mv: torch.Tensor):
        scalar = torch.abs(mv[0]).item()
        vector = torch.sqrt(torch.sum(mv[1:4]**2)).item()
        bivector = torch.sqrt(torch.sum(mv[4:7]**2)).item()
        trivector = torch.abs(mv[7]).item()
        return scalar, vector, bivector, trivector

    def rotor_exp(self, bivector: torch.Tensor, theta: float = 1.0) -> torch.Tensor:
        rotor = torch.zeros_like(bivector)
        norm = torch.sqrt(torch.sum(bivector[4:7]**2)).item()
        if norm < 1e-8:
            rotor[0] = 1.0
            return rotor
        B = bivector.clone()
        B[4:7] = B[4:7] / norm
        half_theta = theta / 2
        rotor[0] = torch.cos(torch.tensor(half_theta))
        rotor[4:7] = torch.sin(torch.tensor(half_theta)) * B[4:7]
        return rotor

    def rotor_between(self, mv1: torch.Tensor, mv2: torch.Tensor) -> torch.Tensor:
        gp = self.geometric_product(mv1, mv2)
        biv = torch.zeros_like(gp)
        biv[4:7] = gp[4:7]
        if torch.sqrt(torch.sum(biv[4:7]**2)).item() < 1e-8:
            return torch.zeros_like(gp)
        return self.rotor_exp(biv, theta=1.0)

    def similarity(self, a: torch.Tensor, b: torch.Tensor) -> float:
        gp = self.geometric_product(a, b)
        return float(gp[0])


class TokenMultivectorTokenizer(GATokenizer):
    """
    Token-level Multivector Tokenizer with Geometric Encoding

    - Subword vocabulary via rotor-guided merging
    - Each token has a multivector (stored as nn.Parameter)
    - Encoding uses geometric similarity
    """

    def __init__(self, max_vocab_size: int = 5000):
        self.max_vocab_size = max_vocab_size
        self.engine = CliffordEngine3D()
        self.vocab: List[str] = []
        self.token_to_mv: Dict[str, torch.nn.Parameter] = {}
        self.merges: List[Tuple[str, str]] = []

    def _is_chinese(self, char: str) -> bool:
        return '\u4e00' <= char <= '\u9fff'

    def train(self, texts: List[str]):
        # Base characters
        all_chars = set("".join(texts))
        for c in sorted(all_chars):
            self.vocab.append(c)
            mv = self.engine.embed_char(c)
            self.token_to_mv[c] = nn.Parameter(mv)

        # Rotor-guided merging
        bigram_count = defaultdict(int)
        for text in texts:
            for a, b in zip(text, text[1:]):
                if self._is_chinese(a) or self._is_chinese(b):
                    continue
                bigram_count[(a, b)] += 1

        scored = []
        for (a, b), count in bigram_count.items():
            if a in self.token_to_mv and b in self.token_to_mv:
                rotor = self.engine.rotor_between(
                    self.token_to_mv[a].data, self.token_to_mv[b].data
                )
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
                merged_mv = self.engine.normalize((mv_a + mv_b) / 2 + 0.1 * self.engine.random_multivector())
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

            char_mvs = [self.token_to_mv.get(c, torch.zeros(self.engine.mv_dim)) for c in sw]
            input_mv = torch.stack([m.data if isinstance(m, nn.Parameter) else m for m in char_mvs]).mean(dim=0)

            best_score = -float("inf")
            best_idx = 0

            for idx, token in enumerate(self.vocab):
                if token in self.token_to_mv:
                    score = self.engine.similarity(input_mv, self.token_to_mv[token].data)
                    if score > best_score:
                        best_score = score
                        best_idx = idx

            ids.append(best_idx)

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
