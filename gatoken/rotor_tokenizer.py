"""
RotorSubwordTokenizer v9 - Rotor-Guided Merging + Rich Chinese Embedding

- All scripts (including Chinese): Rotor-guided subword merging
- Chinese characters get richer multivector embeddings (vector + bivector)
"""

import torch
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
        if self._is_chinese(char):
            # Richer embedding for Chinese
            idx = (ord(char) % 3) + 1
            mv[idx] = 1.0
            biv_idx = 4 + (ord(char) % 3)
            mv[biv_idx] = 0.6
        else:
            # Simple vector for alphabetic scripts
            idx = (ord(char) % 3) + 1
            mv[idx] = 1.0
        return mv

    def _is_chinese(self, char: str) -> bool:
        return '\u4e00' <= char <= '\u9fff'

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


class RotorSubwordTokenizer(GATokenizer):
    """
    v9: Rotor-guided merging for all scripts + richer Chinese embeddings
    """

    def __init__(self, max_vocab_size: int = 5000):
        self.max_vocab_size = max_vocab_size
        self.engine = CliffordEngine3D()
        self.vocab: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        self.char_to_mv: Dict[str, torch.Tensor] = {}
        self.merges: List[Tuple[str, str]] = []

    def train(self, texts: List[str]):
        all_chars = set("".join(texts))
        for i, c in enumerate(sorted(all_chars)):
            self.vocab[c] = i
            self.id_to_token[i] = c
            self.char_to_mv[c] = self.engine.embed_char(c)

        bigram_count = defaultdict(int)
        for text in texts:
            for a, b in zip(text, text[1:]):
                bigram_count[(a, b)] += 1

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
