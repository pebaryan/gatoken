"""
Clifford Algebra engine for Cl(3,0) — shared module.

Provides:
- Correct geometric product (verified against canonical Cl(3) rules)
- Differentiable GP via matrix multiplication
- Rotor operations
- Grade extraction
- Character embedding (no degenerate ord%3)
"""

import torch
import torch.nn as nn
from typing import Tuple

# Basis: {1, e1, e2, e3, e12, e13, e23, e123}
# Indices: 0=1, 1=e1, 2=e2, 3=e3, 4=e12, 5=e13, 6=e23, 7=e123

# GP table for Cl(3,0), derived from canonical anticommutation rules:
#   ei*ei = 1, ei*ej = -ej*ei for i != j.
# Verified exhaustively against an independent derivation in run_gp_tests().
GP_TABLE = {
    (0,0):( 0, 1),(0,1):( 1, 1),(0,2):( 2, 1),(0,3):( 3, 1),
    (0,4):( 4, 1),(0,5):( 5, 1),(0,6):( 6, 1),(0,7):( 7, 1),
    (1,0):( 1, 1),(1,1):( 0, 1),(1,2):( 4, 1),(1,3):( 5, 1),
    (1,4):( 2, 1),(1,5):( 3, 1),(1,6):( 7, 1),(1,7):( 6, 1),
    (2,0):( 2, 1),(2,1):( 4,-1),(2,2):( 0, 1),(2,3):( 6, 1),
    (2,4):( 1,-1),(2,5):( 7,-1),(2,6):( 3, 1),(2,7):( 5,-1),
    (3,0):( 3, 1),(3,1):( 5,-1),(3,2):( 6,-1),(3,3):( 0, 1),
    (3,4):( 7, 1),(3,5):( 1,-1),(3,6):( 2,-1),(3,7):( 4, 1),
    (4,0):( 4, 1),(4,1):( 2,-1),(4,2):( 1, 1),(4,3):( 7, 1),
    (4,4):( 0,-1),(4,5):( 6,-1),(4,6):( 5, 1),(4,7):( 3,-1),
    (5,0):( 5, 1),(5,1):( 3,-1),(5,2):( 7,-1),(5,3):( 1, 1),
    (5,4):( 6, 1),(5,5):( 0,-1),(5,6):( 4,-1),(5,7):( 2, 1),
    (6,0):( 6, 1),(6,1):( 7, 1),(6,2):( 3,-1),(6,3):( 2, 1),
    (6,4):( 5,-1),(6,5):( 4, 1),(6,6):( 0,-1),(6,7):( 1,-1),
    (7,0):( 7, 1),(7,1):( 6, 1),(7,2):( 5,-1),(7,3):( 4, 1),
    (7,4):( 3,-1),(7,5):( 2, 1),(7,6):( 1,-1),(7,7):( 0,-1),
}


def _build_gp_matrix() -> torch.Tensor:
    """Build 8x8 GP matrix for differentiable computation.

    The matrix M is such that gp(a, b) = M @ (a ⊗ b) reshaped,
    but for efficiency we use the bilinear form directly.
    """
    # GP matrix: gp_result[k] = sum_{i,j} GP_MAT[k, i, j] * a[i] * b[j]
    gp_mat = torch.zeros(8, 8, 8)
    for (i, j), (k, s) in GP_TABLE.items():
        gp_mat[k, i, j] = s
    return gp_mat


GP_MAT = _build_gp_matrix()


class CliffordEngine3D(nn.Module):
    """Differentiable Cl(3,0) engine using matrix-based geometric product."""

    def __init__(self):
        super().__init__()
        self.mv_dim = 8
        # Don't register GP_MAT as a parameter — it's fixed algebra
        self.register_buffer('gp_mat', GP_MAT.clone())

    def geometric_product(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Differentiable geometric product via matrix multiplication.

        a, b: tensors of shape (8,) or (batch, 8)
        Returns: tensor of same leading shape
        """
        if a.dim() == 1:
            a = a.unsqueeze(0)
            b = b.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False

        # outer product: (batch, 8, 1) * (batch, 1, 8) = (batch, 8, 8)
        outer = a.unsqueeze(-1) * b.unsqueeze(-2)  # (batch, 8, 8)

        # einsum: result[k] = gp_mat[k,i,j] * outer[i,j]
        # gp_mat shape: (8, 8, 8), outer shape: (batch, 8, 8)
        result = torch.einsum('kij,bij->bk', self.gp_mat, outer)

        if squeeze:
            result = result.squeeze(0)
        return result

    def grade_norms(self, mv: torch.Tensor):
        """Extract norms of each grade. Works with gradients."""
        scalar = torch.abs(mv[0])
        vector = torch.sqrt(torch.sum(mv[1:4] ** 2) + 1e-12)
        bivector = torch.sqrt(torch.sum(mv[4:7] ** 2) + 1e-12)
        trivector = torch.abs(mv[7])
        return scalar, vector, bivector, trivector

    def rotor_exp(self, bivector: torch.Tensor, theta: float = None) -> torch.Tensor:
        """R = exp(theta/2 * B_hat) = cos(theta/2) + sin(theta/2) * B_hat.

        If `theta` is None, uses theta = ||bivector[4:7]||, so the rotor's
        magnitude actually reflects the input bivector's strength. Previously
        theta was hardcoded to 1.0, which made the rotor norm constant and
        collapsed the alignment score across all merge candidates.
        """
        norm = torch.sqrt(torch.sum(bivector[4:7] ** 2) + 1e-12)
        if theta is None:
            theta = norm
        half_theta = theta / 2 if torch.is_tensor(theta) else torch.tensor(theta / 2)
        cos_h = torch.cos(half_theta)
        sin_h = torch.sin(half_theta)

        rotor = torch.zeros_like(bivector)
        rotor[0] = cos_h
        B_hat = bivector[4:7] / norm
        rotor[4:7] = sin_h * B_hat
        return rotor

    def rotor_between(self, mv1: torch.Tensor, mv2: torch.Tensor) -> torch.Tensor:
        """Approximate rotor relating mv1 to mv2 via the bivector part of GP."""
        gp = self.geometric_product(mv1, mv2)
        biv = torch.zeros_like(gp)
        biv[4:7] = gp[4:7]
        return self.rotor_exp(biv)

    def embed_char(self, char: str) -> torch.Tensor:
        """Embed a character as a multivector using Unicode codepoint hashing.

        Uses a simple but non-degenerate hash: maps each character
        to a unique vector direction in Cl(3) based on its codepoint.
        """
        code = ord(char)
        mv = torch.zeros(self.mv_dim)

        # Use 3 different hash functions for the 3 vector basis directions
        # This gives ~2^3 = 8 distinguishable directions per group,
        # far better than ord % 3 which collapses to only 3.
        mv[1] = torch.sin(torch.tensor(code * 0.1))       # e1 component
        mv[2] = torch.cos(torch.tensor(code * 0.137))     # e2 component
        mv[3] = torch.sin(torch.tensor(code * 0.271 + 1))  # e3 component

        # Add a small bivector component for script distinction
        if self._is_chinese(char):
            mv[4] = 0.5 * torch.sin(torch.tensor(code * 0.19))
            mv[5] = 0.5 * torch.cos(torch.tensor(code * 0.23))

        # Normalize the vector part to unit length (true L2 norm).
        vec_norm = torch.norm(mv[1:4])
        if vec_norm > 1e-8:
            mv[1:4] = mv[1:4] / vec_norm

        return mv

    def _is_chinese(self, char: str) -> bool:
        return '\u4e00' <= char <= '\u9fff'

    def normalize(self, mv: torch.Tensor) -> torch.Tensor:
        norm = torch.norm(mv)
        if norm < 1e-8:
            return mv
        return mv / norm


# === Unit tests ===

# Basis blade indices used to derive the GP table from scratch.
# 0 -> 1 (scalar), 1 -> e1, 2 -> e2, 3 -> e3, 4 -> e12, 5 -> e13, 6 -> e23, 7 -> e123
_BASIS_BLADES = [(), (1,), (2,), (3,), (1,2), (1,3), (2,3), (1,2,3)]
_BLADE_INDEX = {b: i for i, b in enumerate(_BASIS_BLADES)}


def _multiply_blade(a, b):
    """Multiply two sorted-tuple blades using ei*ei=1, ei*ej=-ej*ei.
    Returns (sign, sorted_tuple) of the canonical result."""
    seq = list(a) + list(b)
    sign = 1
    while True:
        for i in range(len(seq) - 1):
            if seq[i] > seq[i+1]:
                seq[i], seq[i+1] = seq[i+1], seq[i]
                sign = -sign
                break
            elif seq[i] == seq[i+1]:
                del seq[i:i+2]
                break
        else:
            break
    return sign, tuple(seq)


def _derive_gp_table():
    """Derive the full 64-entry GP table from first principles."""
    table = {}
    for i, a in enumerate(_BASIS_BLADES):
        for j, b in enumerate(_BASIS_BLADES):
            sign, result = _multiply_blade(a, b)
            table[(i, j)] = (_BLADE_INDEX[result], sign)
    return table


def run_gp_tests():
    """Verify all 64 GP_TABLE entries against an independent derivation,
    plus end-to-end algebra properties (associativity, pseudoscalar identities)."""
    derived = _derive_gp_table()
    sign_mismatches = [(k, GP_TABLE[k], derived[k]) for k in derived if GP_TABLE[k] != derived[k]]

    errors = []
    if sign_mismatches:
        for (i, j), code, correct in sign_mismatches:
            errors.append(f"GP_TABLE[({i},{j})] = {code}, should be {correct}")

    engine = CliffordEngine3D()

    def mv(idx, s=1.0):
        v = torch.zeros(8); v[idx] = s; return v

    e1, e2, e3 = mv(1), mv(2), mv(3)
    e12, e13, e23 = mv(4), mv(5), mv(6)
    I = mv(7)

    # In Cl(3,0) the pseudoscalar I commutes with all elements.
    for v, label in [(e1, "e1"), (e2, "e2"), (e3, "e3"), (e12, "e12"), (e23, "e23")]:
        lhs = engine.geometric_product(v, I)
        rhs = engine.geometric_product(I, v)
        if not torch.allclose(lhs, rhs, atol=1e-6):
            errors.append(f"{label}*I != I*{label}: pseudoscalar should commute")

    # Vectors with disjoint-index bivectors commute (e.g., e1 with e23).
    for v_idx, b_idx, label in [(1, 6, "e1*e23"), (2, 5, "e2*e13"), (3, 4, "e3*e12")]:
        v, b = mv(v_idx), mv(b_idx)
        lhs = engine.geometric_product(v, b)
        rhs = engine.geometric_product(b, v)
        if not torch.allclose(lhs, rhs, atol=1e-6):
            errors.append(f"{label}: vector and disjoint-index bivector should commute")

    # Associativity spot-check across mixed grades.
    for a, b, c in [(e23, e1, e1), (e12, e13, e23), (e1, e2, I)]:
        lhs = engine.geometric_product(engine.geometric_product(a, b), c)
        rhs = engine.geometric_product(a, engine.geometric_product(b, c))
        if not torch.allclose(lhs, rhs, atol=1e-6):
            errors.append(f"associativity violated for triple")

    if errors:
        print("GP TESTS FAILED:")
        for e in errors:
            print(f"  {e}")
        return False
    print(f"All GP tests passed (64/64 table entries, commutativity, associativity).")
    return True


if __name__ == "__main__":
    run_gp_tests()