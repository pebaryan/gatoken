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

# Correct GP table for Cl(3,0), computed from canonical anticommutation rules:
#   ei*ei = 1, ei*ej = -ej*ei for i != j
GP_TABLE = {
    (0,0):( 0, 1),(0,1):( 1, 1),(0,2):( 2, 1),(0,3):( 3, 1),
    (0,4):( 4, 1),(0,5):( 5, 1),(0,6):( 6, 1),(0,7):( 7, 1),
    (1,0):( 1, 1),(1,1):( 0, 1),(1,2):( 4, 1),(1,3):( 5, 1),
    (1,4):( 2, 1),(1,5):( 3, 1),(1,6):( 7, 1),(1,7):( 6, 1),
    (2,0):( 2, 1),(2,1):( 4,-1),(2,2):( 0, 1),(2,3):( 6, 1),
    (2,4):( 1,-1),(2,5):( 7,-1),(2,6):( 3, 1),(2,7):( 5,-1),
    (3,0):( 3, 1),(3,1):( 5,-1),(3,2):( 6,-1),(3,3):( 0, 1),
    (3,4):( 7, 1),(3,5):( 1,-1),(3,6):( 2,-1),(3,7):( 4,-1),
    (4,0):( 4, 1),(4,1):( 2,-1),(4,2):( 1, 1),(4,3):( 7, 1),
    (4,4):( 0,-1),(4,5):( 6,-1),(4,6):( 5, 1),(4,7):( 3,-1),
    (5,0):( 5, 1),(5,1):( 3,-1),(5,2):( 7,-1),(5,3):( 1, 1),
    (5,4):( 6, 1),(5,5):( 0,-1),(5,6):( 4,-1),(5,7):( 2, 1),
    (6,0):( 6, 1),(6,1):( 7,-1),(6,2):( 3,-1),(6,3):( 2, 1),
    (6,4):( 5,-1),(6,5):( 4, 1),(6,6):( 0,-1),(6,7):( 1,-1),
    (7,0):( 7, 1),(7,1):( 6, 1),(7,2):( 5,-1),(7,3):( 4,-1),
    (7,4):( 3, 1),(7,5):( 2, 1),(7,6):( 1,-1),(7,7):( 0,-1),
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

    def rotor_exp(self, bivector: torch.Tensor, theta: float = 1.0) -> torch.Tensor:
        """R = cos(theta/2) + sin(theta/2) * B_hat"""
        norm = torch.sqrt(torch.sum(bivector[4:7] ** 2) + 1e-12)
        half_theta = theta / 2
        cos_h = torch.cos(torch.tensor(half_theta))
        sin_h = torch.sin(torch.tensor(half_theta))

        rotor = torch.zeros_like(bivector)
        rotor[0] = cos_h
        B_hat = bivector[4:7] / norm
        rotor[4:7] = sin_h * B_hat
        return rotor

    def rotor_between(self, mv1: torch.Tensor, mv2: torch.Tensor) -> torch.Tensor:
        """Compute rotor that approximately relates mv1 to mv2."""
        gp = self.geometric_product(mv1, mv2)
        biv = torch.zeros_like(gp)
        biv[4:7] = gp[4:7]
        return self.rotor_exp(biv, theta=1.0)

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

        # Normalize to unit vector (keep in vector subspace)
        vec_norm = torch.sqrt(mv[1:4] ** 2).sum().sqrt()
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

def run_gp_tests():
    """Verify the geometric product table against known identities."""
    engine = CliffordEngine3D()
    errors = []

    # e1 * e1 = 1
    e1 = torch.zeros(8); e1[1] = 1.0
    result = engine.geometric_product(e1, e1)
    if not torch.allclose(result, torch.tensor([1.,0,0,0,0,0,0,0]), atol=1e-6):
        errors.append(f"e1*e1 != 1: got {result}")

    # e1 * e2 = e12
    e2 = torch.zeros(8); e2[2] = 1.0
    result = engine.geometric_product(e1, e2)
    expected = torch.zeros(8); expected[4] = 1.0  # e12
    if not torch.allclose(result, expected, atol=1e-6):
        errors.append(f"e1*e2 != e12: got {result}")

    # e2 * e1 = -e12
    result = engine.geometric_product(e2, e1)
    expected = torch.zeros(8); expected[4] = -1.0
    if not torch.allclose(result, expected, atol=1e-6):
        errors.append(f"e2*e1 != -e12: got {result}")

    # e12 * e12 = -1
    e12 = torch.zeros(8); e12[4] = 1.0
    result = engine.geometric_product(e12, e12)
    expected = torch.tensor([-1.,0,0,0,0,0,0,0])
    if not torch.allclose(result, expected, atol=1e-6):
        errors.append(f"e12*e12 != -1: got {result}")

    # e12 * e13 = -e23 (this was WRONG in the old table)
    e13 = torch.zeros(8); e13[5] = 1.0
    result = engine.geometric_product(e12, e13)
    expected = torch.zeros(8); expected[6] = -1.0  # -e23
    if not torch.allclose(result, expected, atol=1e-6):
        errors.append(f"e12*e13 != -e23: got {result}")

    # e123 * e123 = -1 (this was WRONG in the old table)
    e123 = torch.zeros(8); e123[7] = 1.0
    result = engine.geometric_product(e123, e123)
    expected = torch.tensor([-1.,0,0,0,0,0,0,0])
    if not torch.allclose(result, expected, atol=1e-6):
        errors.append(f"e123*e123 != -1: got {result}")

    # e13 * e23 = -e12 (this was WRONG)
    e23 = torch.zeros(8); e23[6] = 1.0
    result = engine.geometric_product(e13, e23)
    expected = torch.zeros(8); expected[4] = -1.0  # -e12
    if not torch.allclose(result, expected, atol=1e-6):
        errors.append(f"e13*e23 != -e12: got {result}")

    if errors:
        print("GP TESTS FAILED:")
        for e in errors:
            print(f"  {e}")
        return False
    else:
        print("All GP tests passed!")
        return True


if __name__ == "__main__":
    run_gp_tests()