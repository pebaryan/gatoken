# GA Tokenizer Design Note

## Motivation

Standard BPE-style tokenizers are heavily biased toward languages dominant in their training corpora (usually English). This leads to significantly worse tokenization efficiency for other languages — especially Southeast Asian and logographic scripts — resulting in higher inference cost and degraded model performance.

**Goal**: Build a tokenizer that reduces cross-language bias using Geometric Algebra primitives (rotors, multivectors, grade-aware operations).

## Core Idea

Instead of purely statistical merge decisions (frequency-based BPE), use **rotor-guided merging**:

- Characters are embedded as multivectors in Cl(3,0).
- Potential merges are scored using the geometric relationship (rotor) between adjacent characters.
- Grade-aware scoring favors merges that produce strong bivector components (rotation planes).

This shifts tokenization from "how often do these tokens appear together?" toward "how geometrically coherent is this combination?"

## Architecture

### RotorSubwordTokenizer
- **Base algebra**: Cl(3,0) with **correct** geometric product (verified by unit tests)
- **Embedding**: Non-degenerate Unicode codepoint hashing (sin/cos basis functions)
- **Chinese**: Richer embedding (vector + bivector components)
- **Merging**: Rotor-guided + grade-aware scoring for all scripts (including Chinese)

### TokenMultivectorTokenizer
- Extends RotorSubwordTokenizer with **learnable multivector representations**
- Each subword token is an `nn.Parameter` multivector
- Geometric product is fully **differentiable** via matrix-based computation
- Supports training with E + C + B objectives (see below)

### CliffordEngine3D (shared module)
- Correct Cl(3,0) geometric product (15 sign errors fixed from initial version)
- Differentiable via `torch.einsum`
- Rotor exponentiation and grade analysis
- Unit tests verify: e1*e1=1, e1*e2=e12, e12*e13=-e23, e123*e123=-1

## Training Objectives

### E: Rotor Consistency Loss
- Related tokens → clean bivector rotors (low trivector noise)
- Unrelated tokens → discourage clean rotors
- Encourages geometric structure in the multivector space

### C: Grade-wise Prediction Loss
- Scalar component → token frequency/importance
- Bivector norm → relational/syntactic properties
- Encourages each grade to carry meaningful information

### B: Reconstruction Loss
- Merged token multivector should be reconstructible from its components
- Reconstruction via: `normalize((mv_a + mv_b)/2 + 0.3 * GP(mv_a, mv_b))`
- Encourages compositional structure

## Current Results (49 sentences × 3 languages)

| Language    | Fertility | tokens/char | Ratio vs English |
|-------------|-----------|-------------|------------------|
| English     | 3.295     | 0.570       | 1.00×            |
| Indonesian  | 3.820     | 0.574       | **1.16×**        |
| Chinese     | 7.490     | 0.553       | **2.27×**        |

Average deviation from perfect parity: 0.798

### Comparison with GPT-2

| Tokenizer                        | Fertility Ratio (id/en) | Parity Score |
|----------------------------------|--------------------------|--------------|
| GPT-2                            | 2.32×                    | 0.43         |
| RotorSubwordTokenizer            | 1.16×                    | 0.80         |

## Key Learnings

1. **15 sign errors in original GP table** — fixed and verified with unit tests.
2. Rotor exponentiation + accurate geometric product gave the largest single improvement.
3. Grade-aware scoring (favoring bivectors) further improved fairness.
4. `ord(char) % 3` embedding was degenerate (only 3 dimensions for thousands of characters) — replaced with sin/cos hashing.
5. Chinese merging should not be skipped entirely — it performs best when included.
6. Training loop using differentiable GP now actually propagates gradients.

## Limitations

- Chinese (and other CJK) still requires ~2.2× more tokens than alphabetic scripts.
- Training data is small (synthetic parallel sentences).
- Iterative BPE-style merging (not single-pass) would improve vocabulary quality.

## Next Steps

- Implement iterative BPE-style merging (recompute frequencies after each merge)
- Compare against strong baselines (Llama-3, Qwen, etc.)
- Add byte-level fallback for truly low-resource scripts
- Expand to more languages (Thai, Vietnamese, Korean, etc.)
- Fine-tune multivector embeddings on real data (not just synthetic text)

---

*Last updated: May 2026*