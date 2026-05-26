# GA Tokenizer Design Note

## Motivation

Standard BPE-style tokenizers are heavily biased toward languages dominant in their training corpora (usually English). This leads to significantly worse tokenization efficiency for other languages — especially Southeast Asian and logographic scripts — resulting in higher inference cost and degraded model performance.

**Goal**: Build a tokenizer that reduces cross-language bias using Geometric Algebra primitives (rotors, multivectors, grade-aware operations).

## Core Idea

Instead of purely statistical merge decisions (frequency-based BPE), use **rotor-guided merging**:

- Characters are embedded as multivectors.
- Potential merges are scored using the geometric relationship (rotor) between adjacent characters.
- Grade-aware scoring favors merges that produce strong bivector components (rotation planes).

This shifts tokenization from "how often do these tokens appear together?" toward "how geometrically coherent is this combination?"

## Architecture (v9)

- **Base algebra**: Cl(3,0)
- **Embedding**:
  - Alphabetic scripts: Simple vector embedding
  - Chinese: Richer embedding (vector + bivector components)
- **Merging**: Rotor-guided + grade-aware scoring (boost bivector, penalize trivector)
- **Merging scope**: Applied across all scripts (including Chinese)

## Current Results (49 sentences × 3 languages)

| Language    | Fertility | tokens/char | Ratio vs English |
|-------------|-----------|-------------|------------------|
| English     | 3.768     | 0.652       | 1.00×            |
| Indonesian  | 4.158     | 0.625       | **1.10×**        |
| Chinese     | 8.286     | 0.611       | **2.20×**        |

- **English ↔ Indonesian** parity is strong.
- **Chinese** remains significantly worse (logographic challenge).

**Average deviation from perfect parity**: 0.765

## Key Learnings

1. Rotor exponentiation + accurate geometric product gave the largest single improvement.
2. Grade-aware scoring (favoring bivectors) further improved fairness.
3. Richer embeddings for Chinese helped, but the fundamental difficulty of logographic scripts remains.
4. Completely disabling merging for Chinese was harmful.

## Limitations

- Chinese (and other CJK) still requires ~2× more tokens than alphabetic scripts.
- Current embeddings are still relatively simple.
- Training data is small (synthetic parallel sentences).

## Next Steps

- Scale training data significantly
- Explore full multivector token embeddings (beyond rotor merging)
- Add byte-level fallback or hybrid strategies for CJK
- Expand to more languages (Thai, Vietnamese, Korean, etc.)
- Compare against strong baselines (Llama-3, Qwen, etc.)

---

*Last updated: May 2026*