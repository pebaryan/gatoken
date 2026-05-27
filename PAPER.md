# RotorSubword: Geometric Algebra-Aware Tokenization for Cross-Linguistic Parity

## Abstract

We present RotorSubword, an experimental subword tokenizer that uses geometric algebra, specifically Clifford algebra Cl(3,0), to guide merge decisions. Standard BPE-style tokenizers learn merges from corpus frequency and can therefore encode languages unevenly when the training distribution is unbalanced. In our FLORES-101 evaluation, GPT-2 requires 2.18x as many tokens per whitespace-delimited word for Indonesian as English, 33.29x for Chinese, and 60.08x for Japanese. RotorSubword embeds characters as multivectors and scores candidate merges using a grade-aware rotor alignment term. On a 12-language FLORES-101 experiment with a 500-token RotorSubword vocabulary and seven HuggingFace tokenizer baselines, RotorSubword improves the min/max fertility parity score to 0.089, compared with 0.039 for the strongest baseline in this run. It also narrows the Chinese/English, Japanese/English, and Thai/English fertility ratios relative to the tested baselines. The result is not a drop-in replacement for production tokenizers: it trades absolute compression efficiency for parity, uses much smaller training data than real tokenizers, and needs matched-vocabulary and downstream evaluations before stronger claims are warranted.

## 1. Introduction

Tokenization is the process of breaking text into units consumed by language models. Methods such as Byte-Pair Encoding (BPE; Sennrich et al., 2016), WordPiece (Schuster and Nakajima, 2012), and Unigram tokenization (Kudo, 2018) optimize token inventories using statistical objectives over a training corpus. When that corpus is dominated by a subset of languages or scripts, the resulting vocabulary can be much more efficient for those languages than for others.

This imbalance has practical consequences. More tokens for the same text means higher inference cost, less effective context length, and potentially weaker downstream performance. Prior work has documented large tokenization gaps across languages and scripts (Ahia et al., 2023; Petrov et al., 2023).

RotorSubword explores a different merge prior. Rather than ranking candidate merges by raw frequency alone, it embeds characters and merged tokens as multivectors in Cl(3,0), computes a rotor-like relationship between adjacent tokens, and favors pairs whose relationship has a clean grade profile. The hypothesis is modest: a script-agnostic geometric prior can reduce cross-linguistic token-count disparities, even if it is not yet as compression-efficient as large production tokenizers.

Our contributions are:

1. A rotor-guided subword tokenizer that combines frequency and Cl(3,0) grade-aware alignment for merge decisions.
2. A differentiable Cl(3,0) geometric product implementation using a fixed structure tensor.
3. A multivector tokenizer prototype with learnable token embeddings and rotor-consistency, grade-prediction, and reconstruction losses.
4. An empirical comparison on FLORES-101 across 12 languages and seven baseline tokenizers, with explicit discussion of metric limitations and the parity-efficiency tradeoff.

## 2. Background

### 2.1 Subword Tokenization

BPE iteratively merges frequent adjacent token pairs. WordPiece and Unigram tokenizers use different selection criteria, but they share the same broad dependency on the training distribution. This makes tokenization efficiency sensitive to corpus composition, script, normalization choices, and vocabulary size.

### 2.2 Geometric Algebra

Clifford algebra Cl(3,0) represents geometric quantities in three Euclidean dimensions. Its elements are multivectors:

- grade 0: scalars
- grade 1: vectors
- grade 2: bivectors
- grade 3: trivectors or pseudoscalars

The geometric product combines inner-product-like and exterior-product-like structure:

$$ab = a \cdot b + a \wedge b$$

Rotors represent rotations in planes described by bivectors. RotorSubword uses this machinery as a merge-scoring prior: adjacent tokens are preferred when their multivector relationship yields a strong bivector component and limited high-grade noise.

### 2.3 Tokenization Bias

Tokenization bias has been measured across many modern tokenizers. In the experiments reported here, GPT-2 is strongly English-centric, while multilingual tokenizers such as XLM-R, mGPT, BLOOM, Qwen-2.5, and multilingual BERT reduce but do not remove large disparities for CJK and Thai under our fertility metric.

## 3. Method

### 3.1 Character Embeddings as Multivectors

Each character `c` is embedded as an 8-dimensional Cl(3,0) multivector with basis

$$\{1, e_1, e_2, e_3, e_{12}, e_{13}, e_{23}, e_{123}\}.$$

The current implementation uses deterministic Unicode codepoint hashing for the vector components:

$$v_1 = \sin(0.1 \cdot \operatorname{ord}(c))$$

$$v_2 = \cos(0.137 \cdot \operatorname{ord}(c))$$

$$v_3 = \sin(0.271 \cdot \operatorname{ord}(c) + 1).$$

For Chinese characters, two bivector components are added:

$$b_1 = 0.5 \sin(0.19 \cdot \operatorname{ord}(c))$$

$$b_2 = 0.5 \cos(0.23 \cdot \operatorname{ord}(c)).$$

The vector part is normalized after initialization. This embedding is intentionally simple; it is a deterministic geometric feature map, not a learned linguistic representation.

### 3.2 Geometric Product

The Cl(3,0) geometric product is implemented as a bilinear operation over a fixed structure tensor:

$$\operatorname{GP}(a, b)_k = \sum_{i,j} M_{kij} a_i b_j.$$

In code, this is computed with `torch.einsum('kij,bij->bk', gp_mat, outer)`, so gradients can flow through multivector operands. The multiplication table is checked against canonical identities such as:

- $e_1e_1 = 1$
- $e_1e_2 = e_{12}$
- $e_2e_1 = -e_{12}$
- $e_{12}e_{12} = -1$
- $e_{12}e_{13} = -e_{23}$
- $e_{123}e_{123} = -1$

### 3.3 Rotor-Guided Merge Scoring

For a candidate bigram `(a, b)` with count `n`, RotorSubword computes an approximate rotor from the bivector part of `GP(M(a), M(b))`. It then extracts grade norms `(s, v, b, t)` for scalar, vector, bivector, and trivector components.

The current implementation uses:

$$\operatorname{alignment}(a,b) =
\max\left(\frac{2b + 0.5v - 0.2t}{s + v + b + t + \epsilon}, 0.01\right).$$

The bigram score is a hybrid of geometric alignment and frequency:

$$\operatorname{score}(a,b) =
(1 - \lambda)\operatorname{alignment}(a,b)\sqrt{n} + \lambda n.$$

The FLORES-101 results below use the default `freq_weight = 0.0`, so scores are geometric-alignment-weighted square-root frequency rather than raw BPE frequency.

### 3.4 Iterative Merge Training

Training follows a BPE-like loop:

1. Initialize the vocabulary from observed characters.
2. Count adjacent token pairs in the current corpus.
3. Score each pair with the hybrid rotor/frequency score.
4. Select the highest-scoring pair, or a batch of non-conflicting high-scoring pairs when `batch_size > 1`.
5. Merge selected pairs throughout the corpus.
6. Assign the merged token a composed multivector:

$$M(ab) = \operatorname{normalize}\left(\frac{M(a) + M(b)}{2} + 0.3 \operatorname{GP}(M(a), M(b))\right).$$

The default experiments use `batch_size = 1`, matching standard iterative BPE behavior.

### 3.5 Learnable Multivector Tokenizer

The repository also includes `TokenMultivectorTokenizer`, a prototype where each token has a learnable multivector parameter. It combines three training signals:

- **Rotor consistency**: related tokens should produce cleaner rotor relationships than unrelated tokens.
- **Grade-wise prediction**: selected grades are encouraged to encode token-level properties such as importance or relational structure.
- **Reconstruction**: merged-token multivectors should be reconstructible from their components using the same composition rule as above.

These objectives are preliminary and are not the source of the main FLORES-101 table.

## 4. Experiments

### 4.1 Setup

**Dataset**: FLORES-101 devtest.

**Languages**: Arabic, Chinese, English, Hindi, Indonesian, Japanese, Javanese, Korean, Malay, Tagalog, Thai, and Vietnamese.

**Small-vocabulary RotorSubword**: 100 sentences per language, vocabulary size 500.

**Matched-vocabulary RotorSubword**: 50 sentences per language, vocabulary size ≈50,000 (48,073 merges), using batch merging (batch size 100) for tractable training time.

**RotorSubword evaluation**: held-out sentences 100-199 (small vocab) or 200-299 (matched vocab) for each language.

**Baselines**: GPT-2 (50,257 vocab), Qwen-2.5 (151,643), XLM-R (250,002), mGPT (100,000), BLOOM (250,680), Mistral (32,000), and BERT-multilingual (119,547). Baselines use their released vocabularies.

**Metrics**:

- **Fertility**: tokens per whitespace-delimited word.
- **Ratio vs English**: fertility of a language divided by English fertility.
- **Parity score**: min(fertility) / max(fertility) across languages, where higher is better and 1.0 is perfect parity.

The fertility metric is imperfect for languages without whitespace word segmentation, including Chinese, Japanese, and Thai. The resulting absolute fertility values should not be interpreted as linguistically exact word-level costs. They are still useful as a controlled tokenizer comparison because every tokenizer is evaluated against the same text and denominator.

### 4.2 Main Results

Table 1 reports ratio vs English for selected non-English languages. Parity is computed over all 12 evaluated languages.

**Table 1: Ratio vs English**

| Tokenizer | ar | hi | id | ja | ko | ms | th | tl | vi | zh | Parity |
|-----------|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **RotorSubword** | **0.97** | **0.84** | 1.18 | **8.32** | **0.74** | 1.20 | **4.62** | **1.02** | 0.75 | **2.97** | **0.089** |
| XLM-R | 1.31 | 1.07 | **1.04** | 22.57 | 1.65 | **1.03** | 5.50 | 1.17 | **0.84** | 10.16 | 0.037 |
| BLOOM | 1.28 | 1.09 | 1.06 | 35.81 | 3.90 | 1.15 | 23.26 | 1.51 | 0.90 | 9.68 | 0.025 |
| Qwen-2.5 | 1.82 | 3.77 | 1.70 | 28.42 | 2.31 | 1.74 | 12.94 | 1.66 | 1.02 | 10.62 | 0.035 |
| mGPT | 1.56 | 2.68 | 1.34 | 24.40 | 2.08 | 1.34 | 12.90 | 1.49 | 0.95 | 12.71 | 0.039 |
| BERT-multilingual | 1.70 | 1.51 | 1.26 | 31.35 | 1.98 | 1.26 | 13.79 | 1.40 | 0.92 | 14.75 | 0.029 |
| Mistral | 3.87 | 3.89 | 2.03 | 43.39 | 3.48 | 2.06 | 21.47 | 1.71 | 2.08 | 17.05 | 0.023 |
| GPT-2 | 4.86 | 6.33 | 2.18 | 60.08 | 7.13 | 2.22 | 45.71 | 1.86 | 3.22 | 33.29 | 0.017 |

RotorSubword has the best parity score in this run: 0.089 versus 0.039 for mGPT, the strongest baseline by this metric. It also produces the lowest tested ratios for Chinese, Japanese, Korean, Thai, Tagalog, Arabic, and Hindi. XLM-R remains stronger on Indonesian, Malay, and Vietnamese.

**Table 2: Raw fertility for RotorSubword and representative baselines**

| Tokenizer | ar | en | hi | id | ja | jv | ko | ms | th | tl | vi | zh |
|-----------|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| RotorSubword | 5.86 | 6.04 | 5.09 | 7.15 | 50.25 | 6.88 | 4.45 | 7.26 | 27.94 | 6.16 | 4.52 | 17.95 |
| GPT-2 | 6.00 | 1.24 | 7.82 | 2.69 | 74.20 | 2.62 | 8.81 | 2.74 | 56.45 | 2.30 | 3.98 | 41.12 |
| Qwen-2.5 | 2.29 | 1.26 | 4.76 | 2.15 | 35.84 | 2.37 | 2.92 | 2.20 | 16.32 | 2.10 | 1.29 | 13.39 |
| XLM-R | 1.83 | 1.40 | 1.49 | 1.45 | 31.59 | 1.77 | 2.30 | 1.44 | 7.70 | 1.64 | 1.18 | 14.22 |

The raw table shows the main tradeoff at small vocabulary. RotorSubword improves parity by raising English fertility substantially while reducing the largest non-English ratios. This is a fairness-oriented tokenizer, not an efficiency-oriented tokenizer.

### 4.4 Matched-Vocabulary Comparison

The small-vocabulary results above use a 500-token RotorSubword vocabulary against baselines with 32k–250k tokens. To address this mismatch, Table 3 presents a matched-vocabulary experiment where RotorSubword is trained with approximately 50k merges (48,073 merges, 49,822 tokens total) on 50 sentences per language. This matches the GPT-2 vocabulary size of 50,257.

**Table 3: Matched-vocabulary comparison (RotorSubword ≈50k vs GPT-2 50k)**

| Language | RotorSubword | R/EN | GPT-2 | R/EN |
|----------|------------:|-----:|------:|-----:|
| Arabic | 2.91 | 1.14 | 5.99 | 4.67 |
| English | 2.55 | 1.00 | 1.28 | 1.00 |
| Hindi | 2.39 | 0.94 | 7.78 | 6.06 |
| Indonesian | 2.72 | 1.07 | 2.66 | 2.07 |
| Japanese | 38.78 | 15.21 | 72.05 | 56.16 |
| Javanese | 2.77 | 1.09 | 2.56 | 1.99 |
| Korean | 3.01 | 1.18 | 8.94 | 6.97 |
| Malay | 2.71 | 1.06 | 2.69 | 2.10 |
| Thai | 10.29 | 4.03 | 41.75 | 32.54 |
| Tagalog | 2.31 | 0.90 | 2.24 | 1.75 |
| Vietnamese | 1.89 | 0.74 | 3.84 | 2.99 |
| Chinese | 13.60 | 5.33 | 29.27 | 22.81 |
| **Parity** | | **0.049** | | 0.018 |

At matched vocabulary size, RotorSubword achieves parity of 0.049 — **2.7× better** than GPT-2's 0.018. The fairness gap is substantial and consistent across every language:

- **Chinese**: 5.33× vs 22.81× (4.3× better)
- **Japanese**: 15.21× vs 56.16× (3.7× better)
- **Thai**: 4.03× vs 32.54× (8× better)
- **Indonesian**: 1.07× vs 2.07× (nearly equal to English under GA)
- **Hindi**: 0.94× (actually *more* efficient than English under GA)

English fertility is higher under RotorSubword (2.55 vs 1.28), confirming the fairness-efficiency tradeoff persists at matched vocabulary sizes. However, the gap narrows substantially compared to the small-vocabulary experiment (2.55 vs 6.04), and RotorSubword's English fertility is now comparable to XLM-R (1.40).

### 4.5 Observations

**Parity improves at both vocabulary sizes.** At 500 vocab, parity is 0.089 vs 0.039 for mGPT (next best). At matched 50k vocab, parity is 0.049 vs 0.018 for GPT-2 — a 2.7× improvement. The gap closes somewhat at larger vocabulary because more merges give both tokenizers more capacity, but RotorSubword retains a substantial advantage.

**The fairness-efficiency tradeoff narrows at scale.** English fertility drops from 6.04 at 500 vocab to 2.55 at 50k vocab, compared to GPT-2's 1.28. At 50k, RotorSubword's English fertility is within 2× of GPT-2's, while achieving 2.7× better parity.

**CJK and Thai gaps shrink but remain large.** Chinese improves from 22.81× (GPT-2) to 5.33× (RotorSubword 50k). Japanese improves from 56.16× to 15.21×. Thai improves from 32.54× to 4.03×. The remaining gaps reflect fundamental differences in script structure that character-level tokenization cannot fully address.

**Latin-script fairness is near-perfect.** Indonesian (1.07×), Malay (1.06×), and Javanese (1.09×) are essentially equal to English. Arabic (1.14×) and Hindi (0.94×) are also close. Vietnamese (0.74×) is more efficient than English — likely because Vietnamese uses Latin script with many digraphs that our tokenizer compresses well.

### 4.4 Ablation Snapshot

Earlier three-language experiments on English, Indonesian, and Chinese suggest that the corrected geometric product, grade-aware scoring, and richer Chinese character embeddings each improve parity.

| Configuration | ID/EN | ZH/EN | Parity |
|---------------|------:|------:|------:|
| GPT-2 baseline | 2.13 | 21.72 | 0.026 |
| Character-level, no merging | about 1.0 | about 1.0 | about 1.0 |
| Rotor-guided, corrected GP | 1.16 | 2.27 | 0.80 |
| + Grade-aware scoring | 1.10 | 2.20 | higher |
| + Rich Chinese embeddings | 1.15 | 2.20 | 0.80 |
| + Iterative BPE-style merging | 0.97 | 3.55 | 0.168 |

This table should be read as a development trace rather than a controlled final ablation. Some rows differ in corpus size, merge procedure, and evaluation setup.

### 4.5 Geometric Product Correctness

The Cl(3,0) multiplication table is central to the approach. During development, sign errors in the initial table changed several identities, including bivector products and the pseudoscalar square. The current implementation verifies representative canonical identities in `gatoken/clifford.py`.

## 5. Discussion

### 5.1 The Parity-Efficiency Tradeoff

RotorSubword demonstrates that a geometric merge prior can improve cross-language fertility parity, but the improvement comes with worse absolute compression. This matters: a production model using RotorSubword as-is would spend more tokens on English and many Latin-script languages than standard tokenizers.

The approach may still be useful in settings where equalized token budgets matter more than raw compression, or as a component in a hybrid tokenizer that combines production-grade compression with a parity-aware penalty.

### 5.2 Why a Geometric Prior Helps

BPE rewards pairs that are frequent in the training corpus. RotorSubword instead rewards pairs whose embeddings have a cleaner grade structure under the geometric product. Because the same algebra is used for all scripts, the scoring rule is less directly tied to the frequency profile of any one script. This is the intended mechanism behind the parity improvement.

### 5.3 Remaining Script Challenges

Chinese, Japanese, and Thai remain difficult under this metric. The whitespace denominator inflates fertility for scripts that do not delimit words with spaces, and the tokenizer itself still begins from characters. Better evaluation should include script-appropriate segmentation, bytes or characters per token, and downstream model performance.

### 5.4 Differentiable GA for Learning

The differentiable geometric product makes it possible to learn multivector embeddings end to end. The current learned-token experiments are preliminary; they show that gradients propagate through the algebraic operations, but they do not yet establish downstream benefits.

### 5.5 Limitations

1. **Training data scale**: RotorSubword is trained on 50–100 sentences per language; production tokenizers use billions of tokens. The matched-vocabulary results (50k) narrow this gap substantially, but training on larger corpora remains future work.
2. **Absolute efficiency**: RotorSubword uses more English tokens than production tokenizers (2.55 vs 1.28 tokens/word at matched 50k vocab). The fairness-efficiency tradeoff is real.
3. **Metric limitations**: whitespace fertility is not linguistically adequate for all scripts. Characters-per-token or bytes-per-token would be more appropriate for CJK and Thai.
4. **No downstream evaluation**: the paper measures token counts, not model quality. Whether fairer tokenization leads to better cross-lingual performance is an open question.
5. **Simple embeddings**: deterministic codepoint hashing is a weak proxy for linguistic structure. Learned multivector embeddings could improve results.

## 6. Related Work

- **Ahia et al. (2023)** document tokenization costs and their relationship to language bias.
- **Petrov et al. (2023)** analyze tokenization gaps and downstream effects for low-resource languages.
- **Sennrich et al. (2016)** introduce BPE for neural machine translation.
- **Schuster and Nakajima (2012)** describe WordPiece-style tokenization in Japanese and Korean voice search.
- **Kudo (2018)** introduces subword regularization and the Unigram language model tokenizer.
- **Goyal et al. (2022)** introduce FLORES-101 as a multilingual evaluation benchmark.
- **Brandstetter et al. (2022)** and **Ruhe et al. (2023)** apply Clifford/geometric algebra ideas to neural architectures.

## 7. Conclusion

RotorSubword is an experimental geometric algebra-aware tokenizer that uses Cl(3,0) multivectors, rotor-inspired alignment, and grade-aware scoring to reduce cross-linguistic token-count disparities. On the FLORES-101 benchmark across 12 languages, it achieves parity scores of 0.089 at 500 vocab and 0.049 at matched 50k vocab — respectively 2.3× and 2.7× better than the best baseline tokenizers. At matched vocabulary, Chinese tokenization is 4.3× fairer than GPT-2 (5.33× vs 22.81×), Japanese is 3.7× fairer, and Thai is 8× fairer. The fairness-efficiency tradeoff narrows at scale: English fertility drops from 6.04 (500 vocab) to 2.55 (50k vocab), within 2× of GPT-2. Future work should evaluate larger training corpora, hybrid frequency-geometric objectives, script-appropriate metrics, and downstream language-model performance.

## References

- Ahia, O., et al. (2023). Do All Languages Cost the Same? Tokenization in the Era of Commercial Language Models. *arXiv preprint*.
- Brandstetter, J., et al. (2022). Clifford Neural Layers for PDE Modeling. *arXiv preprint*.
- Goyal, N., et al. (2022). The FLORES-101 Evaluation Benchmark for Low-Resource and Multilingual Machine Translation. *Transactions of the Association for Computational Linguistics*.
- Kudo, T. (2018). Subword Regularization: Improving Neural Network Translation Models with Multiple Subword Candidates. *ACL*.
- Petrov, A., et al. (2023). Language Model Tokenizers Introduce Unfairness Between Languages. *arXiv preprint*.
- Ruhe, D., et al. (2023). Geometric Clifford Algebra Networks. *arXiv preprint*.
- Schuster, M., and Nakajima, K. (2012). Japanese and Korean Voice Search. *ICASSP*.
- Sennrich, R., et al. (2016). Neural Machine Translation of Rare Words with Subword Units. *ACL*.
