# gatoken

Exploration of **Geometric Algebra-aware tokenizers** with focus on reducing language bias (starting with Indonesian vs English).

## Goals

- Measure language bias in tokenization (SEA languages)
- Provide a clean interface for GA-based tokenizers
- Start simple, then experiment with rotor-based / multivector approaches

## Structure

```
gatoken/
├── gatoken/
│   ├── ga_interface.py     # GATokenizer abstract class + StandardTokenizer
│   └── metrics.py          # Fertility, tokens/char, parity metrics
├── scripts/
│   └── eval_bias.py        # Indonesian vs English evaluation
└── data/                   # Test sets
```

## Quick Start

```bash
cd gatoken
pip install transformers

python scripts/eval_bias.py --tokenizer gpt2
```

## GA Tokenizer Interface

Any new GA tokenizer should inherit from `GATokenizer`:

```python
from gatoken import GATokenizer

class MyGATokenizer(GATokenizer):
    def encode(self, text): ...
    def decode(self, ids): ...
    def tokenize(self, text): ...
    @property
    def vocab_size(self): ...
```

## Next Steps

- Add more SEA languages (Malay, Tagalog, Thai, Vietnamese)
- Implement first GA tokenizer (rotor-based merging)
- Expand test sets with parallel sentences
