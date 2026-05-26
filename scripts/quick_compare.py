"""Quick comparison: GA vs GPT-2 on full corpus."""
from gatoken import RotorSubwordTokenizer, StandardTokenizer, compute_metrics
from scripts.corpus import get_all_corpora
from transformers import AutoTokenizer

corpora = get_all_corpora()
all_texts = []
for lang, texts in corpora.items():
    all_texts.extend(texts[:5])

# GA tokenizer
tok = RotorSubwordTokenizer(max_vocab_size=200)
tok.train(all_texts)

# GPT-2 baseline
gpt2 = AutoTokenizer.from_pretrained('gpt2')
gpt2_tok = StandardTokenizer(gpt2)

en_ga = compute_metrics(tok, corpora['en'], 'en')
en_gpt = compute_metrics(gpt2_tok, corpora['en'], 'en')

header = f"{'Lang':<6} {'GA_Fert':>8} {'GA_R/EN':>8}    {'GPT2_Fert':>10} {'GPT2_R/EN':>10}"
print(header)
print("-" * len(header))

for lang in ['en', 'id', 'zh', 'ms', 'vi']:
    ga_m = compute_metrics(tok, corpora[lang], lang)
    gpt_m = compute_metrics(gpt2_tok, corpora[lang], lang)
    ga_ratio = ga_m.fertility / en_ga.fertility if en_ga.fertility > 0 else 0
    gpt_ratio = gpt_m.fertility / en_gpt.fertility if en_gpt.fertility > 0 else 0
    print(f"{lang:<6} {ga_m.fertility:>8.2f} {ga_ratio:>8.2f}    {gpt_m.fertility:>10.2f} {gpt_ratio:>10.2f}")

# Parity
ferts_ga = [compute_metrics(tok, corpora[l], l).fertility for l in ['en','id','zh','ms','vi']]
ferts_gpt = [compute_metrics(gpt2_tok, corpora[l], l).fertility for l in ['en','id','zh','ms','vi']]
print(f"\nGA parity: {min(ferts_ga)/max(ferts_ga):.3f}")
print(f"GPT-2 parity: {min(ferts_gpt)/max(ferts_gpt):.3f}")