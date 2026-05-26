"""
Training script for TokenMultivectorTokenizer
with E + C + B losses (Rotor Consistency + Grade-wise + Reconstruction).
"""

import torch
from collections import defaultdict
from gatoken import TokenMultivectorTokenizer


def build_cooccurrence(texts):
    cooc = defaultdict(int)
    for text in texts:
        tokens = list(text)
        for i in range(len(tokens)):
            for j in range(i+1, min(i+4, len(tokens))):
                cooc[(tokens[i], tokens[j])] += 1
                cooc[(tokens[j], tokens[i])] += 1
    return cooc


def sample_pairs(vocab, cooc, num_pairs=128):
    pairs = []
    labels = []
    targets = []

    vocab_list = list(vocab)

    for _ in range(num_pairs):
        i = torch.randint(0, len(vocab_list), (1,)).item()
        j = torch.randint(0, len(vocab_list), (1,)).item()

        t1, t2 = vocab_list[i], vocab_list[j]
        score = cooc.get((t1, t2), 0)

        label = 1 if score > 0 else 0
        pairs.append((i, j))
        labels.append(label)

        t_i = {
            'scalar': torch.tensor(0.6 if label == 1 else 0.3),
            'bivector': torch.tensor(0.4 if label == 1 else 0.1)
        }
        t_j = {
            'scalar': torch.tensor(0.6 if label == 1 else 0.3),
            'bivector': torch.tensor(0.4 if label == 1 else 0.1)
        }
        targets.append((t_i, t_j))

    return pairs, labels, targets


def main():
    texts = [
        "hello world",
        "saya suka makan nasi goreng",
        "the quick brown fox jumps over the lazy dog",
        "artificial intelligence is transforming many industries",
    ]

    print("Training tokenizer...")
    tokenizer = TokenMultivectorTokenizer(max_vocab_size=500)
    tokenizer.train(texts)
    print(f"Vocab size: {tokenizer.vocab_size}")

    cooc = build_cooccurrence(texts)
    params = [p for p in tokenizer.token_to_mv.values() if isinstance(p, torch.nn.Parameter)]
    optimizer = torch.optim.Adam(params, lr=3e-4)

    engine = tokenizer.engine
    lambda_rotor = 1.0
    lambda_grade = 0.5
    lambda_recon = 0.3

    print("\nTraining with E + C + B objective...\n")

    for step in range(500):
        pairs, labels, targets = sample_pairs(tokenizer.vocab, cooc, 96)

        loss = 0.0
        for (i, j), label, (t_i, t_j) in zip(pairs, labels, targets):
            mv_i = tokenizer.token_to_mv[tokenizer.vocab[i]]
            mv_j = tokenizer.token_to_mv[tokenizer.vocab[j]]

            # E: Rotor Consistency
            rotor = engine.rotor_between(mv_i.data, mv_j.data)
            _, _, biv, tri = engine.grade_norms(rotor)
            if label == 1:
                r_loss = tri + 0.2 * max(0.3 - biv, 0)
            else:
                r_loss = max(biv - 0.2, 0) + 0.1 * tri

            # C: Grade-wise
            g_loss = (mv_i[0] - t_i['scalar'])**2 + (mv_j[0] - t_j['scalar'])**2
            g_loss += (biv - t_i['bivector'])**2 * 0.5

            # B: Reconstruction (for merged tokens)
            recon_loss = 0.0
            if len(tokenizer.vocab[i]) > 1:
                # Try to reconstruct from first two characters
                chars = list(tokenizer.vocab[i])
                if len(chars) >= 2 and chars[0] in tokenizer.token_to_mv and chars[1] in tokenizer.token_to_mv:
                    mv_a = tokenizer.token_to_mv[chars[0]].data
                    mv_b = tokenizer.token_to_mv[chars[1]].data
                    recon = engine.normalize((mv_a + mv_b) / 2 + 0.3 * engine.geometric_product(mv_a, mv_b))
                    recon_loss += torch.norm(mv_i.data - recon)

            loss += lambda_rotor * r_loss + lambda_grade * g_loss + lambda_recon * recon_loss

        loss = loss / len(pairs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            print(f"Step {step:3d} | Loss: {loss:.4f}")

    print("\nTraining finished.")


if __name__ == "__main__":
    main()
