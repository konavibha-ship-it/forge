"""
Phase 3: A small transformer, trained from scratch using PyTorch.

This is a genuine (tiny) GPT: token + position embeddings, multi-head
self-attention with a causal mask (so it can only look at past tokens,
never future ones), feed-forward blocks, and a final linear layer that
predicts the next character. Same architecture family as real GPT models
— just far smaller (thousands of parameters instead of billions).

Unlike Phase 2 (where you built the autograd engine by hand), here we use
PyTorch's autograd — the hand-built version proved you understand what's
happening underneath; now we use the real, fast, production tool.

Install PyTorch first (one-time, free, CPU version is enough for this):
    pip install torch

This script trains a character-level model on a small embedded text
sample, then generates new text from it. Don't expect Shakespeare — with
this little data and this small a model, the output will be crude. The
point is watching a REAL transformer, trained from nothing, produce output
that has any structure at all (real words, plausible letter patterns).
"""

import torch
import torch.nn as nn
from torch.nn import functional as F

# ---------------------------------------------------------------------------
# Hyperparameters — kept small on purpose so this trains in a minute or two
# on a normal CPU, no GPU required.
# ---------------------------------------------------------------------------
BLOCK_SIZE = 16      # how many characters of context the model sees at once
BATCH_SIZE = 16      # how many sequences we train on per step
N_EMBD = 32          # size of each token's embedding vector
N_HEAD = 4           # number of attention heads
N_LAYER = 2          # number of transformer blocks stacked
LEARNING_RATE = 1e-3
MAX_ITERS = 500
EVAL_INTERVAL = 100

torch.manual_seed(42)

# ---------------------------------------------------------------------------
# Tiny embedded training text. Replace this with a real downloaded book
# (see the "what to try next" notes at the bottom) once this runs.
# ---------------------------------------------------------------------------
TEXT = """
the quick brown fox jumps over the lazy dog.
the dog barks at the fox. the fox runs away quickly.
a wise old owl lived in an oak. the more he saw, the less he spoke.
the sun rises in the east and sets in the west.
practice and patience turn small steps into real progress.
""" * 20  # repeat so there's enough data for the model to find patterns


class Head(nn.Module):
    """One self-attention head."""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(N_EMBD, head_size, bias=False)
        self.query = nn.Linear(N_EMBD, head_size, bias=False)
        self.value = nn.Linear(N_EMBD, head_size, bias=False)
        # causal mask: token i can only attend to tokens <= i
        self.register_buffer("tril", torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        # attention scores ("how much should each token look at each other token")
        wei = q @ k.transpose(-2, -1) * (C ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        v = self.value(x)
        return wei @ v


class MultiHeadAttention(nn.Module):
    """Several attention heads running in parallel, results concatenated."""

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(N_EMBD, N_EMBD)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.proj(out)


class FeedForward(nn.Module):
    """A simple two-layer MLP applied to each token independently."""

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """One transformer block: attention, then feed-forward, each with a
    residual connection and layer norm (this is the standard "pre-norm"
    transformer block design)."""

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, N_EMBD)
        self.position_embedding = nn.Embedding(BLOCK_SIZE, N_EMBD)
        self.blocks = nn.Sequential(*[Block(N_EMBD, N_HEAD) for _ in range(N_LAYER)])
        self.ln_f = nn.LayerNorm(N_EMBD)
        self.lm_head = nn.Linear(N_EMBD, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding(idx)                                  # (B, T, C)
        pos_emb = self.position_embedding(torch.arange(T))                   # (T, C)
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)                                             # (B, T, vocab_size)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -BLOCK_SIZE:]           # only keep last BLOCK_SIZE tokens
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]                 # focus on the last time step
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


def get_batch(data):
    ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([data[i:i + BLOCK_SIZE] for i in ix])
    y = torch.stack([data[i + 1:i + BLOCK_SIZE + 1] for i in ix])
    return x, y


if __name__ == "__main__":
    # --- build vocab and encode the text (character-level, like Phase 1) ---
    chars = sorted(set(TEXT))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    def encode(s):
        return [stoi[c] for c in s]

    def decode(ids):
        return "".join(itos[i] for i in ids)

    data = torch.tensor(encode(TEXT), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data, val_data = data[:n], data[n:]

    print(f"vocab size: {vocab_size}")
    print(f"training characters: {len(train_data)}, validation: {len(val_data)}\n")

    model = TinyGPT(vocab_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"model parameters: {num_params:,}\n")

    for step in range(MAX_ITERS):
        xb, yb = get_batch(train_data)
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % EVAL_INTERVAL == 0 or step == MAX_ITERS - 1:
            with torch.no_grad():
                xv, yv = get_batch(val_data)
                _, val_loss = model(xv, yv)
            print(f"step {step:4d}  train loss {loss.item():.4f}  val loss {val_loss.item():.4f}")

    print("\nGenerating text from the trained model:")
    context = torch.zeros((1, 1), dtype=torch.long)  # start from a single blank token
    generated = model.generate(context, max_new_tokens=200)[0].tolist()
    print(decode(generated))

    print("\nTraining complete. Loss going down and output showing real words")
    print("(even if grammar is rough) means the transformer is genuinely learning")
    print("language structure from raw text — the same mechanism behind GPT.")