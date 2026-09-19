"""
Scaled-up transformer training on a REAL dataset (Tiny Shakespeare, ~1MB),
with a bigger model than Phase 3's toy version, checkpoint saving, and
automatic GPU use when available (works on CPU too, just slower).

Run locally (CPU, works but slower — expect several minutes):
    python data/download_data.py     # one-time, downloads the dataset
    python model/train_large.py

Run on a free GPU (much faster — recommended for this step):
    1. Go to kaggle.com -> Create -> New Notebook
    2. In the notebook settings (right sidebar), set Accelerator to GPU
    3. Upload this file, architecture.py, and shakespeare.txt as notebook
       inputs/files (or paste this script's contents into a cell)
    4. Run it — training will be noticeably faster on GPU
    5. Download the resulting checkpoint.pt from the notebook's output
       files back to your D:\\forge\\model\\ folder

Either way, the output is model/checkpoint.pt — your trained model, saved
to disk, ready to load later without retraining (see generate.py).
"""

import os
import time
import torch
from architecture import GPT

# ---------------------------------------------------------------------------
# Hyperparameters — meaningfully bigger than Phase 3's toy model, but still
# modest enough to finish in a reasonable time on CPU, and quickly on GPU.
# ---------------------------------------------------------------------------
BLOCK_SIZE = 128
BATCH_SIZE = 64
N_EMBD = 192
N_HEAD = 6
N_LAYER = 6
DROPOUT = 0.1
LEARNING_RATE = 3e-4
MAX_ITERS = 3000
EVAL_INTERVAL = 300
EVAL_ITERS = 50  # batches averaged per evaluation, for a less noisy loss reading

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "shakespeare.txt")
CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "checkpoint.pt")

torch.manual_seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"


def get_batch(data, block_size, batch_size, device):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, train_data, val_data, block_size, batch_size, eval_iters, device):
    """Average loss over several batches, for a more stable reading than
    a single batch's loss (which is naturally noisy)."""
    model.eval()
    out = {}
    for split_name, data in (("train", train_data), ("val", val_data)):
        losses = torch.zeros(eval_iters)
        for i in range(eval_iters):
            xb, yb = get_batch(data, block_size, batch_size, device)
            _, loss = model(xb, yb)
            losses[i] = loss.item()
        out[split_name] = losses.mean().item()
    model.train()
    return out


if __name__ == "__main__":
    print(f"Using device: {device}")
    if device == "cpu":
        print("(No GPU detected — this will still work, just slower. "
              "See this file's docstring for running on a free Kaggle GPU.)\n")

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}\n"
            "Run 'python data/download_data.py' first."
        )

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        text = f.read()

    chars = sorted(set(text))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    def encode(s):
        return [stoi[c] for c in s]

    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data, val_data = data[:n], data[n:]

    print(f"Dataset: {len(text):,} characters, vocab size {vocab_size}")
    print(f"Train: {len(train_data):,} chars, Val: {len(val_data):,} chars\n")

    model = GPT(vocab_size, N_EMBD, N_HEAD, N_LAYER, BLOCK_SIZE, DROPOUT).to(device)
    print(f"Model parameters: {model.num_params():,}\n")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    start_time = time.time()
    for step in range(MAX_ITERS):
        xb, yb = get_batch(train_data, BLOCK_SIZE, BATCH_SIZE, device)
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % EVAL_INTERVAL == 0 or step == MAX_ITERS - 1:
            losses = estimate_loss(model, train_data, val_data, BLOCK_SIZE, BATCH_SIZE, EVAL_ITERS, device)
            elapsed = time.time() - start_time
            print(f"step {step:5d}  train loss {losses['train']:.4f}  "
                  f"val loss {losses['val']:.4f}  ({elapsed:.1f}s elapsed)")

    total_time = time.time() - start_time
    print(f"\nTraining finished in {total_time:.1f}s.")

    # --- save the checkpoint: weights + vocab + config, everything needed
    # to reload this exact model later without retraining ---
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": model.config,
        "stoi": stoi,
        "itos": itos,
    }, CHECKPOINT_PATH)
    print(f"Checkpoint saved to {CHECKPOINT_PATH}")

    # --- generate a sample to see what the trained model produces ---
    print("\nSample generation:")
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    generated = model.generate(context, max_new_tokens=300, temperature=0.8)[0].tolist()
    print("".join(itos[i] for i in generated))