"""
Load a saved checkpoint (model/checkpoint.pt) and generate text from it,
without retraining. This is real "inference" — using an already-trained
model, which is what actually happens every time you use a deployed AI
product; training only happens once (or occasionally), inference happens
every single time someone uses the model.

Run:
    python model/generate.py
    python model/generate.py --prompt "ROMEO:" --length 400 --temperature 0.7
"""

import os
import argparse
import torch
from architecture import GPT

CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "checkpoint.pt")


def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]

    model = GPT(**config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, checkpoint["stoi"], checkpoint["itos"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="", help="Text to start generation from.")
    parser.add_argument("--length", type=int, default=300, help="Number of characters to generate.")
    parser.add_argument("--temperature", type=float, default=0.8,
                         help="Higher = more random/creative, lower = more predictable.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(
            f"No checkpoint found at {CHECKPOINT_PATH}\n"
            "Run 'python model/train_large.py' first to train and save one."
        )

    model, stoi, itos = load_model(CHECKPOINT_PATH, device)
    print(f"Loaded model ({model.num_params():,} parameters) on {device}\n")

    if args.prompt:
        # Encode the prompt, skipping any characters the model never saw
        # during training (its vocabulary is fixed to what it learned from).
        context_ids = [stoi[c] for c in args.prompt if c in stoi]
        if not context_ids:
            print("Warning: none of the prompt's characters are in the "
                  "model's vocabulary — starting from empty context instead.")
            context_ids = [0]
    else:
        context_ids = [0]

    context = torch.tensor([context_ids], dtype=torch.long, device=device)
    generated = model.generate(context, max_new_tokens=args.length, temperature=args.temperature)
    text = "".join(itos[i] for i in generated[0].tolist())

    print(text)