# Forge — An AI Stack, Built Entirely From Scratch

Forge is a from-scratch implementation of a modern AI stack: a tokenizer,
an autograd engine, a trained transformer language model, a systems-level
inference engine in Rust, a retrieval system, an agent loop, and a shipped
CLI application — built end-to-end, with zero paid tools, zero API keys,
and zero cost.

The goal wasn't to *use* AI tools. It was to *build* the things AI tools
are made of, by hand, well enough to understand exactly what's happening
at every layer.

---

## What's actually in here

| Component | What it is | Built from scratch? |
|---|---|---|
| **Tokenizer** | Character-level + real Byte Pair Encoding (BPE) | ✅ No libraries |
| **Autograd engine** | A `Value` class with automatic differentiation (backpropagation) | ✅ No libraries |
| **Neural network** | Neuron → Layer → MLP, trained with gradient descent | ✅ Built on the autograd engine above |
| **Transformer** | GPT-style architecture: multi-head self-attention, causal masking, layer norm | ✅ PyTorch for tensor ops, architecture hand-written |
| **Trained language model** | Trained on Tiny Shakespeare (~1.1M characters), 2.7M parameters | ✅ Real training run, real dataset |
| **Rust matrix engine** | Matrix multiplication from first principles | ✅ Zero external crates |
| **Retrieval (RAG)** | TF-IDF search **and** real neural embeddings | ✅ Both implemented and compared |
| **Agent loop** | Thought → Action → Observation → Answer, with tool selection | ✅ Calculator + search tools |
| **Shipped app** | An interactive CLI that ties everything together | ✅ Runs fully locally |

---

## The engineering story, not just the code

The interesting part of this project isn't that everything worked — it's
where things *didn't*, and what that revealed:

- **The TF-IDF retrieval system had a real, discoverable limitation.**
  Asking *"What did the developer use?"* failed to match a fact containing
  "using free tools" — because TF-IDF only measures word overlap, not
  meaning. Swapping in real neural embeddings (`sentence-transformers`)
  fixed it, and the fix is directly comparable in the repo
  (`rag_agent/retrieval.py` vs. `rag_agent/neural_retrieval.py`).

- **Training on a real dataset took 18 hours on a CPU**, not the
  optimistic minutes originally estimated. That's a genuine lesson in
  compute cost, and the direct motivation for adding free-GPU support
  (Kaggle) and a resumable training loop (`--resume`, checkpointing every
  N steps) so a long run is never lost to an interruption.

- **The generated text is genuinely legible as "trying to be
  Shakespeare"** — archaic phrasing, verse-like structure, correct
  spelling — while still being clearly a small model's output, not
  cherry-picked. That gap between "structurally correct" and "fully
  coherent" is exactly the kind of thing you only understand by training
  a model yourself, at a scale where you can watch it happen.

---

## Project structure