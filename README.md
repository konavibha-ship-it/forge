# Forge — Build Your Own AI Stack From Scratch

A 10-year, zero-cost project: build an entire LLM stack yourself, from
tokenizer to trained model to deployed app. Every phase uses free tools only.

## How to use this repo
Work through the phases in order. Each phase has its own folder with a
`README.md` explaining what to build and why. Check off tasks as you finish
them. Commit after every working milestone — small, frequent commits.

## Phases

- [ ] **Phase 1 — `tokenizer/`** — Build a tokenizer from scratch (Python)
- [ ] **Phase 2 — `engine/`** — Build a tiny autograd/neural net engine (Python → C++)
- [ ] **Phase 3 — `model/`** — Train a small transformer from scratch (Python + PyTorch)
- [ ] **Phase 4 — `inference/`** — Fast inference engine (Rust)
- [ ] **Phase 5 — `rag_agent/`** — Retrieval + agent loop (Python + Rust)
- [ ] **Phase 6 — `app/`** — Ship it as a real local app (Rust/Python)

## Tools (all free, no credits, no card required)
- VS Code + Python/Pylance/Jupyter/C++/rust-analyzer extensions
- Python 3.10+, GCC/Clang, Rust (via rustup)
- Git + GitHub
- Kaggle notebooks (30 hrs/week free GPU) and/or Google Colab free tier
- Hugging Face datasets, Wikipedia dumps, Project Gutenberg (free text data)

## Setup

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Progress log
Keep a running log of what you build and learn — this becomes your portfolio
story later.

- YYYY-MM-DD: Started Phase 1.
