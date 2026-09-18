"""
Phase 6: Forge CLI — the shipped application.

This ties together everything built across Phases 1-5 into one real,
usable command-line tool:
  - Phase 5's agent loop (decide -> act -> observe -> answer)
  - Phase 5's retrieval system (TF-IDF knowledge base search)
  - A calculator tool
  - A simple REPL (Read-Eval-Print Loop) so you can actually use it

Run it, ask it questions, and it responds — entirely locally, no cloud,
no API key, no cost. This is "Forge" — the finished product of the
10-year roadmap's first project.

Requires retrieval.py and agent.py to be in ../rag_agent/ relative to
this file (that's where Phase 5 built them).
"""

from __future__ import annotations
import sys
import os

# Allow importing from the rag_agent/ folder (Phase 5's code) without
# copy-pasting it again — real projects share code across folders like this.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "rag_agent"))

from neural_retrieval import NeuralVectorStore    # noqa: E402
from agent import tool_calculator, make_search_tool, run_agent  # noqa: E402


DEFAULT_KNOWLEDGE_BASE = [
    "The Eiffel Tower is located in Paris, France, and was completed in 1889.",
    "Python is a popular programming language known for readability.",
    "Rust is a systems programming language focused on memory safety.",
    "The Great Wall of China stretches over 13,000 miles.",
    "Transformers are a neural network architecture built around attention.",
    "Mount Everest is the tallest mountain above sea level on Earth.",
    "PyTorch is a deep learning framework widely used for research.",
    "The Amazon rainforest produces a significant share of Earth's oxygen.",
    "Forge is a 10-year project to build an AI stack entirely from scratch.",
]


def build_agent():
    """Set up the vector store and tools, return a ready-to-use tools dict."""
    store = NeuralVectorStore()
    for doc in DEFAULT_KNOWLEDGE_BASE:
        store.add(doc)

    tools = {
        "calculator": tool_calculator,
        "search": make_search_tool(store),
    }
    return tools, store


def print_banner():
    print("=" * 60)
    print("  FORGE — your own AI stack, built from scratch")
    print("=" * 60)
    print("Ask a question. I can do math or search a small knowledge base.")
    print("Type 'add: <fact>' to teach me something new.")
    print("Type 'quit' or 'exit' to stop.\n")


def repl():
    """The main interactive loop — Read a line, Evaluate it, Print the
    result, Loop. This is the same basic structure every CLI tool
    (including python's own interactive shell) is built on."""
    tools, store = build_agent()
    print_banner()

    while True:
        try:
            user_input = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting. Thanks for using Forge.")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit"):
            print("Exiting. Thanks for using Forge.")
            break

        if user_input.lower().startswith("add:"):
            new_fact = user_input[len("add:"):].strip()
            if new_fact:
                store.add(new_fact)
                print(f"forge> Learned: {new_fact!r}\n")
            else:
                print("forge> Nothing to add — write 'add: your fact here'.\n")
            continue

        answer = run_agent(user_input, tools)
        print(f"forge> {answer}\n")


if __name__ == "__main__":
    repl()