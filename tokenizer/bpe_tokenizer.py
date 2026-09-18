"""
Phase 1 stretch goal: Byte Pair Encoding (BPE) tokenizer, from scratch.

This is the algorithm real models (GPT, Llama) use. The idea:

1. Start by treating every CHARACTER as a token (like CharTokenizer).
2. Look at the text as a sequence of tokens, and count every adjacent PAIR
   of tokens (e.g. "t"+"h", "h"+"e", ...).
3. Find the most frequent pair, and merge it into one new token
   (e.g. "t"+"h" -> "th").
4. Repeat steps 2-3 until you've done `num_merges` merges, or hit your
   target vocab size.

The result: common chunks like "the", "ing", "tion" become single tokens,
which is far more efficient than one-token-per-character for real text.

Reference: Karpathy's "Let's build the GPT Tokenizer" (free, YouTube)
walks through this exact algorithm in detail if you want to go deeper.
"""

from __future__ import annotations
from collections import Counter


class BPETokenizer:
    def __init__(self):
        self.merges: dict[tuple[int, int], int] = {}   # (id1, id2) -> new_id
        self.vocab: dict[int, bytes] = {}               # id -> bytes

    def train(self, text: str, num_merges: int) -> None:
        """Learn `num_merges` merge rules from the given text."""
        # Start from raw UTF-8 bytes -> ids 0-255. Working in bytes (not
        # characters) is what real BPE tokenizers do, since it handles any
        # language/emoji/symbol without needing a huge starting vocab.
        tokens = list(text.encode("utf-8"))

        # Base vocab: every byte value maps to itself.
        self.vocab = {i: bytes([i]) for i in range(256)}

        for i in range(num_merges):
            pair_counts = self._count_pairs(tokens)
            if not pair_counts:
                break  # no more pairs to merge (text fully merged)

            best_pair = max(pair_counts, key=pair_counts.get)
            new_id = 256 + i  # new token ids start after the 256 byte ids

            tokens = self._merge(tokens, best_pair, new_id)
            self.merges[best_pair] = new_id
            self.vocab[new_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]

            print(
                f"merge {i + 1}/{num_merges}: {best_pair} -> {new_id} "
                f"({self.vocab[new_id]!r}) had {pair_counts[best_pair]} occurrences"
            )

    def encode(self, text: str) -> list[int]:
        """Turn text into token ids, applying learned merges in order."""
        tokens = list(text.encode("utf-8"))

        while len(tokens) >= 2:
            pair_counts = self._count_pairs(tokens)
            # Pick the pair that was learned EARLIEST in training (lowest
            # merge id) — merges must be applied in the order they were
            # learned for encoding to match training.
            candidate = min(
                pair_counts, key=lambda p: self.merges.get(p, float("inf"))
            )
            if candidate not in self.merges:
                break  # no more applicable merges
            tokens = self._merge(tokens, candidate, self.merges[candidate])

        return tokens

    def decode(self, ids: list[int]) -> str:
        """Turn token ids back into text."""
        byte_chunks = [self.vocab[i] for i in ids]
        return b"".join(byte_chunks).decode("utf-8", errors="replace")

    @staticmethod
    def _count_pairs(tokens: list[int]) -> Counter:
        return Counter(zip(tokens, tokens[1:]))

    @staticmethod
    def _merge(tokens: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
        """Replace every occurrence of `pair` in `tokens` with `new_id`."""
        merged = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == pair:
                merged.append(new_id)
                i += 2
            else:
                merged.append(tokens[i])
                i += 1
        return merged


if __name__ == "__main__":
    # Quick manual demo — run this file directly:
    #   python tokenizer/bpe_tokenizer.py
    sample = (
        "the quick brown fox jumps over the lazy dog. "
        "the dog barks at the fox. the fox runs away quickly."
        "Once upon a time, in a quiet little neighborhood, a golden retriever puppy named Candy changed everything"
        "She had fur the color of warm toasted marshmallows and a tail that never stopped wagging"
        "From the very first day she arrived, she brought an undeniable sweetness into the household"
        "She was a bundle of pure, unadulterated energy, bound to make everyone smile"
    )

    tok = BPETokenizer()
    tok.train(sample, num_merges=100)

    print()
    print(f"Final vocab size: {len(tok.vocab)}")

    encoded = tok.encode(sample)
    print(f"Original length (bytes): {len(sample.encode('utf-8'))}")
    print(f"Encoded length (tokens): {len(encoded)}")

    decoded = tok.decode(encoded)
    assert decoded == sample, "Round-trip failed!"
    print("Round-trip successful: decode(encode(text)) == text")