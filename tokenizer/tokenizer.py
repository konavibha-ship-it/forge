"""
Phase 1: A tokenizer built entirely from scratch, no libraries.

Start here: a character-level tokenizer. This is the simplest possible
version — every unique character in your text becomes one token.

Once this works end-to-end (encode -> decode -> matches original), the next
upgrade is Byte Pair Encoding (BPE), which merges frequent character pairs
into single tokens (this is what GPT-style models actually use). That's the
Phase 1 "stretch goal" — build this version first, get it working, then
extend the same file to add BPE.
"""

from __future__ import annotations


class CharTokenizer:
    """A minimal character-level tokenizer.

    Usage:
        text = "hello world"
        tok = CharTokenizer.from_text(text)
        ids = tok.encode(text)
        back = tok.decode(ids)
        assert back == text
    """

    def __init__(self, vocab: list[str]):
        # vocab: a list of unique characters, order defines the token ids
        self.vocab = vocab
        self.stoi = {ch: i for i, ch in enumerate(vocab)}  # string -> int
        self.itos = {i: ch for i, ch in enumerate(vocab)}  # int -> string

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        """Build a tokenizer's vocabulary from a sample of text."""
        unique_chars = sorted(set(text))
        return cls(unique_chars)

    def encode(self, text: str) -> list[int]:
        """Turn a string into a list of token ids."""
        try:
            return [self.stoi[ch] for ch in text]
        except KeyError as e:
            raise ValueError(
                f"Character {e.args[0]!r} not in vocabulary. "
                "This tokenizer only knows characters seen in from_text()."
            )

    def decode(self, ids: list[int]) -> str:
        """Turn a list of token ids back into a string."""
        return "".join(self.itos[i] for i in ids)

    def vocab_size(self) -> int:
        return len(self.vocab)


if __name__ == "__main__":
    # Quick manual demo — run this file directly to see it work:
    #   python tokenizer/tokenizer.py
    sample = "hello world, this is forge phase one!"
    tok = CharTokenizer.from_text(sample)

    print(f"Vocab size: {tok.vocab_size()}")
    print(f"Vocab: {tok.vocab}")

    encoded = tok.encode(sample)
    print(f"Encoded: {encoded}")

    decoded = tok.decode(encoded)
    print(f"Decoded: {decoded}")

    assert decoded == sample, "Round-trip failed!"
    print("Round-trip successful: decode(encode(text)) == text")
