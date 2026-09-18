"""
Tests for the Phase 1 tokenizer.

Run with:  pytest tokenizer/test_tokenizer.py -v
"""

import pytest
from tokenizer import CharTokenizer


def test_round_trip_simple():
    text = "hello world"
    tok = CharTokenizer.from_text(text)
    assert tok.decode(tok.encode(text)) == text


def test_round_trip_with_punctuation():
    text = "Forge: build your own AI stack, from scratch!"
    tok = CharTokenizer.from_text(text)
    assert tok.decode(tok.encode(text)) == text


def test_vocab_size_matches_unique_chars():
    text = "aaabbbccc"
    tok = CharTokenizer.from_text(text)
    assert tok.vocab_size() == 3  # a, b, c


def test_unseen_character_raises():
    tok = CharTokenizer.from_text("abc")
    with pytest.raises(ValueError):
        tok.encode("xyz")


def test_empty_string():
    tok = CharTokenizer.from_text("abc")
    assert tok.encode("") == []
    assert tok.decode([]) == ""


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
