# Phase 1 — Tokenizer

## Goal
Turn text into numbers a model can process, and back again. Build it
yourself — no libraries — so you understand exactly what every AI model
does at its very first step.

## Files
- `tokenizer.py` — the tokenizer implementation (character-level, working)
- `test_tokenizer.py` — tests proving it works correctly

## Run it

```bash
# From the forge/ root folder, with your virtual environment active:
cd tokenizer
python tokenizer.py          # runs the demo at the bottom of the file
pytest test_tokenizer.py -v  # runs the automated tests
```

You should see all 5 tests pass.

## What to do next (in order)

1. **Run the code above.** Confirm it works before changing anything.
2. **Read `tokenizer.py` line by line.** Make sure you understand `stoi`
   (string-to-int) and `itos` (int-to-string) — this pattern shows up
   everywhere in ML code.
3. **Try it on a bigger, real text file.** Download a free book from
   [Project Gutenberg](https://www.gutenberg.org/) as a `.txt` file, put it
   in `../data/`, and build a tokenizer from it instead of the demo string.
4. **Upgrade to word-level tokenization.** Instead of splitting by
   character, split by word (`text.split()`). Compare vocab sizes.
5. **Stretch goal: implement Byte Pair Encoding (BPE).** This is what real
   models (GPT, Llama) use. The idea: start character-level, then
   repeatedly merge the most frequent adjacent pair of tokens into a new
   token, until you hit a target vocab size (e.g. 500). Karpathy's
   "Let's build the GPT Tokenizer" video (free, YouTube) walks through
   this exact build if you get stuck.

## Milestone checklist
- [ ] Ran the demo script successfully
- [ ] All tests pass
- [ ] Tried it on a real downloaded text file
- [ ] Built a word-level version
- [ ] Built a BPE version
- [ ] Committed and pushed to GitHub

## When you're done
Move to `../engine/` for Phase 2 — building a tiny neural network engine
from scratch.
