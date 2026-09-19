"""
Downloads a real training dataset: "Tiny Shakespeare" — the complete works
of Shakespeare, concatenated into one ~1MB text file. This is the same
dataset Andrej Karpathy uses in his char-rnn and nanoGPT tutorials — a
genuine, real-world benchmark for small character-level language models,
public domain, free.

Run this once:
    python data/download_data.py
"""

import urllib.request
import os

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "shakespeare.txt")

if __name__ == "__main__":
    if os.path.exists(OUTPUT_PATH):
        size_kb = os.path.getsize(OUTPUT_PATH) / 1024
        print(f"Already downloaded: {OUTPUT_PATH} ({size_kb:.1f} KB)")
    else:
        print(f"Downloading from {URL} ...")
        try:
            urllib.request.urlretrieve(URL, OUTPUT_PATH)
            size_kb = os.path.getsize(OUTPUT_PATH) / 1024
            print(f"Saved to {OUTPUT_PATH} ({size_kb:.1f} KB)")
        except Exception as e:
            print(f"Download failed: {e}")
            print("If this keeps failing, download the file manually from:")
            print(f"  {URL}")
            print(f"and save it as: {OUTPUT_PATH}")