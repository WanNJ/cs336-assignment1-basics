import os
from pathlib import Path
import time

import numpy as np

from cs336_basics.bpe import Tokenizer

PROJECT_PATH = Path(__file__).resolve().parent.parent
BPE_PATH = PROJECT_PATH / "results/bpe"
DATA_PATH = PROJECT_PATH / "data"


def benchmark_tokenizer(tokenizer, text, bytes_size):
    start = time.perf_counter()
    encoded = tokenizer.encode(text)
    end = time.perf_counter()
    execution_time = end - start
    throughput_mbps = bytes_size / execution_time / 1e6  # Convert to MB/s
    encoded = np.array(encoded, dtype=np.uint16)
    print(f"Compression ratio = {bytes_size} / {len(encoded)} = {bytes_size / len(encoded):.2f}")
    print(f"Throughput = {throughput_mbps:.2f} MB / second")


if __name__ == "__main__":
    data_path = DATA_PATH / "owt_sample.txt"
    # data_path = DATA_PATH / "TinyStories-sample.txt"

    tinystory_tokenizer = Tokenizer.from_files(
        BPE_PATH / "tiny_story_trained_vocab.pkl",
        BPE_PATH / "tiny_story_trained_merges.pkl",
        ["<|endoftext|>"]
    )
    owt_tokenizer = Tokenizer.from_files(
        BPE_PATH / "owt_trained_vocab.pkl",
        BPE_PATH / "owt_trained_merges.pkl",
        ["<|endoftext|>"]
    )

    with open(data_path, "r") as file:
        target_text = file.read()
    bytes_size = os.path.getsize(data_path)

    print("Benchmarking TinyStory Tokenizer:")
    benchmark_tokenizer(tinystory_tokenizer, target_text, bytes_size)

    print("\nBenchmarking OWT Tokenizer:")
    benchmark_tokenizer(owt_tokenizer, target_text, bytes_size)
