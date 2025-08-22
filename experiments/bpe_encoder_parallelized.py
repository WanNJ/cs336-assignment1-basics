import os
from pathlib import Path
import time

import numpy as np

from cs336_basics.bpe import Tokenizer

PROJECT_PATH = Path(__file__).resolve().parent.parent
BPE_PATH = PROJECT_PATH / "results/bpe"
DATA_PATH = PROJECT_PATH / "data"


def benchmark_tokenizer(tokenizer: Tokenizer, input_path):
    bytes_size = os.path.getsize(input_path)

    start = time.perf_counter()
    encoded = tokenizer.encode_file_parallelized(input_path)
    end = time.perf_counter()

    execution_time = end - start
    throughput_mbps = bytes_size / execution_time / 1e6  # Convert to KB/s
    encoded = np.array(encoded, dtype=np.uint16)
    print(f"Compression ratio = {bytes_size} / {len(encoded)} = {bytes_size / len(encoded):.2f}")
    print(f"Throughput = {throughput_mbps:.2f} MB / second\n")


if __name__ == "__main__":
    owt_data_path = DATA_PATH / "owt_valid.txt"
    tinystory_data_path = DATA_PATH / "TinyStoriesV2-GPT4-valid.txt"

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

    print("Benchmarking TinyStory Tokenizer on TinyStory data:")
    benchmark_tokenizer(tinystory_tokenizer, tinystory_data_path)
    print("Benchmarking OWT Tokenizer on TinyStory data:")
    benchmark_tokenizer(owt_tokenizer, tinystory_data_path)

    print("Benchmarking TinyStory Tokenizer on OWT data:")
    benchmark_tokenizer(tinystory_tokenizer, owt_data_path)
    print("Benchmarking OWT Tokenizer with {data_path}:")
    benchmark_tokenizer(owt_tokenizer, owt_data_path)
