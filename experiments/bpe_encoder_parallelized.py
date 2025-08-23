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
    print(f"Processed {bytes_size / 1024 / 1024:.2f} MB of data, took {execution_time} seconds")
    print(f"Compression ratio = {bytes_size} / {len(encoded)} = {bytes_size / len(encoded):.2f}")
    print(f"Throughput = {throughput_mbps:.2f} MB / second\n")


def test_with_validation_data():
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

    tinystory_data_path = DATA_PATH / "TinyStoriesV2-GPT4-valid.txt"
    print("Benchmarking TinyStory Tokenizer on TinyStory data:")
    benchmark_tokenizer(tinystory_tokenizer, tinystory_data_path)
    print("Benchmarking OWT Tokenizer on TinyStory data:")
    benchmark_tokenizer(owt_tokenizer, tinystory_data_path)

    owt_data_path = DATA_PATH / "owt_valid.txt"
    print("Benchmarking TinyStory Tokenizer on OWT data:")
    benchmark_tokenizer(tinystory_tokenizer, owt_data_path)
    print("Benchmarking OWT Tokenizer on OWT data:")
    benchmark_tokenizer(owt_tokenizer, owt_data_path)


def encode_train_dataset():
    owt_train_data_path = DATA_PATH / "owt_train.txt"
    tinystory_train_data_path = DATA_PATH / "TinyStoriesV2-GPT4-train.txt"

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

    tinystory_encoded = tinystory_tokenizer.encode_file_parallelized(tinystory_train_data_path)
    np.save("results/bpe/tinystory_encoded.npy", np.array(tinystory_encoded, dtype=np.uint16))

    owt_encoded = owt_tokenizer.encode_file_parallelized(owt_train_data_path)
    np.save("results/bpe/owt_encoded.npy", arr=np.array(owt_encoded, dtype=np.uint16))


if __name__ == "__main__":
    test_with_validation_data()
    encode_train_dataset()
