from pathlib import Path

import numpy as np

from cs336_basics.bpe import Tokenizer

PROJECT_PATH = Path(__file__).resolve().parent.parent
BPE_PATH = PROJECT_PATH / "results/bpe"
DATA_PATH = PROJECT_PATH / "data"


if __name__ == "__main__":
    tokenizer = Tokenizer.from_files(
        BPE_PATH / "owt_trained_vocab.pkl",
        BPE_PATH / "owt_trained_merges.pkl",
        ["<|endoftext|>"]
    )

    output_path = "results/bpe/tinystory_sample_encoded.npy"
    tokenizer.encode_file_parallelized(DATA_PATH / "TinyStoriesV2-GPT4-valid.txt", output_path)

    with open(output_path, "rb") as f:
        encoded = np.load(f)
    decoded = tokenizer.decode(encoded)

    with open(DATA_PATH / "TinyStoriesV2-GPT4-valid.txt", encoding='utf-8') as f:
        original = f.read()
    print(decoded == original)
