"""Module that implements Byte Pair Encoding/Decoding"""
import os
from collections import defaultdict
from multiprocessing import Pool, cpu_count

import regex as re

from cs336_basics.chunking import find_chunk_boundaries


PAT = PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
STRING_ENCODING = 'utf-8'
NUM_PROCESSES=cpu_count()


def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
):
    """_summary_

    Args:
        input_path (str): _description_
        vocab_size (int): _description_
        special_tokens (list[str]): _description_
    """
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, NUM_PROCESSES, b"<|endoftext|>")

    # Construct sub task arguments.
    task_args = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        task_args.append((input_path, start, end, special_tokens))

    with Pool(processes=NUM_PROCESSES) as pool:
        count_results = pool.starmap(get_pre_token_count, task_args, chunksize=1)

    # Merge pre tokenization results.
    merged_token_counts = defaultdict(int)
    for token_count in count_results:
        for token_tuple, num in token_count.items():
            merged_token_counts[token_tuple] += num

    # Train BPE
    left_vocab_slots = vocab_size
    left_vocab_slots -= (256 + len(special_tokens))
    merges = []
    while left_vocab_slots > 0:
        # Merges most frequent pairs and update pre-token count.
        merged_pair, new_token_count = bpe_merge(
            merged_token_counts
        )
        merged_token_counts = new_token_count
        merges.append(merged_pair)
        left_vocab_slots -= 1

    return get_vocabulary(merges, special_tokens), merges


def get_pre_token_count(
    input_path: str,
    start: int,
    end: str,
    special_tokens: list[str],
) -> dict[tuple[bytes], int]:
    """Read the file data in the sub process to parallelize the disk read.
    """
    print(f"Worker {os.getpid()} processing data from {start} to {end}")

    # NOTE: do not use mode = r and encoding = 'utf-8'
    with open(input_path, 'rb') as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8")

    vocab = defaultdict(int)
    segments = re.split('|'.join([re.escape(special_token) for special_token in special_tokens]), chunk)
    for segment in segments:
        matches = re.finditer(PAT, segment)
        for match in matches:
            # TODO: Might be too slow, profile it.
            # vocab[tuple([char.encode(STRING_ENCODING) for char in match.group()])] += 1
            vocab[tuple(map(int.to_bytes, match.group().encode(encoding=STRING_ENCODING)))] += 1
            # NOTE: Why do we break down bytes of a single character instead of preserving them?
    return vocab

def bpe_merge(
    token_counts: dict[tuple[bytes], int]
):
    # TODO: This function might be optimized. Figure it out.
    bytes_pair_count = count_byte_pair(token_counts)
    # deterministically break ties in pair frequency by
    # preferring the lexicographically greater pair
    most_frequent_pair = max(bytes_pair_count, key=lambda x: (bytes_pair_count.get(x), x))
    combined = b''.join(most_frequent_pair)
    new_token_counts = defaultdict(int)
    for token_tuple in token_counts:
        new_token_tuple = ()
        l = r = 0
        while r < len(token_tuple) - 1:
            if token_tuple[r:r+2] == most_frequent_pair:
                new_token_tuple += token_tuple[l:r]
                new_token_tuple += (combined,)
                r += 2
                l = r
            else:
                r += 1
        new_token_tuple += token_tuple[l:]
        # This handles both cases: no matter the new token changes or not.
        new_token_counts[new_token_tuple] += token_counts[token_tuple]

    return most_frequent_pair, new_token_counts


def count_byte_pair(token_counts: dict[tuple[bytes], int]) -> dict[tuple[bytes, bytes], int]:
    result = defaultdict(int)
    for token_bytes in token_counts.keys():
        for bytes_pair in zip(token_bytes[:-1], token_bytes[1:]):
            result[bytes_pair] += token_counts[token_bytes]
    return result


def get_vocabulary(merges: list[tuple[bytes, bytes]], special_tokens: list[str]):
    # Generate Vocabulary
    vocab = {i: bytes([i]) for i in range(256)}
    current_token_id = 256
    for special_token in special_tokens:
        vocab[current_token_id] = special_token.encode(STRING_ENCODING)
        current_token_id += 1
    for merge in merges:
        vocab[current_token_id] = b''.join(merge)
        current_token_id += 1
    return vocab


if __name__ == "__main__":
    vocab, merges = train_bpe(
        "/Users/jackwan/workspace/cs336/assignment1-basics/data/TinyStories-TinyTinySet.txt",
        500,
        special_tokens=["<|endoftext|>"]
    )
    print(f"Longest word in vocabulary: {max(vocab.values(), key=lambda x: len(x))}")
    print(f"Top 3 merges: {merges[:3]}")
