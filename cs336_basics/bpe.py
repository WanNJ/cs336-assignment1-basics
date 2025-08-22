"""Module that implements Byte Pair Encoding/Decoding"""
import pickle
from copy import copy
import os
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from collections.abc import Iterable, Iterator

import regex as re

from cs336_basics.chunking import find_chunk_boundaries
from cs336_basics.data import bytes_to_tuple


PAT = PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
STRING_ENCODING = 'utf-8'
NUM_PROCESSES=cpu_count()

# NOTE: multiprocess.Queue get(), put(), join(), possibility to deadlock

class NoMoreMerges(Exception):
    pass

def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list]:
    """Train BPE tokenizer.

    Returns:
        vocab: token id to bytes mapping.
        merges: bytes pair generated during training.
    """
    # NOTE: mode = 'rb' is crucial.
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, NUM_PROCESSES, b"<|endoftext|>")

    # Construct sub task arguments.
    task_args = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        task_args.append((input_path, start, end, special_tokens))

    # Synchronize processes and get results.
    with Pool(processes=NUM_PROCESSES) as pool:
        count_results = pool.starmap(get_pre_token_count, task_args, chunksize=1)

    print("Finished pre-tokenization.")

    # Merge pre tokenization results.
    merged_token_counts = defaultdict(int)
    for token_count in count_results:
        for token_tuple, num in token_count.items():
            merged_token_counts[token_tuple] += num

    # Create byte pair --> pretoken index cache
    # NOTE: crucial step to optimize BPE. Could provide 10x performance gain.
    byte_pair_count = defaultdict(int)
    byte_pair_index = defaultdict(set)
    for token_bytes in merged_token_counts.keys():
        for b1, b2 in zip(token_bytes, token_bytes[1:]):
            byte_pair_count[(b1, b2)] += merged_token_counts[token_bytes]
            byte_pair_index[(b1, b2)].add(token_bytes)
            
    # Train BPE
    left_vocab_slots = vocab_size
    left_vocab_slots -= (256 + len(special_tokens))
    merges = []
    while left_vocab_slots > 0:
        # Merges most frequent pairs and update pre-token count.
        try:
            merged_pair = bpe_merge(
                merged_token_counts, byte_pair_count, byte_pair_index
            )
        except NoMoreMerges:
            print(f"BPE merge stopped with {left_vocab_slots} merges left")
            break
        merges.append(merged_pair)
        left_vocab_slots -= 1

        # Print progress.
        if left_vocab_slots % 100 == 0:
            print(f"{left_vocab_slots} merges remain to be performed.")

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

    str_vocab_count = defaultdict(int)

    # Pre-Tokenization
    pattern = '|'.join(map(re.escape,  special_tokens))
    segments = re.split(pattern, chunk) if pattern else [chunk]
    for segment in segments:
        for match in re.finditer(PAT, segment):
            # TODO: Might be too slow, profile it.
            # NOTE: Why do we break down bytes of a single character instead of preserving them like the following one?
            # vocab[tuple([char.encode(STRING_ENCODING) for char in match.group()])] += 1
             str_vocab_count[match.group()] += 1
       
    # Delay the process of converting string to bytes to optimize performance
    bytes_vocab_count = {}
    for key, count in str_vocab_count.items():
        bytes_vocab_count[tuple(map(int.to_bytes, key.encode(encoding=STRING_ENCODING)))] = count
    return bytes_vocab_count


def merge_pretoken_tuple(byte_tuple: tuple[bytes], merge: tuple[bytes], merged_bytes: bytes):
    new_byte_tuple = ()
    l = r = 0
    while r < len(byte_tuple) - 1:
        if byte_tuple[r:r+2] == merge:
            new_byte_tuple += byte_tuple[l:r]
            new_byte_tuple += (merged_bytes,)
            r += 2
            l = r
        else:
            r += 1
    new_byte_tuple += byte_tuple[l:]
    return new_byte_tuple


def bpe_merge(
    token_counts: dict[tuple[bytes], int],
    byte_pair_count: dict[tuple[bytes, bytes], int],
    byte_pair_index: dict[tuple[bytes, bytes], set]
):
    """Perform one step of BPE merge.

    IMPORTANT: This function updates the three parameters in-place.
    """
    # TODO: Fix this termination logic.
    if not byte_pair_count:
        raise NoMoreMerges('No more merges needed.')

    # deterministically break ties in pair frequency by
    # preferring the lexicographically greater pair
    most_frequent_pair = max(byte_pair_count, key=lambda x: (byte_pair_count.get(x), x))
    merged_pair_bytes = b''.join(most_frequent_pair)

    affected_tokens = byte_pair_index[most_frequent_pair]
    for token_tuple in copy(affected_tokens):
        new_token_tuple = merge_pretoken_tuple(token_tuple, most_frequent_pair, merged_pair_bytes)
        assert new_token_tuple != token_tuple

        # Update bytes pair count and index.
        for b1, b2 in zip(token_tuple, token_tuple[1:]):
            byte_pair_count[(b1, b2)] -= token_counts[token_tuple]
            # NOTE: do not use remove, otherwise KeyError might occur.
            byte_pair_index[(b1, b2)].discard(token_tuple)
        for b1, b2 in zip(new_token_tuple, new_token_tuple[1:]):
            byte_pair_count[(b1, b2)] += token_counts[token_tuple]
            byte_pair_index[(b1, b2)].add(new_token_tuple)

        # Update token bytes tuple after merge.
        token_counts[new_token_tuple] = token_counts[token_tuple]
        # NOTE: Deletion has to be the last step because we accessed token_counts[token_tuple] in previous steps.
        del token_counts[token_tuple]

    # After merge, the count of previously most frequent pair should go to zero.
    assert byte_pair_count[most_frequent_pair] == 0
    return most_frequent_pair


def count_byte_pair(token_counts: dict[tuple[bytes], int]) -> dict[tuple[bytes, bytes], int]:
    result = defaultdict(int)
    for token_bytes in token_counts.keys():
        for bytes_pair in zip(token_bytes[:-1], token_bytes[1:]):
            result[bytes_pair] += token_counts[token_bytes]
    return result


def get_vocabulary(merges: list[tuple[bytes, bytes]], special_tokens: list[str]):
    # NOTE: should we try to encode most frequent pair with lower number?
    vocab = {i: bytes([i]) for i in range(256)}
    current_token_id = 256
    for special_token in special_tokens:
        vocab[current_token_id] = special_token.encode(STRING_ENCODING)
        current_token_id += 1
    for merge in merges:
        vocab[current_token_id] = b''.join(merge)
        current_token_id += 1
    return vocab


class Tokenizer:
    def __init__(self,
                 vocab: dict[int, bytes],
                 merges: list[tuple[bytes, bytes]],
                 special_tokens: list[str] | None = None):
        # Make sure special_tokens are in vocabulary
        if special_tokens:
            for special_token in special_tokens:
                special_token_bytes = special_token.encode(STRING_ENCODING)
                if special_token_bytes not in vocab.values():
                    print(f"WARNING: vocabulary does not contain {special_token} initially.")
                    vocab[len(vocab)] = special_token_bytes

        self.bytes_map: dict[int, bytes] = vocab
        self.token_id_map: dict[bytes, int] = {bytes: token_id for token_id, bytes in vocab.items()}
        self.merge_ranks: dict[tuple, int] = {merges[i]: i for i in range(len(merges))}
        self.special_tokens = special_tokens
        if special_tokens is None:
            self.special_tokens = []
        self.special_tokens.sort(key=len, reverse=True)

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        """Class method that constructs and return a Tokenizer from a serialized vocabulary and list of merges
        (in the same format that your BPE training code output) and (optionally) a list of special
        tokens."""
        with open(vocab_filepath, 'rb') as vocab_file:
            vocab = pickle.load(vocab_file)
        with open(merges_filepath, 'rb') as merges_file:
            merges = pickle.load(merges_file)
        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        """Encode an input text into a sequence of token IDs."""
        # TODO: Optimize encoding performance, too slow.
        encoded_token_ids = []
        pattern = '|'.join(map(re.escape,  self.special_tokens))
        # NOTE: the pattern is in () to preserve the special tokens in the result.
        segments = re.split(f"({pattern})", text) if self.special_tokens else [text]

        for segment in segments:
            if segment in self.special_tokens:
                encoded_token_ids.append(self.token_id_map[segment.encode(STRING_ENCODING)])
                continue
            for match in re.finditer(PAT, segment):
                pretoken_bytes = match.group().encode(STRING_ENCODING)
                pretoken_byte_tuple = bytes_to_tuple(pretoken_bytes)

                # NOTE: O(len(token) ^ 2)
                while len(pretoken_byte_tuple) > 1:
                    # Find the highest priority merge in the pretoken.
                    # NOTE: O(len(token))
                    bytes_pairs = zip(pretoken_byte_tuple, pretoken_byte_tuple[1:])
                    min_rank = float("inf")
                    merge = None
                    for pair in bytes_pairs:
                        rank = self.merge_ranks.get(pair, float("inf"))
                        # float("inf") < float("inf") is False
                        if rank < min_rank:
                            min_rank = rank
                            merge = pair
                    # If no valid ranked pair is present, we're done.
                    if merge is None:
                        break
                    # Merge the highest priority pair.
                    # NOTE: O(len(token))
                    pretoken_byte_tuple = merge_pretoken_tuple(
                        pretoken_byte_tuple,
                        merge, 
                        b''.join(merge)
                    )

                for merged_bytes in pretoken_byte_tuple:
                    encoded_token_ids.append(self.token_id_map[merged_bytes])
        return encoded_token_ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """Given an iterable of strings (e.g., a Python file handle), return a generator
        that lazily yields token IDs. This is required for memory-efficient tokenization of
        large files that we cannot directly load into memory."""
        for s in iterable:
            # NOTE: must use yield from, instead of yield
            yield from self.encode(s)

    def decode(self, ids: list[int]) -> str:
        result_bytes = b''
        for token_id in ids:
            if token_id not in self.bytes_map:
                raise ValueError(f"token_id {token_id} not in our vocabulary")
            result_bytes += self.bytes_map[token_id]
        return result_bytes.decode('utf-8', errors='replace')


if __name__ == "__main__":
    import time
    starttime = time.time()
    vocab, merges = train_bpe(
        "/Users/jackwan/workspace/cs336/assignment1-basics/data/owt_train.txt",
        32000,
        special_tokens=["<|endoftext|>"]
    )
    endtime = time.time()
    print(f"Finished. Took {endtime - starttime} seconds.")
    print(f"Longest word in vocabulary: {max(vocab.values(), key=lambda x: len(x))}")
    print(f"Top 10 merges: {merges[:10]}")

    # import pickle
    # with open('results/bpe/owt_trained_vocab.pkl', 'wb') as file:
    #     pickle.dump(vocab, file)
    # with open('results/bpe/owt_trained_merges.pkl', 'wb') as file:
    #     pickle.dump(merges, file)
