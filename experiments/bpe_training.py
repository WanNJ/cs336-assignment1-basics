import time

from cs336_basics.bpe import train_bpe


if __name__ == "__main__":
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
