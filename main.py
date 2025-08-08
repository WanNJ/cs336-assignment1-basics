import pickle

with open("/Users/jackwan/workspace/cs336/assignment1-basics/trained_vocab.pkl", mode="rb") as f:
    vocab = pickle.load(f, encoding='utf-8')
    print(vocab)
