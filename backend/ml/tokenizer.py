import json
import re
from collections import Counter
from pathlib import Path

from dataset import How2SignDataset


DATA_DIR = "data/how2sign_landmarks"
VOCAB_FILE = "data/how2sign_landmarks/vocab.json"

SPECIAL_TOKENS = [
    "<pad>",
    "<sos>",
    "<eos>",
    "<unk>",
]


def tokenize(text):
    """
    Convert an English sentence into lowercase tokens.

    Example:
        "My name is Dr. Art Bowler."
        ->
        ["my", "name", "is", "dr", "art", "bowler", "."]
    """

    text = text.lower().strip()

    # Keep words and punctuation as separate tokens.
    tokens = re.findall(
        r"\w+|[^\w\s]",
        text
    )

    return tokens


def build_vocabulary(dataset):
    """
    Build vocabulary from training sentences only.
    """

    counter = Counter()

    for index in range(len(dataset)):

        sample = dataset.get_sample(index)

        tokens = tokenize(
            sample["sentence"]
        )

        counter.update(tokens)

    # Start with special tokens.
    vocab = {}

    for token in SPECIAL_TOKENS:
        vocab[token] = len(vocab)

    # Add normal tokens alphabetically.
    for token in sorted(counter.keys()):
        vocab[token] = len(vocab)

    return vocab, counter


def save_vocabulary(vocab, counter):

    output_path = Path(VOCAB_FILE)

    stats = {
        "vocab_size": len(vocab),
        "special_tokens": SPECIAL_TOKENS,
        "vocab": vocab,
        "token_counts": dict(counter),
    }

    with open(output_path, "w") as f:
        json.dump(
            stats,
            f,
            indent=2,
            ensure_ascii=False
        )

    print("\nVocabulary saved to:")
    print(output_path)

    print("\nVocabulary size:", len(vocab))


def encode_sentence(sentence, vocab):

    tokens = tokenize(sentence)

    unk_id = vocab["<unk>"]

    encoded = [
        vocab["<sos>"]
    ]

    encoded.extend(
        vocab.get(token, unk_id)
        for token in tokens
    )

    encoded.append(
        vocab["<eos>"]
    )

    return encoded
def load_vocabulary(vocab_file):
    """
    Load vocabulary from JSON file.
    """

    with open(vocab_file, "r") as f:
        data = json.load(f)

    return data["vocab"]

def decode_sentence(token_ids, vocab):

    reverse_vocab = {
        index: token
        for token, index in vocab.items()
    }

    tokens = []

    for token_id in token_ids:

        token = reverse_vocab.get(
            token_id,
            "<unk>"
        )

        if token in {
            "<pad>",
            "<sos>",
            "<eos>",
        }:
            continue

        tokens.append(token)

    return " ".join(tokens)


if __name__ == "__main__":

    print("Loading training dataset...")

    dataset = How2SignDataset(
        DATA_DIR,
        split="train"
    )

    print("Training samples:", len(dataset))

    print("\nBuilding vocabulary...")

    vocab, counter = build_vocabulary(
        dataset
    )

    save_vocabulary(
        vocab,
        counter
    )

    # Test tokenization.
    test_sentence = (
        "My name is Dr. Art Bowler."
    )

    print("\nTest sentence:")
    print(test_sentence)

    tokens = tokenize(
        test_sentence
    )

    print("\nTokens:")
    print(tokens)

    encoded = encode_sentence(
        test_sentence,
        vocab
    )

    print("\nEncoded:")
    print(encoded)

    decoded = decode_sentence(
        encoded,
        vocab
    )

    print("\nDecoded:")
    print(decoded)