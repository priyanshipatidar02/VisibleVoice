import torch
from torch.utils.data import Dataset

from dataset import How2SignDataset
from preprocessing import preprocess_geometric_features
from tokenizer import (
    load_vocabulary,
    encode_sentence,
)


class How2SignTorchDataset(Dataset):

    def __init__(self, data_dir, split, vocab_file):
        self.dataset = How2SignDataset(
            data_dir,
            split=split
        )

        self.vocab = load_vocabulary(
            vocab_file
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):

        sample = self.dataset.get_sample(index)

        features = preprocess_geometric_features(
            sample["geometric"]
        )

        features = torch.tensor(
            features,
            dtype=torch.float32
        )

        target_tokens = encode_sentence(
            sample["sentence"],
            self.vocab
        )

        target_tokens = torch.tensor(
            target_tokens,
            dtype=torch.long
        )

        return {
            "features": features,
            "target_tokens": target_tokens,
            "sentence": sample["sentence"],
            "sample_key": sample["sample_key"],
            "n_frames": sample["n_frames"],
        }


if __name__ == "__main__":

    dataset = How2SignTorchDataset(
        "data/how2sign_landmarks",
        split="train",
        vocab_file="data/how2sign_landmarks/vocab.json"
    )

    print("Dataset size:", len(dataset))

    sample = dataset[1]

    print("\nSample:")
    print("Sample key:", sample["sample_key"])
    print("Features shape:", sample["features"].shape)
    print("Features dtype:", sample["features"].dtype)
    print("Target tokens:", sample["target_tokens"])
    print("Target shape:", sample["target_tokens"].shape)
    print("Frames:", sample["n_frames"])
    print("Sentence:", sample["sentence"])