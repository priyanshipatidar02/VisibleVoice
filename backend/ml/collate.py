import torch
from torch.nn.utils.rnn import pad_sequence


def collate_fn(batch):
    """
    Combine variable-length input sequences and
    variable-length target sentences.

    Input features:
        (T, 36)

    Target tokens:
        (target_length,)

    Output:

        features:
            (B, max_T, 36)

        padding_mask:
            (B, max_T)

        lengths:
            (B,)

        target_tokens:
            (B, max_target_length)

        target_padding_mask:
            (B, max_target_length)

        target_lengths:
            (B,)
    """

    # --------------------------------------------------
    # Input features
    # --------------------------------------------------

    features = [
        sample["features"]
        for sample in batch
    ]

    lengths = torch.tensor(
        [x.shape[0] for x in features],
        dtype=torch.long
    )

    padded_features = pad_sequence(
        features,
        batch_first=True,
        padding_value=0.0
    )

    max_length = padded_features.shape[1]

    padding_mask = (
        torch.arange(max_length).unsqueeze(0)
        >= lengths.unsqueeze(1)
    )

    # --------------------------------------------------
    # Target sentences
    # --------------------------------------------------

    target_tokens = [
        sample["target_tokens"]
        for sample in batch
    ]

    target_lengths = torch.tensor(
        [x.shape[0] for x in target_tokens],
        dtype=torch.long
    )

    # <pad> token ID is 0.
    padded_targets = pad_sequence(
        target_tokens,
        batch_first=True,
        padding_value=0
    )

    max_target_length = padded_targets.shape[1]

    target_padding_mask = (
        torch.arange(max_target_length).unsqueeze(0)
        >= target_lengths.unsqueeze(1)
    )

    # --------------------------------------------------
    # Other information
    # --------------------------------------------------

    sentences = [
        sample["sentence"]
        for sample in batch
    ]

    sample_keys = [
        sample["sample_key"]
        for sample in batch
    ]

    return {
        "features": padded_features,
        "padding_mask": padding_mask,
        "lengths": lengths,

        "target_tokens": padded_targets,
        "target_padding_mask": target_padding_mask,
        "target_lengths": target_lengths,

        "sentences": sentences,
        "sample_keys": sample_keys,
    }


if __name__ == "__main__":

    from train_dataset import How2SignTorchDataset
    from torch.utils.data import DataLoader

    dataset = How2SignTorchDataset(
        "data/how2sign_landmarks",
        split="train",
        vocab_file="data/how2sign_landmarks/vocab.json"
    )

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=collate_fn
    )

    batch = next(iter(loader))

    print("Input features shape:")
    print(batch["features"].shape)

    print("\nInput padding mask shape:")
    print(batch["padding_mask"].shape)

    print("\nInput sequence lengths:")
    print(batch["lengths"])

    print("\nTarget tokens shape:")
    print(batch["target_tokens"].shape)

    print("\nTarget padding mask shape:")
    print(batch["target_padding_mask"].shape)

    print("\nTarget sequence lengths:")
    print(batch["target_lengths"])

    print("\nSentences:")

    for sentence in batch["sentences"]:
        print("-", sentence)

    print("\nSample keys:")

    for key in batch["sample_keys"]:
        print("-", key)