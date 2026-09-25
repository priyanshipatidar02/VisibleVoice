import numpy as np
import torch
from torch.utils.data import Dataset

from dataset import How2SignDataset
from preprocessing import preprocess_geometric_features
from tokenizer import (
    load_vocabulary,
    encode_sentence,
)


# --------------------------------------------------
# Configuration
# --------------------------------------------------

MAX_FRAMES = 256

LANDMARK_FEATURES = 128 * 3
GEOMETRIC_FEATURES = 36

# One flag for left-hand presence
# One flag for right-hand presence
HAND_PRESENCE_FEATURES = 2

TOTAL_FEATURES = (
    LANDMARK_FEATURES
    + GEOMETRIC_FEATURES
    + HAND_PRESENCE_FEATURES
)


# --------------------------------------------------
# Temporal sampling
# --------------------------------------------------

def temporal_sample(features, max_frames=MAX_FRAMES):
    """
    Reduce very long sequences to max_frames
    while preserving temporal coverage.

    Shorter sequences are left unchanged.
    """

    num_frames = features.shape[0]

    if num_frames <= max_frames:
        return features

    indices = np.linspace(
        0,
        num_frames - 1,
        max_frames
    ).astype(np.int64)

    return features[indices]


# --------------------------------------------------
# Landmark preprocessing
# --------------------------------------------------

def preprocess_landmarks(landmarks):
    """
    Convert landmarks from:

        (T, 128, 3)

    to:

        (T, 384)

    Missing / invalid numerical values are replaced
    with zero.
    """

    landmarks = np.asarray(
        landmarks,
        dtype=np.float32
    )

    landmarks = np.nan_to_num(
        landmarks,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    landmarks = landmarks.reshape(
        landmarks.shape[0],
        -1
    )

    return landmarks


# --------------------------------------------------
# Hand presence
# --------------------------------------------------

def create_hand_presence_features(valid_mask):
    """
    Create two frame-level features:

        1. left hand present
        2. right hand present

    valid_mask shape:

        (T, 128)

    Landmark layout:

        0:33    -> pose
        33:54   -> left hand
        54:75   -> right hand
        75:128  -> face

    A hand is considered present when all of its
    21 landmarks are valid.
    """

    valid_mask = np.asarray(
        valid_mask,
        dtype=bool
    )

    left_hand = valid_mask[:, 33:54].all(axis=1)
    right_hand = valid_mask[:, 54:75].all(axis=1)

    hand_presence = np.stack(
        [
            left_hand.astype(np.float32),
            right_hand.astype(np.float32),
        ],
        axis=1
    )

    return hand_presence


# --------------------------------------------------
# Combined feature representation
# --------------------------------------------------

def create_combined_features(
    landmarks,
    geometric,
    valid_mask
):
    """
    Create the final frame representation:

        384 landmark coordinates
        + 36 geometric features
        + 2 hand-presence features

        = 422 features per frame
    """

    landmark_features = preprocess_landmarks(
        landmarks
    )

    geometric_features = preprocess_geometric_features(
        geometric
    )

    geometric_features = np.asarray(
        geometric_features,
        dtype=np.float32
    )

    hand_presence_features = create_hand_presence_features(
        valid_mask
    )

    # ----------------------------------------------
    # Shape validation
    # ----------------------------------------------

    if landmark_features.shape[0] != geometric_features.shape[0]:
        raise ValueError(
            "Landmark and geometric feature frame counts "
            "do not match."
        )

    if landmark_features.shape[0] != hand_presence_features.shape[0]:
        raise ValueError(
            "Landmark and hand-presence feature frame counts "
            "do not match."
        )

    combined = np.concatenate(
        [
            landmark_features,
            geometric_features,
            hand_presence_features,
        ],
        axis=1
    )

    if combined.shape[1] != TOTAL_FEATURES:
        raise ValueError(
            f"Expected {TOTAL_FEATURES} features, "
            f"got {combined.shape[1]}."
        )

    return combined.astype(np.float32)


# --------------------------------------------------
# PyTorch Dataset
# --------------------------------------------------

class How2SignTorchDataset(Dataset):

    def __init__(
        self,
        data_dir,
        split,
        vocab_file
    ):

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

        sample = self.dataset.get_sample(
            index
        )

        features = create_combined_features(
            landmarks=sample["landmarks"],
            geometric=sample["geometric"],
            valid_mask=sample["valid_mask"]
        )

        features = temporal_sample(
            features,
            max_frames=MAX_FRAMES
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


# --------------------------------------------------
# Verification
# --------------------------------------------------

if __name__ == "__main__":

    dataset = How2SignTorchDataset(
        "data/how2sign_landmarks",
        split="train",
        vocab_file="data/how2sign_landmarks/vocab.json"
    )

    print("Dataset size:", len(dataset))
    print("Expected feature dimension:", TOTAL_FEATURES)

    print("\nChecking first 5 samples...")

    for i in range(5):

        sample = dataset[i]

        features = sample["features"]

        print(
            f"\nSample {i + 1}"
        )

        print(
            "Key:",
            sample["sample_key"]
        )

        print(
            "Original frames:",
            sample["n_frames"]
        )

        print(
            "Processed shape:",
            tuple(features.shape)
        )

        print(
            "Contains NaN:",
            torch.isnan(features).any().item()
        )

        print(
            "Contains Inf:",
            torch.isinf(features).any().item()
        )

        print(
            "Min:",
            features.min().item()
        )

        print(
            "Max:",
            features.max().item()
        )

        assert features.shape[1] == TOTAL_FEATURES
        assert not torch.isnan(features).any()
        assert not torch.isinf(features).any()

    print(
        "\nDataset feature verification PASSED."
    )