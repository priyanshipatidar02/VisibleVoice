import json
import numpy as np
from pathlib import Path

from dataset import How2SignDataset


DATA_DIR = "data/how2sign_landmarks"
STATS_FILE = "data/how2sign_landmarks/normalization_stats.json"


def load_normalization_stats():
    """
    Load normalization statistics calculated
    from the training dataset.
    """

    stats_path = Path(STATS_FILE)

    if not stats_path.exists():
        raise FileNotFoundError(
            f"Normalization stats not found: {stats_path}"
        )

    with open(stats_path, "r") as f:
        stats = json.load(f)

    mean = np.asarray(
        stats["mean"],
        dtype=np.float32
    )

    std = np.asarray(
        stats["std"],
        dtype=np.float32
    )

    return mean, std


def preprocess_geometric_features(geometric):
    """
    Normalize geometric features using global
    training-set statistics.

    Input:
        geometric: (T, 36)

    Output:
        processed: (T, 36)
    """

    geometric = np.asarray(
        geometric,
        dtype=np.float32
    )

    if geometric.ndim != 2:
        raise ValueError(
            f"Expected 2D array (T, 36), "
            f"got shape {geometric.shape}"
        )

    if geometric.shape[1] != 36:
        raise ValueError(
            f"Expected 36 features, "
            f"got {geometric.shape[1]}"
        )

    geometric = np.nan_to_num(
        geometric,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    mean, std = load_normalization_stats()

    processed = (
        geometric - mean
    ) / std

    return processed.astype(np.float32)


if __name__ == "__main__":

    dataset = How2SignDataset(
        DATA_DIR,
        split="train"
    )

    print("Training samples:", len(dataset))

    sample = dataset.get_sample(1)

    print("\nOriginal:")
    print("Sentence:", sample["sentence"])
    print("Shape:", sample["geometric"].shape)

    processed = preprocess_geometric_features(
        sample["geometric"]
    )

    print("\nAfter preprocessing:")
    print("Shape:", processed.shape)
    print("Dtype:", processed.dtype)
    print("Contains NaN:", np.isnan(processed).any())
    print("Contains Inf:", np.isinf(processed).any())

    print("\nFirst frame:")
    print(processed[0])