import json
import numpy as np
from pathlib import Path

from dataset import How2SignDataset


DATA_DIR = "data/how2sign_landmarks"
OUTPUT_FILE = "data/how2sign_landmarks/normalization_stats.json"


def calculate_stats():

    dataset = How2SignDataset(
        DATA_DIR,
        split="train"
    )

    print("Training samples:", len(dataset))

    feature_sum = np.zeros(36, dtype=np.float64)
    feature_squared_sum = np.zeros(36, dtype=np.float64)

    total_frames = 0

    for index in range(len(dataset)):

        if index % 500 == 0:
            print(f"Processing {index}/{len(dataset)}")

        sample = dataset.get_sample(index)

        features = np.asarray(
            sample["geometric"],
            dtype=np.float64
        )

        # Replace invalid values
        features = np.nan_to_num(
            features,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

        feature_sum += features.sum(axis=0)
        feature_squared_sum += (features ** 2).sum(axis=0)

        total_frames += features.shape[0]

    mean = feature_sum / total_frames

    variance = (
        feature_squared_sum / total_frames
    ) - (mean ** 2)

    variance = np.maximum(variance, 0.0)

    std = np.sqrt(variance)

    std[std < 1e-6] = 1.0

    stats = {
        "num_training_samples": len(dataset),
        "num_training_frames": total_frames,
        "mean": mean.tolist(),
        "std": std.tolist(),
    }

    output_path = Path(OUTPUT_FILE)

    with open(output_path, "w") as f:
        json.dump(
            stats,
            f,
            indent=2
        )

    print("\nNormalization statistics saved to:")
    print(output_path)

    print("\nTotal training frames:", total_frames)
    print("Mean shape:", mean.shape)
    print("Std shape:", std.shape)


if __name__ == "__main__":
    calculate_stats()