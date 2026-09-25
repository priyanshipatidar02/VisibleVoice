import numpy as np
import pandas as pd
from pathlib import Path


class How2SignDataset:

    def __init__(self, data_dir, split=None):
        self.data_dir = Path(data_dir)

        metadata_path = self.data_dir / "metadata.parquet"

        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Metadata file not found: {metadata_path}"
            )

        self.metadata = pd.read_parquet(metadata_path)

        # Only keep samples whose landmark shards exist locally.
        self.metadata = self.metadata[
            self.metadata["shard_file"].apply(
                lambda x: (self.data_dir / x).exists()
            )
        ].reset_index(drop=True)
        if split is not None:
            if split not in {"train", "val", "test"}:
              raise ValueError(
            "split must be one of: train, val, test"
        )

        self.metadata = self.metadata[
        self.metadata["split"] == split
        ].reset_index(drop=True)

        self._shards = {}

    def __len__(self):
        return len(self.metadata)

    def _load_shard(self, shard_file):
        if shard_file not in self._shards:
            path = self.data_dir / shard_file
            self._shards[shard_file] = np.load(
                path,
                allow_pickle=True
            )

        return self._shards[shard_file]

    def get_sample(self, index):
        row = self.metadata.iloc[index]

        sample_key = row["sample_key"]
        shard_file = row["shard_file"]

        shard = self._load_shard(shard_file)

        landmark_key = f"{sample_key}__landmarks_image"
        geometric_key = f"{sample_key}__features_geometric"
        mask_key = f"{sample_key}__valid_mask"

        landmarks = shard[landmark_key]
        geometric = shard[geometric_key]
        valid_mask = shard[mask_key]

        sentence = row["sentence"]

        return {
            "sample_key": sample_key,
            "landmarks": landmarks,
            "geometric": geometric,
            "valid_mask": valid_mask,
            "sentence": sentence,
            "n_frames": int(row["n_frames"]),
        }


if __name__ == "__main__":

    data_dir = "data/how2sign_landmarks"

    dataset = How2SignDataset(data_dir)

    print("Samples available locally:", len(dataset))

    if len(dataset) > 0:

        sample = dataset.get_sample(0)

        print("\nFirst sample:")
        print("Sample key:", sample["sample_key"])
        print("Landmarks shape:", sample["landmarks"].shape)
        print("Geometric shape:", sample["geometric"].shape)
        print("Valid mask shape:", sample["valid_mask"].shape)
        print("Number of frames:", sample["n_frames"])
        print("Sentence:", sample["sentence"])