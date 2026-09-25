import pandas as pd
import numpy as np
from pathlib import Path


DATA_DIR = Path("data/how2sign_landmarks")


def analyze_split(df, split):
    lengths = df[df["split"] == split]["n_frames"].to_numpy()

    print(f"\n{split.upper()}")
    print("-" * 40)
    print(f"Samples:   {len(lengths):,}")
    print(f"Min:       {lengths.min()}")
    print(f"Max:       {lengths.max()}")
    print(f"Mean:      {lengths.mean():.2f}")
    print(f"Median:    {np.median(lengths):.2f}")
    print(f"90th pct:  {np.percentile(lengths, 90):.0f}")
    print(f"95th pct:  {np.percentile(lengths, 95):.0f}")
    print(f"99th pct:  {np.percentile(lengths, 99):.0f}")

    for limit in [128, 160, 192, 256, 320, 384, 512]:
        percentage = (lengths > limit).mean() * 100
        print(
            f"> {limit:3d} frames: {percentage:6.2f}% "
            f"({(lengths > limit).sum():,} samples)"
        )


def main():
    metadata_path = DATA_DIR / "metadata.parquet"

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Metadata not found: {metadata_path}"
        )

    df = pd.read_parquet(metadata_path)

    # Only analyze samples whose shard exists locally.
    df = df[
        df["shard_file"].apply(
            lambda x: (DATA_DIR / x).exists()
        )
    ].reset_index(drop=True)

    print(f"Total local samples: {len(df):,}")

    analyze_split(df, "train")
    analyze_split(df, "val")
    analyze_split(df, "test")


if __name__ == "__main__":
    main()