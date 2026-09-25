from pathlib import Path
import subprocess
import pandas as pd


REPO_ID = "martinctl/how2sign-asl-landmarks"

DATA_DIR = Path("data/how2sign_landmarks")
SHARDS_DIR = DATA_DIR / "shards"

TARGETS = {
    "train": 5000,
    "val": 500,
    "test": 500,
}


def get_required_shards(metadata, split, target_samples):
    split_df = metadata[metadata["split"] == split]

    shard_counts = (
        split_df.groupby("shard_file")
        .size()
        .reset_index(name="samples")
    )

    selected = []
    total = 0

    for _, row in shard_counts.iterrows():

        shard_file = row["shard_file"]
        samples = int(row["samples"])

        selected.append(shard_file)
        total += samples

        if total >= target_samples:
            break

    return selected, total


def download_shard(shard_file):

    filename = Path(shard_file).name
    local_path = SHARDS_DIR / filename

    if local_path.exists():
        print(f"Already exists: {filename}")
        return

    print(f"\nDownloading: {filename}")

    command = [
        "hf",
        "download",
        REPO_ID,
        "--repo-type",
        "dataset",
        "--include",
        shard_file,
        "--local-dir",
        str(DATA_DIR),
    ]

    subprocess.run(command, check=True)


def main():

    SHARDS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    metadata_path = DATA_DIR / "metadata.parquet"

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Metadata not found: {metadata_path}"
        )

    metadata = pd.read_parquet(metadata_path)

    all_selected = []

    for split, target in TARGETS.items():

        selected, estimated_samples = get_required_shards(
            metadata,
            split,
            target
        )

        print("\n" + "=" * 60)
        print(f"{split.upper()}")
        print("=" * 60)

        print(f"Target samples: {target}")
        print(f"Selected shards: {len(selected)}")
        print(f"Samples represented: {estimated_samples}")

        all_selected.extend(selected)

    # Remove duplicates while preserving order.
    all_selected = list(dict.fromkeys(all_selected))

    print("\n" + "=" * 60)
    print("DOWNLOAD PLAN")
    print("=" * 60)

    print(f"Total unique shards: {len(all_selected)}")

    answer = input(
        "\nStart downloading these shards? [y/N]: "
    ).strip().lower()

    if answer != "y":
        print("Download cancelled.")
        return

    for shard_file in all_selected:
        download_shard(shard_file)

    print("\nAll requested shards downloaded.")


if __name__ == "__main__":
    main()