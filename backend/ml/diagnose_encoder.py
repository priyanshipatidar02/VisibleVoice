import sys
import json
from pathlib import Path

import numpy as np
import torch

# Allow imports from project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from ml.dataset import How2SignDataset
from ml.seq2seq_model import SignToSentenceModel
from ml.train_dataset import (
    TOTAL_FEATURES,
    MAX_FRAMES,
    create_combined_features,
    temporal_sample,
)


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = PROJECT_ROOT / "data" / "how2sign_landmarks"
CHECKPOINT = PROJECT_ROOT / "checkpoints_422" / "latest.pt"

DEVICE = torch.device("cpu")
NUM_SAMPLES = 5


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("VISIBLEVOICE - ENCODER DIAGNOSTIC")
    print("=" * 60)

    print(f"Device: {DEVICE}")
    print(f"Checkpoint: {CHECKPOINT}")

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    print("\nLoading dataset...")

    dataset = How2SignDataset(
        data_dir=DATA_DIR,
        split="train",
    )

    # Only inspect first NUM_SAMPLES samples
    dataset.metadata = (
        dataset.metadata
        .iloc[:NUM_SAMPLES]
        .reset_index(drop=True)
    )

    print(f"Samples: {len(dataset)}")

    # --------------------------------------------------------
    # Vocabulary
    # --------------------------------------------------------

    vocab_path = DATA_DIR / "vocab.json"

    with open(vocab_path, "r") as f:
        vocab_data = json.load(f)

    vocab = vocab_data["vocab"]

    print(f"Vocabulary size: {len(vocab)}")

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    print("\nLoading model...")

    model = SignToSentenceModel(
        input_size=TOTAL_FEATURES,
        vocab_size=len(vocab),
    )

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=DEVICE,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(DEVICE)
    model.eval()

    print("Model loaded.")

    # --------------------------------------------------------
    # Encoder representations
    # --------------------------------------------------------

    print("\nExtracting encoder representations...")

    representations = []
    lengths = []
    sentences = []

    with torch.no_grad():

        for i in range(NUM_SAMPLES):

            # ------------------------------------------------
            # Get raw sample
            # ------------------------------------------------

            sample = dataset.get_sample(i)

            # ------------------------------------------------
            # Create the SAME 422-feature representation
            # used during training
            # ------------------------------------------------

            features = create_combined_features(
                sample["landmarks"],
                sample["geometric"],
                sample["valid_mask"],
            )

            # ------------------------------------------------
            # Apply the SAME temporal sampling
            # used during training
            # ------------------------------------------------

            features = temporal_sample(
                features,
                max_frames=MAX_FRAMES,
            )

            # ------------------------------------------------
            # Convert to tensor
            #
            # Model expects:
            # (batch, frames, features)
            # ------------------------------------------------

            source = torch.tensor(
                features,
                dtype=torch.float32,
            ).unsqueeze(0).to(DEVICE)

            # ------------------------------------------------
            # Actual sequence length after temporal sampling
            # ------------------------------------------------

            source_length = torch.tensor(
                [features.shape[0]],
                dtype=torch.long,
                device=DEVICE,
            )

            # ------------------------------------------------
            # Run ONLY the encoder
            # ------------------------------------------------

            encoder_outputs, hidden, cell = model.encoder(
                source,
                source_length,
            )

            # ------------------------------------------------
            # Encoder hidden state
            #
            # hidden shape:
            #
            # [num_layers * directions, batch, hidden_size]
            #
            # For our 2-layer bidirectional LSTM:
            #
            # hidden[0] = layer 1 forward
            # hidden[1] = layer 1 backward
            # hidden[2] = layer 2 forward
            # hidden[3] = layer 2 backward
            #
            # We use the final layer.
            # ------------------------------------------------

            final_forward = hidden[-2, 0]
            final_backward = hidden[-1, 0]

            representation = torch.cat(
                [
                    final_forward,
                    final_backward,
                ],
                dim=0,
            )

            representation = (
                representation
                .cpu()
                .numpy()
            )

            representations.append(
                representation
            )

            lengths.append(
                features.shape[0]
            )

            sentences.append(
                sample["sentence"]
            )

            print(
                f"  Sample {i + 1}: "
                f"{features.shape[0]} frames → "
                f"representation {representation.shape}"
            )

    # --------------------------------------------------------
    # Convert representations to NumPy array
    # --------------------------------------------------------

    representations = np.array(
        representations,
        dtype=np.float32,
    )

    # --------------------------------------------------------
    # Representation shape
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("ENCODER REPRESENTATION SHAPE")
    print("=" * 60)

    print(
        f"Representations: {representations.shape}"
    )

    # --------------------------------------------------------
    # Pairwise Euclidean distances
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("PAIRWISE EUCLIDEAN DISTANCES")
    print("=" * 60)

    distances = []

    for i in range(NUM_SAMPLES):

        for j in range(i + 1, NUM_SAMPLES):

            distance = np.linalg.norm(
                representations[i]
                - representations[j]
            )

            distances.append(distance)

            print(
                f"Sample {i + 1} ↔ Sample {j + 1}: "
                f"{distance:.4f}"
            )

    # --------------------------------------------------------
    # Distance statistics
    # --------------------------------------------------------

    if distances:

        distances = np.array(
            distances,
            dtype=np.float32,
        )

        print("\nDistance statistics:")

        print(
            f"Mean distance:   "
            f"{distances.mean():.4f}"
        )

        print(
            f"Std distance:    "
            f"{distances.std():.4f}"
        )

        print(
            f"Minimum distance:"
            f" {distances.min():.4f}"
        )

        print(
            f"Maximum distance:"
            f" {distances.max():.4f}"
        )

    # --------------------------------------------------------
    # Sentence information
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("SAMPLES")
    print("=" * 60)

    for i in range(NUM_SAMPLES):

        print(f"\nSample {i + 1}")
        print(f"Frames: {lengths[i]}")
        print(f"Sentence: {sentences[i]}")

    # --------------------------------------------------------
    # Representation statistics
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("REPRESENTATION STATISTICS")
    print("=" * 60)

    print(
        f"Mean: {representations.mean():.6f}"
    )

    print(
        f"Std:  {representations.std():.6f}"
    )

    print(
        f"Min:  {representations.min():.6f}"
    )

    print(
        f"Max:  {representations.max():.6f}"
    )

    # --------------------------------------------------------
    # Final message
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()