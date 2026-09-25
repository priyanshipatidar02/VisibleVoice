import sys
import json
from pathlib import Path

import numpy as np
import torch

# ============================================================
# PROJECT IMPORTS
# ============================================================

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

# We specifically want the three problematic samples
SAMPLE_INDICES = [2, 3, 4]


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("VISIBLEVOICE - FULL ENCODER OUTPUT DIAGNOSTIC")
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

    print(f"Training samples available: {len(dataset)}")

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
    # Extract full encoder outputs
    # --------------------------------------------------------

    encoder_outputs_list = []
    sentences = []
    lengths = []

    print("\nExtracting full encoder outputs...")

    with torch.no_grad():

        for index in SAMPLE_INDICES:

            sample = dataset.get_sample(index)

            # ------------------------------------------------
            # SAME preprocessing used during training
            # ------------------------------------------------

            features = create_combined_features(
                sample["landmarks"],
                sample["geometric"],
                sample["valid_mask"],
            )

            features = temporal_sample(
                features,
                max_frames=MAX_FRAMES,
            )

            source = torch.tensor(
                features,
                dtype=torch.float32,
            ).unsqueeze(0).to(DEVICE)

            source_length = torch.tensor(
                [features.shape[0]],
                dtype=torch.long,
                device=DEVICE,
            )

            # ------------------------------------------------
            # Encoder
            # ------------------------------------------------

            encoder_outputs, hidden, cell = model.encoder(
                source,
                source_length,
            )

            # Remove batch dimension
            #
            # Original:
            # (1, T, 256)
            #
            # Becomes:
            # (T, 256)
            #
            output = (
                encoder_outputs[0]
                .cpu()
                .numpy()
            )

            encoder_outputs_list.append(output)

            sentences.append(
                sample["sentence"]
            )

            lengths.append(
                features.shape[0]
            )

            print(
                f"\nSample {index + 1}"
            )

            print(
                f"Sentence: {sample['sentence']}"
            )

            print(
                f"Frames: {features.shape[0]}"
            )

            print(
                f"Encoder output shape: {output.shape}"
            )

            print(
                f"Mean: {output.mean():.6f}"
            )

            print(
                f"Std: {output.std():.6f}"
            )

            print(
                f"Min: {output.min():.6f}"
            )

            print(
                f"Max: {output.max():.6f}"
            )

    # ========================================================
    # Compare global encoder statistics
    # ========================================================

    print("\n" + "=" * 60)
    print("GLOBAL ENCODER OUTPUT DISTANCES")
    print("=" * 60)

    # Because sequences have different lengths, compare
    # their mean encoder representation across time.

    mean_representations = []

    for output in encoder_outputs_list:

        mean_representation = output.mean(
            axis=0
        )

        mean_representations.append(
            mean_representation
        )

    mean_representations = np.array(
        mean_representations,
        dtype=np.float32,
    )

    print(
        f"Mean representation shape: "
        f"{mean_representations.shape}"
    )

    for i in range(len(mean_representations)):

        for j in range(i + 1, len(mean_representations)):

            distance = np.linalg.norm(
                mean_representations[i]
                - mean_representations[j]
            )

            print(
                f"Sample {SAMPLE_INDICES[i] + 1} ↔ "
                f"Sample {SAMPLE_INDICES[j] + 1}: "
                f"{distance:.4f}"
            )

    # ========================================================
    # Compare temporal encoder outputs
    # ========================================================

    print("\n" + "=" * 60)
    print("TEMPORAL ENCODER OUTPUT DIFFERENCE")
    print("=" * 60)

    print(
        "Comparing the first 50 encoder frames "
        "of each sample."
    )

    NUM_COMPARE_FRAMES = 50

    for i in range(len(encoder_outputs_list)):

        for j in range(i + 1, len(encoder_outputs_list)):

            a = encoder_outputs_list[i]
            b = encoder_outputs_list[j]

            compare_length = min(
                NUM_COMPARE_FRAMES,
                a.shape[0],
                b.shape[0],
            )

            a_compare = a[:compare_length]
            b_compare = b[:compare_length]

            distance = np.linalg.norm(
                a_compare - b_compare
            )

            mean_difference = np.mean(
                np.abs(
                    a_compare - b_compare
                )
            )

            print(
                f"\nSample {SAMPLE_INDICES[i] + 1} ↔ "
                f"Sample {SAMPLE_INDICES[j] + 1}"
            )

            print(
                f"Compared frames: {compare_length}"
            )

            print(
                f"Euclidean distance: "
                f"{distance:.4f}"
            )

            print(
                f"Mean absolute difference: "
                f"{mean_difference:.6f}"
            )

    # ========================================================
    # Sentence information
    # ========================================================

    print("\n" + "=" * 60)
    print("SAMPLES BEING COMPARED")
    print("=" * 60)

    for i, index in enumerate(SAMPLE_INDICES):

        print(f"\nSample {index + 1}")
        print(f"Frames: {lengths[i]}")
        print(f"Sentence: {sentences[i]}")

    # ========================================================
    # Final interpretation helper
    # ========================================================

    print("\n" + "=" * 60)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 60)

    print(
        "\nInterpretation:"
    )

    print(
        "Large differences between encoder outputs suggest "
        "that the encoder is preserving input information."
    )

    print(
        "Very small differences suggest that different "
        "sentences are being mapped to similar representations."
    )

    print(
        "\nWe will use these numbers to decide whether "
        "the next investigation should focus on the "
        "encoder/features or the attention/decoder."
    )


if __name__ == "__main__":
    main()