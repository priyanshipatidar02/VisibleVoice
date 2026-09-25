import sys
import json
from pathlib import Path

import torch


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# PROJECT IMPORTS
# ============================================================

from ml.dataset import How2SignDataset
from ml.seq2seq_model import SignToSentenceModel
from ml.train_dataset import (
    TOTAL_FEATURES,
    MAX_FRAMES,
    create_combined_features,
    temporal_sample,
)
from ml.tokenizer import encode_sentence


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "how2sign_landmarks"
)

CHECKPOINT = (
    PROJECT_ROOT
    / "checkpoints_422"
    / "latest.pt"
)

DEVICE = torch.device("cpu")

NUM_SAMPLES = 5

MAX_DECODE_LENGTH = 50


# ============================================================
# PREPARE ONE SAMPLE
# ============================================================

def prepare_source(dataset, index):

    sample = dataset.get_sample(index)

    # --------------------------------------------------------
    # EXACT SAME FEATURE CREATION USED DURING TRAINING
    # --------------------------------------------------------

    features = create_combined_features(
        sample["landmarks"],
        sample["geometric"],
        sample["valid_mask"],
    )

    # --------------------------------------------------------
    # EXACT SAME TEMPORAL SAMPLING USED DURING TRAINING
    # --------------------------------------------------------

    features = temporal_sample(
        features,
        max_frames=MAX_FRAMES,
    )

    # --------------------------------------------------------
    # Source tensor
    #
    # Shape:
    # (1, T, 422)
    # --------------------------------------------------------

    source = torch.tensor(
        features,
        dtype=torch.float32,
        device=DEVICE,
    ).unsqueeze(0)

    sequence_length = features.shape[0]

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # Encoder requires:
    #
    # lengths = [T]
    #
    # This is NOT the attention mask.
    # --------------------------------------------------------

    lengths = torch.tensor(
        [sequence_length],
        dtype=torch.long,
        device=DEVICE,
    )

    # --------------------------------------------------------
    # Attention requires:
    #
    # padding_mask:
    # True  = padded frame
    # False = real frame
    #
    # Because this diagnostic contains NO padding,
    # every value is False.
    #
    # Shape:
    # (1, T)
    # --------------------------------------------------------

    padding_mask = torch.zeros(
        (1, sequence_length),
        dtype=torch.bool,
        device=DEVICE,
    )

    return (
        source,
        lengths,
        padding_mask,
        sample,
    )


# ============================================================
# DECODE TOKEN IDS
# ============================================================

def decode_ids(token_ids, vocab):

    reverse_vocab = {
        index: token
        for token, index in vocab.items()
    }

    tokens = []

    for token_id in token_ids:

        token_id = int(token_id)

        token = reverse_vocab.get(
            token_id,
            "<unk>",
        )

        if token == "<eos>":
            break

        if token in {
            "<pad>",
            "<sos>",
        }:
            continue

        tokens.append(token)

    return " ".join(tokens)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("VISIBLEVOICE - DECODER DIAGNOSTIC")
    print("=" * 60)

    print(
        f"Device: {DEVICE}"
    )

    print(
        f"Checkpoint: {CHECKPOINT}"
    )

    # ========================================================
    # LOAD DATASET
    # ========================================================

    print("\nLoading dataset...")

    dataset = How2SignDataset(
        data_dir=DATA_DIR,
        split="train",
    )

    dataset.metadata = (
        dataset.metadata
        .iloc[:NUM_SAMPLES]
        .reset_index(drop=True)
    )

    print(
        f"Samples: {len(dataset)}"
    )

    # ========================================================
    # LOAD VOCABULARY
    # ========================================================

    print("\nLoading vocabulary...")

    vocab_path = (
        DATA_DIR
        / "vocab.json"
    )

    with open(
        vocab_path,
        "r",
        encoding="utf-8",
    ) as f:

        vocab_data = json.load(f)

    vocab = vocab_data["vocab"]

    print(
        f"Vocabulary size: {len(vocab)}"
    )

    # ========================================================
    # SPECIAL TOKEN IDS
    # ========================================================

    sos_id = vocab["<sos>"]
    eos_id = vocab["<eos>"]
    pad_id = vocab["<pad>"]
    unk_id = vocab["<unk>"]

    print("\nSpecial tokens:")

    print(
        f"SOS: {sos_id}"
    )

    print(
        f"EOS: {eos_id}"
    )

    print(
        f"PAD: {pad_id}"
    )

    print(
        f"UNK: {unk_id}"
    )

    # ========================================================
    # LOAD MODEL
    # ========================================================

    print("\nLoading model...")

    model = SignToSentenceModel(
        vocab_size=len(vocab),
        input_size=TOTAL_FEATURES,
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

    # ========================================================
    # DIAGNOSTIC HEADER
    # ========================================================

    print(
        "\n" + "=" * 60
    )

    print(
        "TEACHER-FORCED vs AUTOREGRESSIVE DECODING"
    )

    print(
        "=" * 60
    )

    # ========================================================
    # PROCESS SAMPLES
    # ========================================================

    with torch.no_grad():

        for sample_index in range(
            NUM_SAMPLES
        ):

            (
                source,
                lengths,
                padding_mask,
                sample,
            ) = prepare_source(
                dataset,
                sample_index,
            )

            sentence = sample["sentence"]

            print(
                "\n" + "-" * 60
            )

            print(
                f"Sample {sample_index + 1}"
            )

            print(
                f"TARGET: {sentence}"
            )

            print(
                f"Frames: {source.shape[1]}"
            )

            # =================================================
            # TOKENIZE TARGET
            # =================================================

            target_ids = encode_sentence(
                sentence,
                vocab,
            )

            target = torch.tensor(
                target_ids,
                dtype=torch.long,
                device=DEVICE,
            ).unsqueeze(0)

            # =================================================
            # ENCODER
            #
            # IMPORTANT:
            #
            # encoder gets LENGTHS
            # attention gets PADDING MASK
            # =================================================

            (
                encoder_outputs,
                encoder_hidden,
                encoder_cell,
            ) = model.encoder(
                source,
                lengths,
            )

            # =================================================
            # TEACHER-FORCED DECODING
            # =================================================

            hidden = model._prepare_decoder_state(
                encoder_hidden
            )

            cell = model._prepare_decoder_state(
                encoder_cell
            )

            teacher_predictions = []

            input_token = target[:, 0]

            teacher_correct = 0
            teacher_total = 0

            for timestep in range(
                1,
                target.shape[1],
            ):

                (
                    prediction,
                    hidden,
                    cell,
                    attention_weights,
                ) = model.decoder(
                    input_token,
                    hidden,
                    cell,
                    encoder_outputs,
                    padding_mask,
                )

                predicted_id = (
                    prediction.argmax(
                        dim=1
                    )
                )

                predicted_id_value = int(
                    predicted_id.item()
                )

                teacher_predictions.append(
                    predicted_id_value
                )

                # ------------------------------------------------
                # TRUE NEXT TOKEN
                # ------------------------------------------------

                actual_id = int(
                    target[
                        0,
                        timestep
                    ].item()
                )

                if (
                    predicted_id_value
                    == actual_id
                ):
                    teacher_correct += 1

                teacher_total += 1

                # ------------------------------------------------
                # TEACHER FORCING
                #
                # Give the decoder the REAL previous token.
                # ------------------------------------------------

                input_token = target[
                    :,
                    timestep
                ]

            # =================================================
            # AUTOREGRESSIVE DECODING
            # =================================================

            # Run encoder again so that this experiment starts
            # from the same source representation.

            (
                encoder_outputs,
                encoder_hidden,
                encoder_cell,
            ) = model.encoder(
                source,
                lengths,
            )

            hidden = model._prepare_decoder_state(
                encoder_hidden
            )

            cell = model._prepare_decoder_state(
                encoder_cell
            )

            autoregressive_predictions = []

            # Start with SOS.
            input_token = torch.tensor(
                [sos_id],
                dtype=torch.long,
                device=DEVICE,
            )

            for step in range(
                MAX_DECODE_LENGTH
            ):

                (
                    prediction,
                    hidden,
                    cell,
                    attention_weights,
                ) = model.decoder(
                    input_token,
                    hidden,
                    cell,
                    encoder_outputs,
                    padding_mask,
                )

                predicted_id = (
                    prediction.argmax(
                        dim=1
                    )
                )

                predicted_id_value = int(
                    predicted_id.item()
                )

                autoregressive_predictions.append(
                    predicted_id_value
                )

                # ------------------------------------------------
                # STOP AT EOS
                # ------------------------------------------------

                if (
                    predicted_id_value
                    == eos_id
                ):
                    break

                # ------------------------------------------------
                # AUTOREGRESSIVE:
                #
                # Feed MODEL prediction back into decoder.
                # ------------------------------------------------

                input_token = predicted_id

            # =================================================
            # DECODE RESULTS
            # =================================================

            teacher_text = decode_ids(
                teacher_predictions,
                vocab,
            )

            autoregressive_text = decode_ids(
                autoregressive_predictions,
                vocab,
            )

            # =================================================
            # PRINT RESULTS
            # =================================================

            print(
                "\nTEACHER-FORCED PREDICTIONS:"
            )

            print(
                teacher_text
            )

            print(
                "\nAUTOREGRESSIVE PREDICTIONS:"
            )

            print(
                autoregressive_text
            )

            # =================================================
            # TEACHER-FORCED ACCURACY
            # =================================================

            if teacher_total > 0:

                teacher_accuracy = (
                    teacher_correct
                    / teacher_total
                    * 100
                )

            else:

                teacher_accuracy = 0.0

            print(
                "\nTeacher-forced "
                "next-token accuracy: "
                f"{teacher_accuracy:.2f}%"
            )

            print(
                f"Correct tokens: "
                f"{teacher_correct}/"
                f"{teacher_total}"
            )

            print(
                "Autoregressive tokens generated: "
                f"{len(autoregressive_predictions)}"
            )

    # ========================================================
    # COMPLETE
    # ========================================================

    print(
        "\n" + "=" * 60
    )

    print(
        "DECODER DIAGNOSTIC COMPLETE"
    )

    print(
        "=" * 60
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()