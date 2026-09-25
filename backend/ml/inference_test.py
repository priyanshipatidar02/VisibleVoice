import torch
from torch.utils.data import DataLoader

from train_dataset import How2SignTorchDataset
from collate import collate_fn
from seq2seq_model import SignToSentenceModel
from tokenizer import decode_sentence


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = "data/how2sign_landmarks"
VOCAB_FILE = "data/how2sign_landmarks/vocab.json"

# IMPORTANT:
# This is the new 420-feature checkpoint.
CHECKPOINT = "checkpoints_420/best.pt"

BATCH_SIZE = 4
NUM_SAMPLES = 10
MAX_OUTPUT_LENGTH = 40

INPUT_FEATURES = 420


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# GREEDY DECODING
# ============================================================

def greedy_decode(
    model,
    source,
    source_lengths,
    source_padding_mask,
    sos_id,
    eos_id,
    max_length=40
):
    """
    Generate an English sentence autoregressively.

    The real target sentence is NOT provided to the decoder.
    """

    model.eval()

    with torch.no_grad():

        # ----------------------------------------------------
        # Encode sign-language sequence
        # ----------------------------------------------------

        encoder_outputs, encoder_hidden, encoder_cell = (
            model.encoder(
                source,
                source_lengths
            )
        )

        # ----------------------------------------------------
        # Prepare decoder states
        # ----------------------------------------------------

        hidden = model._prepare_decoder_state(
            encoder_hidden
        )

        cell = model._prepare_decoder_state(
            encoder_cell
        )

        # ----------------------------------------------------
        # Start with <sos>
        # ----------------------------------------------------

        input_token = torch.full(
            (source.shape[0],),
            sos_id,
            dtype=torch.long,
            device=source.device
        )

        generated_tokens = []

        finished = torch.zeros(
            source.shape[0],
            dtype=torch.bool,
            device=source.device
        )

        # ----------------------------------------------------
        # Generate one token at a time
        # ----------------------------------------------------

        for _ in range(max_length):

            prediction, hidden, cell, _ = (
                model.decoder(
                    input_token,
                    hidden,
                    cell,
                    encoder_outputs,
                    source_padding_mask
                )
            )

            next_token = prediction.argmax(
                dim=1
            )

            generated_tokens.append(
                next_token
            )

            finished = finished | (
                next_token == eos_id
            )

            input_token = next_token

            # Stop once all samples generated <eos>.
            if torch.all(finished):
                break

        if not generated_tokens:

            return torch.empty(
                (source.shape[0], 0),
                dtype=torch.long,
                device=source.device
            )

        return torch.stack(
            generated_tokens,
            dim=1
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("VISIBLEVOICE - 420 FEATURE VALIDATION INFERENCE")
    print("=" * 60)

    print("\nDevice:", DEVICE)

    print("Input features:", INPUT_FEATURES)

    print("Checkpoint:", CHECKPOINT)

    # --------------------------------------------------------
    # Load validation dataset
    # --------------------------------------------------------

    print("\nLoading validation dataset...")

    dataset = How2SignTorchDataset(
        DATA_DIR,
        split="val",
        vocab_file=VOCAB_FILE
    )

    print(
        "Validation samples:",
        len(dataset)
    )

    # --------------------------------------------------------
    # Select samples
    # --------------------------------------------------------

    indices = list(
        range(
            min(NUM_SAMPLES, len(dataset))
        )
    )

    selected_dataset = torch.utils.data.Subset(
        dataset,
        indices
    )

    loader = DataLoader(
        selected_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )

    # --------------------------------------------------------
    # Vocabulary
    # --------------------------------------------------------

    vocab_size = len(
        dataset.vocab
    )

    print(
        "Vocabulary size:",
        vocab_size
    )

    # --------------------------------------------------------
    # Create 420-feature model
    # --------------------------------------------------------

    print("\nCreating 420-feature model...")

    model = SignToSentenceModel(
        vocab_size=vocab_size,
        input_size=INPUT_FEATURES
    ).to(DEVICE)

    # --------------------------------------------------------
    # Load checkpoint
    # --------------------------------------------------------

    print("\nLoading checkpoint:")
    print(CHECKPOINT)

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=DEVICE
    )

    if "model_state_dict" in checkpoint:

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        if "epoch" in checkpoint:

            print(
                "Checkpoint epoch:",
                checkpoint["epoch"]
            )

        if "train_loss" in checkpoint:

            print(
                "Checkpoint training loss:",
                checkpoint["train_loss"]
            )

        if "val_loss" in checkpoint:

            print(
                "Checkpoint validation loss:",
                checkpoint["val_loss"]
            )

    else:

        model.load_state_dict(
            checkpoint
        )

    model.eval()

    print(
        "Checkpoint loaded successfully."
    )

    # --------------------------------------------------------
    # Vocabulary IDs
    # --------------------------------------------------------

    vocab = dataset.vocab

    sos_id = vocab["<sos>"]
    eos_id = vocab["<eos>"]

    # --------------------------------------------------------
    # Generate predictions
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("GENERATING PREDICTIONS")
    print("=" * 60)

    sample_number = 0

    for batch in loader:

        source = batch[
            "features"
        ].to(DEVICE)

        source_lengths = batch[
            "lengths"
        ].to(DEVICE)

        source_padding_mask = batch[
            "padding_mask"
        ].to(DEVICE)

        # ----------------------------------------------------
        # Verify 420 features
        # ----------------------------------------------------

        assert source.shape[-1] == INPUT_FEATURES, (
            f"Expected {INPUT_FEATURES} features, "
            f"got {source.shape[-1]}"
        )

        generated = greedy_decode(
            model,
            source,
            source_lengths,
            source_padding_mask,
            sos_id,
            eos_id,
            max_length=MAX_OUTPUT_LENGTH
        )

        # ----------------------------------------------------
        # Display predictions
        # ----------------------------------------------------

        for i in range(
            source.shape[0]
        ):

            sample_number += 1

            token_ids = generated[
                i
            ].tolist()

            prediction = decode_sentence(
                token_ids,
                vocab
            )

            actual = batch[
                "sentences"
            ][i]

            print(
                f"\nSample {sample_number}"
            )

            print("-" * 50)

            print("Actual:")
            print(actual)

            print("\nPredicted:")
            print(prediction)

            if sample_number >= NUM_SAMPLES:
                break

        if sample_number >= NUM_SAMPLES:
            break

    # --------------------------------------------------------
    # Done
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("420 FEATURE INFERENCE COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()