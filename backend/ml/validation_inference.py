import torch
from torch.utils.data import DataLoader

from train_dataset import How2SignTorchDataset
from collate import collate_fn
from seq2seq_model import SignToSentenceModel
from tokenizer import load_vocabulary


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = "data/how2sign_landmarks"
VOCAB_FILE = "data/how2sign_landmarks/vocab.json"
CHECKPOINT = "checkpoints_420/best.pt"

BATCH_SIZE = 1

# Maximum number of generated tokens.
MAX_GENERATION_LENGTH = 40


# ============================================================
# DEVICE
# ============================================================

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


# ============================================================
# TOKEN HELPERS
# ============================================================

def build_reverse_vocabulary(vocab):
    return {
        index: token
        for token, index in vocab.items()
    }


def decode_tokens(tokens, reverse_vocab):

    words = []

    for token_id in tokens:

        token = reverse_vocab.get(
            int(token_id),
            "<unk>"
        )

        if token == "<eos>":
            break

        if token in {"<pad>", "<sos>"}:
            continue

        words.append(token)

    # Basic punctuation cleanup.
    sentence = " ".join(words)

    sentence = sentence.replace(
        " .",
        "."
    )

    sentence = sentence.replace(
        " ,",
        ","
    )

    sentence = sentence.replace(
        " ?",
        "?"
    )

    sentence = sentence.replace(
        " !",
        "!"
    )

    sentence = sentence.replace(
        " ' ",
        "'"
    )

    return sentence


# ============================================================
# AUTOREGRESSIVE INFERENCE
# ============================================================

def translate(
    model,
    source,
    source_lengths,
    source_padding_mask,
    sos_token,
    eos_token,
):

    model.eval()

    with torch.no_grad():

        # ----------------------------------------------------
        # Encoder
        # ----------------------------------------------------

        encoder_outputs, encoder_hidden, encoder_cell = (
            model.encoder(
                source,
                source_lengths
            )
        )

        # ----------------------------------------------------
        # Prepare decoder state
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

        input_token = torch.tensor(
            [sos_token],
            dtype=torch.long,
            device=DEVICE
        )

        generated_tokens = [
            sos_token
        ]

        # ----------------------------------------------------
        # Generate one token at a time
        # ----------------------------------------------------

        for _ in range(
            MAX_GENERATION_LENGTH
        ):

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
            ).item()

            generated_tokens.append(
                next_token
            )

            # Stop when <eos> is generated.
            if next_token == eos_token:
                break

            input_token = torch.tensor(
                [next_token],
                dtype=torch.long,
                device=DEVICE
            )

    return generated_tokens


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("VISIBLEVOICE - VALIDATION INFERENCE")
    print("=" * 60)

    print("\nDevice:", DEVICE)

    # --------------------------------------------------------
    # Vocabulary
    # --------------------------------------------------------

    vocab = load_vocabulary(
        VOCAB_FILE
    )

    reverse_vocab = build_reverse_vocabulary(
        vocab
    )

    sos_token = vocab["<sos>"]
    eos_token = vocab["<eos>"]

    print(
        "Vocabulary size:",
        len(vocab)
    )

    print(
        "<sos> token:",
        sos_token
    )

    print(
        "<eos> token:",
        eos_token
    )

    # --------------------------------------------------------
    # Validation dataset
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

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    print("\nCreating model...")

    model = SignToSentenceModel(
        vocab_size=len(vocab),
        input_size=420
    ).to(DEVICE)

    # --------------------------------------------------------
    # Load checkpoint
    # --------------------------------------------------------

    print(
        "\nLoading checkpoint:"
    )

    print(
        CHECKPOINT
    )

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=DEVICE
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    print(
        "Checkpoint epoch:",
        checkpoint["epoch"]
    )

    print(
        "Checkpoint validation loss:",
        checkpoint["val_loss"]
    )

    print("\n" + "=" * 60)
    print("RUNNING VALIDATION INFERENCE")
    print("=" * 60)

    # --------------------------------------------------------
    # Test first 10 validation samples
    # --------------------------------------------------------

    for index, batch in enumerate(loader):

        if index >= 10:
            break

        source = batch[
            "features"
        ].to(DEVICE)

        source_lengths = batch[
            "lengths"
        ].to(DEVICE)

        source_padding_mask = batch[
            "padding_mask"
        ].to(DEVICE)

        actual_sentence = batch[
    "sentences"
][0]

        predicted_tokens = translate(
            model,
            source,
            source_lengths,
            source_padding_mask,
            sos_token,
            eos_token
        )

        predicted_sentence = decode_tokens(
            predicted_tokens,
            reverse_vocab
        )

        print("\n" + "-" * 60)

        print(
            f"Sample {index + 1}"
        )

        print(
            "\nACTUAL:"
        )

        print(
            actual_sentence
        )

        print(
            "\nPREDICTED:"
        )

        print(
            predicted_sentence
        )

    print("\n" + "=" * 60)
    print("VALIDATION INFERENCE COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()