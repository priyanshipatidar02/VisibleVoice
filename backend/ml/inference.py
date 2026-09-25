import torch
import numpy as np

from dataset import How2SignDataset
from tokenizer import load_vocabulary

from train_dataset import (
    create_combined_features,
    temporal_sample,
    TOTAL_FEATURES,
)

from seq2seq_model import SignToSentenceModel


# ============================================================
# CONFIG
# ============================================================

DATA_DIR = "data/how2sign_landmarks"
VOCAB_FILE = "data/how2sign_landmarks/vocab.json"
CHECKPOINT = "checkpoints_422/latest.pt"

NUM_SAMPLES = 10
MAX_FRAMES = 256

# IMPORTANT:
# We are testing the SAME 50 training samples used
# in our tiny overfit experiment.
TRAINING_SUBSET_SIZE = 50


DEVICE = torch.device("cpu")


# ============================================================
# LOAD VOCABULARY
# ============================================================

vocab = load_vocabulary(VOCAB_FILE)

id_to_token = {
    index: token
    for token, index in vocab.items()
}


# ============================================================
# LOAD TRAINING DATASET
# ============================================================

dataset = How2SignDataset(
    DATA_DIR,
    split="train"
)

# IMPORTANT:
# Match the exact 50 samples used during training.
dataset.metadata = (
    dataset.metadata
    .iloc[:TRAINING_SUBSET_SIZE]
    .reset_index(drop=True)
)


print("=" * 60)
print("VISIBLEVOICE - TRAINING SET INFERENCE")
print("=" * 60)

print(f"Device: {DEVICE}")
print(f"Training samples available: {len(dataset)}")
print(f"Testing first: {min(NUM_SAMPLES, len(dataset))} samples")
print(f"Vocabulary size: {len(vocab)}")


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading checkpoint...")

checkpoint = torch.load(
    CHECKPOINT,
    map_location=DEVICE
)

model = SignToSentenceModel(
    input_size=TOTAL_FEATURES,
    vocab_size=len(vocab),
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.to(DEVICE)
model.eval()

print("Model loaded successfully.")


# ============================================================
# TOKEN DECODER
# ============================================================

def decode_tokens(token_ids):

    words = []

    for token_id in token_ids:

        token = id_to_token.get(
            int(token_id),
            "<unk>"
        )

        if token in {"<pad>", "<sos>"}:
            continue

        if token == "<eos>":
            break

        words.append(token)

    return " ".join(words)


# ============================================================
# RUN INFERENCE
# ============================================================

print("\nRunning predictions...")
print("=" * 60)


for i in range(
    min(NUM_SAMPLES, len(dataset))
):

    sample = dataset.get_sample(i)

    features = create_combined_features(
        sample["landmarks"],
        sample["geometric"],
        sample["valid_mask"],
    )

    features = temporal_sample(
        features,
        max_frames=MAX_FRAMES
    )

    source = torch.tensor(
        features,
        dtype=torch.float32
    ).unsqueeze(0).to(DEVICE)

    source_lengths = torch.tensor(
        [source.shape[1]],
        dtype=torch.long,
        device=DEVICE
    )

    source_padding_mask = torch.zeros(
        (1, source.shape[1]),
        dtype=torch.bool,
        device=DEVICE
    )


    # ========================================================
    # AUTOREGRESSIVE DECODING
    # ========================================================

    with torch.no_grad():

        sos_id = vocab["<sos>"]

        max_decode_length = 40

        target = torch.full(
            (1, max_decode_length),
            fill_value=vocab["<pad>"],
            dtype=torch.long,
            device=DEVICE,
        )

        target[0, 0] = sos_id

        output = model(
            source,
            source_lengths,
            target=target,
            source_padding_mask=source_padding_mask,
            teacher_forcing_ratio=0.0,
        )


    # ========================================================
    # DECODE
    # ========================================================

    predicted_ids = output.argmax(
        dim=-1
    )[0].cpu().numpy()

    prediction = decode_tokens(
        predicted_ids
    )


    # ========================================================
    # PRINT RESULT
    # ========================================================

    print(f"\nSample {i + 1}")
    print("-" * 60)

    print("TARGET:")
    print(sample["sentence"])

    print("\nPREDICTED:")
    print(prediction)


print("\n" + "=" * 60)
print("TRAINING SET INFERENCE COMPLETE")
print("=" * 60)