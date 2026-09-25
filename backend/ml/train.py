import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from train_dataset import How2SignTorchDataset
from collate import collate_fn
from seq2seq_model import SignToSentenceModel


# ============================================================
# CONFIG
# ============================================================

DATA_DIR = "data/how2sign_landmarks"
VOCAB_FILE = "data/how2sign_landmarks/vocab.json"
CHECKPOINT_DIR = "checkpoints_422"

# Tiny memorization experiment
TRAIN_SAMPLES = 5
VAL_SAMPLES = 5

BATCH_SIZE = 1

LEARNING_RATE = 0.001

EPOCHS = 50

TEACHER_FORCING_RATIO = 1.0

NUM_WORKERS = 0

MAX_FRAMES = 256

INPUT_FEATURES = 422


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device("cpu")


# ============================================================
# CHECKPOINT
# ============================================================

def save_checkpoint(
    model,
    optimizer,
    epoch,
    train_loss,
    val_loss,
    filename,
):

    os.makedirs(
        CHECKPOINT_DIR,
        exist_ok=True
    )

    filepath = os.path.join(
        CHECKPOINT_DIR,
        filename
    )

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "input_features": INPUT_FEATURES,
            "max_frames": MAX_FRAMES,
        },
        filepath,
    )

    print(
        f"\nCheckpoint saved to:\n{filepath}"
    )


# ============================================================
# TRAIN
# ============================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
):

    model.train()

    total_loss = 0.0

    for batch_index, batch in enumerate(loader):

        source = batch["features"].to(DEVICE)

        source_lengths = batch["lengths"].to(DEVICE)

        target = batch["target_tokens"].to(DEVICE)

        source_padding_mask = batch["padding_mask"].to(DEVICE)

        optimizer.zero_grad()

        output = model(
            source,
            source_lengths,
            target,
            source_padding_mask,
            teacher_forcing_ratio=TEACHER_FORCING_RATIO,
        )

        # ----------------------------------------------------
        # Ignore <sos> timestep.
        # ----------------------------------------------------

        output_dim = output.shape[-1]

        output = output[:, 1:, :].contiguous().view(
            -1,
            output_dim
        )

        target_for_loss = target[:, 1:].contiguous().view(
            -1
        )

        loss = criterion(
            output,
            target_for_loss
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()

        total_loss += loss.item()

        print(
            f"  Batch {batch_index + 1}/{len(loader)} "
            f"- Loss: {loss.item():.4f}"
        )

    return total_loss / len(loader)


# ============================================================
# VALIDATION
# ============================================================

def validate(
    model,
    loader,
    criterion,
):

    model.eval()

    total_loss = 0.0

    with torch.no_grad():

        for batch in loader:

            source = batch["features"].to(DEVICE)

            source_lengths = batch["lengths"].to(DEVICE)

            target = batch["target_tokens"].to(DEVICE)

            source_padding_mask = batch["padding_mask"].to(DEVICE)

            output = model(
                source,
                source_lengths,
                target,
                source_padding_mask,
                teacher_forcing_ratio=1.0,
            )

            output_dim = output.shape[-1]

            output = output[:, 1:, :].contiguous().view(
                -1,
                output_dim
            )

            target_for_loss = target[:, 1:].contiguous().view(
                -1
            )

            loss = criterion(
                output,
                target_for_loss
            )

            total_loss += loss.item()

    return total_loss / len(loader)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("VISIBLEVOICE - 5 SAMPLE MEMORIZATION TEST")
    print("=" * 60)

    print(f"\nDevice: {DEVICE}")

    print("\nConfiguration:")
    print(f"Input features: {INPUT_FEATURES}")
    print(f"Training samples: {TRAIN_SAMPLES}")
    print(f"Validation samples: {VAL_SAMPLES}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Teacher forcing: {TEACHER_FORCING_RATIO}")

    # ========================================================
    # DATASET
    # ========================================================

    print("\nLoading training dataset...")

    train_dataset = How2SignTorchDataset(
        DATA_DIR,
        split="train",
        vocab_file=VOCAB_FILE,
    )

    train_dataset.dataset.metadata = (
        train_dataset.dataset.metadata
        .iloc[:TRAIN_SAMPLES]
        .reset_index(drop=True)
    )

    print(
        "Training samples:",
        len(train_dataset)
    )

    print("\nLoading validation dataset...")

    val_dataset = How2SignTorchDataset(
        DATA_DIR,
        split="val",
        vocab_file=VOCAB_FILE,
    )

    val_dataset.dataset.metadata = (
        val_dataset.dataset.metadata
        .iloc[:VAL_SAMPLES]
        .reset_index(drop=True)
    )

    print(
        "Validation samples:",
        len(val_dataset)
    )

    # ========================================================
    # DATALOADERS
    # ========================================================

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
    )

    print(
        "\nTraining batches:",
        len(train_loader)
    )

    print(
        "Validation batches:",
        len(val_loader)
    )

    # ========================================================
    # MODEL
    # ========================================================

    vocab_size = len(
        train_dataset.vocab
    )

    print(
        "\nVocabulary size:",
        vocab_size
    )

    print("\nCreating model...")

    model = SignToSentenceModel(
        vocab_size=vocab_size,
        input_size=INPUT_FEATURES,
        encoder_hidden_size=128,
        encoder_layers=2,
        decoder_hidden_size=256,
        embedding_dim=256,
        dropout=0.2,
    )

    model = model.to(DEVICE)

    # ========================================================
    # LOSS + OPTIMIZER
    # ========================================================

    criterion = nn.CrossEntropyLoss(
        ignore_index=0
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    # ========================================================
    # TRAINING
    # ========================================================

    best_val_loss = float("inf")

    print("\nStarting training...")

    for epoch in range(1, EPOCHS + 1):

        print("\n" + "=" * 60)

        print(
            f"Epoch {epoch}/{EPOCHS}"
        )

        print("=" * 60)

        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
        )

        print(
            f"\nTraining Loss: {train_loss:.4f}"
        )

        print("\nRunning validation...")

        val_loss = validate(
            model,
            val_loader,
            criterion,
        )

        print(
            f"Validation Loss: {val_loss:.4f}"
        )

        save_checkpoint(
            model,
            optimizer,
            epoch,
            train_loss,
            val_loss,
            "latest.pt",
        )

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            save_checkpoint(
                model,
                optimizer,
                epoch,
                train_loss,
                val_loss,
                "best.pt",
            )

            print(
                "New best validation model!"
            )

    # ========================================================
    # COMPLETE
    # ========================================================

    print("\n" + "=" * 60)

    print(
        "5-SAMPLE MEMORIZATION TEST COMPLETE"
    )

    print("=" * 60)

    print(
        f"\nBest validation loss: "
        f"{best_val_loss:.4f}"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()