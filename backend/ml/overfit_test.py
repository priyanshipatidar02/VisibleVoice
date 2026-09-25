import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from train_dataset import How2SignTorchDataset
from collate import collate_fn
from seq2seq_model import SignToSentenceModel


DATA_DIR = "data/how2sign_landmarks"
VOCAB_FILE = "data/how2sign_landmarks/vocab.json"

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)


def main():

    print("Using device:", DEVICE)

    # --------------------------------------------------
    # Dataset
    # --------------------------------------------------

    full_dataset = How2SignTorchDataset(
        DATA_DIR,
        split="train",
        vocab_file=VOCAB_FILE
    )

    # Use only a tiny subset for debugging.
    subset_size = 32

    dataset = Subset(
        full_dataset,
        range(subset_size)
    )

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=collate_fn
    )

    print("Training samples:", subset_size)

    # --------------------------------------------------
    # Model
    # --------------------------------------------------

    vocab_size = len(
        full_dataset.vocab
    )

    model = SignToSentenceModel(
        vocab_size=vocab_size
    ).to(DEVICE)

    # --------------------------------------------------
    # Loss
    # --------------------------------------------------

    criterion = nn.CrossEntropyLoss(
        ignore_index=0
    )

    # --------------------------------------------------
    # Optimizer
    # --------------------------------------------------

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001
    )

    # --------------------------------------------------
    # Training
    # --------------------------------------------------

    epochs = 30

    print("\nStarting overfit test...\n")

    for epoch in range(1, epochs + 1):

        model.train()

        total_loss = 0.0

        for batch in loader:

            source = batch["features"].to(DEVICE)

            source_lengths = batch[
                "lengths"
            ].to(DEVICE)

            target = batch[
                "target_tokens"
            ].to(DEVICE)

            source_padding_mask = batch[
                "padding_mask"
            ].to(DEVICE)

            optimizer.zero_grad()

            output = model(
                source,
                source_lengths,
                target,
                source_padding_mask,
                teacher_forcing_ratio=1.0
            )

            # Ignore timestep 0 because it is <sos>.
            output_dim = output.shape[-1]

            output = output[
                :, 1:, :
            ].contiguous().view(
                -1,
                output_dim
            )

            target = target[
                :, 1:
            ].contiguous().view(
                -1
            )

            loss = criterion(
                output,
                target
            )

            loss.backward()

            # Prevent exploding gradients.
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            optimizer.step()

            total_loss += loss.item()

        average_loss = (
            total_loss / len(loader)
        )

        print(
            f"Epoch {epoch:02d}/{epochs} "
            f"- Loss: {average_loss:.4f}"
        )

    print("\nOverfit test completed.")


if __name__ == "__main__":
    main()