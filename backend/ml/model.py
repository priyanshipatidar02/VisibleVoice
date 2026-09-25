import torch
import torch.nn as nn


class TemporalEncoder(nn.Module):

    def __init__(
        self,
        input_size=36,
        hidden_size=128,
        num_layers=2,
        dropout=0.2,
    ):
        super().__init__()

        self.input_projection = nn.Linear(
            input_size,
            hidden_size
        )

        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.output_size = hidden_size * 2

    def forward(self, x, lengths):

        # x: (B, T, 36)

        x = self.input_projection(x)

        # Pack variable-length sequences so the LSTM
        # does not process padded frames.
        packed = nn.utils.rnn.pack_padded_sequence(
            x,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )

        packed_output, _ = self.lstm(packed)

        # Convert back to padded representation.
        output, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output,
            batch_first=True
        )

        # output: (B, max_T, hidden_size * 2)
        return output


if __name__ == "__main__":

    model = TemporalEncoder()

    # Example batch:
    # 4 sequences, padded to 162 frames
    x = torch.randn(
        4,
        162,
        36
    )

    lengths = torch.tensor(
        [162, 120, 90, 60],
        dtype=torch.long
    )

    output = model(
        x,
        lengths
    )

    print("Input shape:")
    print(x.shape)

    print("\nSequence lengths:")
    print(lengths)

    print("\nOutput shape:")
    print(output.shape)

    print("\nEncoder output size:")
    print(model.output_size)