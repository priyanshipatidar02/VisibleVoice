import torch
import torch.nn as nn


class Encoder(nn.Module):

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

        packed = nn.utils.rnn.pack_padded_sequence(
            x,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )

        packed_output, (hidden, cell) = self.lstm(
            packed
        )

        outputs, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output,
            batch_first=True
        )

        # outputs:
        # (B, T, hidden_size * 2)

        return outputs, hidden, cell


class Attention(nn.Module):

    def __init__(
        self,
        encoder_dim,
        decoder_dim,
    ):
        super().__init__()

        self.attention = nn.Linear(
            encoder_dim + decoder_dim,
            decoder_dim
        )

        self.energy = nn.Linear(
            decoder_dim,
            1,
            bias=False
        )

    def forward(
        self,
        decoder_hidden,
        encoder_outputs,
        padding_mask
    ):

        # decoder_hidden:
        # (B, decoder_dim)

        # encoder_outputs:
        # (B, T, encoder_dim)

        batch_size = encoder_outputs.shape[0]
        sequence_length = encoder_outputs.shape[1]

        decoder_hidden = decoder_hidden.unsqueeze(1)

        decoder_hidden = decoder_hidden.repeat(
            1,
            sequence_length,
            1
        )

        combined = torch.cat(
            (
                encoder_outputs,
                decoder_hidden
            ),
            dim=2
        )

        energy = torch.tanh(
            self.attention(combined)
        )

        scores = self.energy(
            energy
        ).squeeze(2)

        # Ignore padded encoder frames.
        scores = scores.masked_fill(
            padding_mask,
            -1e10
        )

        attention_weights = torch.softmax(
            scores,
            dim=1
        )

        context = torch.bmm(
            attention_weights.unsqueeze(1),
            encoder_outputs
        )

        context = context.squeeze(1)

        return context, attention_weights


class Decoder(nn.Module):

    def __init__(
        self,
        vocab_size,
        embedding_dim=256,
        encoder_dim=256,
        hidden_size=256,
        dropout=0.2,
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=0
        )

        self.lstm = nn.LSTM(
            input_size=embedding_dim + encoder_dim,
            hidden_size=hidden_size,
            batch_first=True
        )

        self.attention = Attention(
            encoder_dim,
            hidden_size
        )

        self.output_projection = nn.Linear(
            hidden_size + encoder_dim,
            vocab_size
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        input_token,
        hidden,
        cell,
        encoder_outputs,
        padding_mask
    ):

        # input_token:
        # (B,)

        embedded = self.embedding(
            input_token
        )

        embedded = self.dropout(
            embedded
        )

        # Use the last decoder layer's hidden state.
        decoder_hidden = hidden[-1]

        context, attention_weights = self.attention(
            decoder_hidden,
            encoder_outputs,
            padding_mask
        )

        lstm_input = torch.cat(
            (
                embedded,
                context
            ),
            dim=1
        )

        lstm_input = lstm_input.unsqueeze(1)

        output, (hidden, cell) = self.lstm(
            lstm_input,
            (hidden, cell)
        )

        output = output.squeeze(1)

        prediction_input = torch.cat(
            (
                output,
                context
            ),
            dim=1
        )

        prediction = self.output_projection(
            prediction_input
        )

        return (
            prediction,
            hidden,
            cell,
            attention_weights
        )


class SignToSentenceModel(nn.Module):

    def __init__(
        self,
        vocab_size,
        input_size=36,
        encoder_hidden_size=128,
        encoder_layers=2,
        decoder_hidden_size=256,
        embedding_dim=256,
        dropout=0.2,
    ):
        super().__init__()

        self.encoder = Encoder(
            input_size=input_size,
            hidden_size=encoder_hidden_size,
            num_layers=encoder_layers,
            dropout=dropout
        )

        encoder_dim = (
            encoder_hidden_size * 2
        )

        self.decoder = Decoder(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            encoder_dim=encoder_dim,
            hidden_size=decoder_hidden_size,
            dropout=dropout
        )

        self.encoder_dim = encoder_dim
        self.decoder_hidden_size = decoder_hidden_size

    def forward(
        self,
        source,
        source_lengths,
        target,
        source_padding_mask,
        teacher_forcing_ratio=0.5
    ):

        batch_size = source.shape[0]

        target_length = target.shape[1]

        vocab_size = (
            self.decoder.output_projection.out_features
        )

        outputs = torch.zeros(
            batch_size,
            target_length,
            vocab_size,
            device=source.device
        )

        encoder_outputs, encoder_hidden, encoder_cell = (
            self.encoder(
                source,
                source_lengths
            )
        )

        # The bidirectional encoder has twice the
        # hidden dimension of a single decoder layer.
        # Combine forward and backward states.

        hidden = self._prepare_decoder_state(
            encoder_hidden
        )

        cell = self._prepare_decoder_state(
            encoder_cell
        )

        # First token is <sos>.
        input_token = target[:, 0]

        for timestep in range(
            1,
            target_length
        ):

            prediction, hidden, cell, _ = (
                self.decoder(
                    input_token,
                    hidden,
                    cell,
                    encoder_outputs,
                    source_padding_mask
                )
            )

            outputs[:, timestep] = prediction

            teacher_force = (
                torch.rand(1).item()
                < teacher_forcing_ratio
            )

            best_prediction = prediction.argmax(
                dim=1
            )

            input_token = (
                target[:, timestep]
                if teacher_force
                else best_prediction
            )

        return outputs

    def _prepare_decoder_state(self, state):

        # Encoder state:
        # (encoder_layers * 2, B, encoder_hidden)

        # Decoder state needs:
        # (decoder_layers, B, decoder_hidden)

        batch_size = state.shape[1]

        state = state.view(
            -1,
            2,
            batch_size,
            state.shape[2]
        )

        # Take the final encoder layer.
        forward_state = state[-1, 0]
        backward_state = state[-1, 1]

        combined = torch.cat(
            (
                forward_state,
                backward_state
            ),
            dim=1
        )

        # Encoder combined size = 256.
        # Decoder hidden size = 256.
        combined = combined.unsqueeze(0)

        return combined


if __name__ == "__main__":

    VOCAB_SIZE = 6396

    model = SignToSentenceModel(
        vocab_size=VOCAB_SIZE
    )

    batch_size = 4
    sequence_length = 162
    target_length = 19

    source = torch.randn(
        batch_size,
        sequence_length,
        36
    )

    source_lengths = torch.tensor(
        [12, 62, 162, 88],
        dtype=torch.long
    )

    source_padding_mask = (
        torch.arange(sequence_length).unsqueeze(0)
        >= source_lengths.unsqueeze(1)
    )

    target = torch.randint(
        0,
        VOCAB_SIZE,
        (
            batch_size,
            target_length
        )
    )

    # Every target starts with <sos> = 1.
    target[:, 0] = 1

    output = model(
        source,
        source_lengths,
        target,
        source_padding_mask,
        teacher_forcing_ratio=1.0
    )

    print("Source shape:")
    print(source.shape)

    print("\nTarget shape:")
    print(target.shape)

    print("\nModel output shape:")
    print(output.shape)

    print("\nExpected:")
    print(
        f"(batch_size, target_length, vocab_size)"
    )