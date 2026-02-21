"""
Vanilla RNN cell implemented from scratch.

Implements the basic recurrent neural network cell with tanh activation:
    h_t = tanh(W_ih @ x_t + W_hh @ h_{t-1} + b)

This is the foundational recurrent unit, though it suffers from vanishing
gradients for long sequences (solved by LSTM/GRU).
"""

import torch
import torch.nn as nn


class RNNCell(nn.Module):
    """
    Vanilla RNN cell from scratch.

    Implements a single step of the recurrent computation:
        h_t = tanh(W_ih @ x_t + W_hh @ h_{t-1} + b)

    where:
        x_t: input at timestep t
        h_{t-1}: hidden state from previous timestep
        h_t: new hidden state
    """

    def __init__(self, input_dim: int, hidden_dim: int, bias: bool = True):
        """
        Initialize RNN cell.

        Args:
            input_dim: Dimension of input features
            hidden_dim: Dimension of hidden state
            bias: Whether to include bias terms
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Input-to-hidden transformation
        self.W_ih = nn.Linear(input_dim, hidden_dim, bias=bias)

        # Hidden-to-hidden transformation
        self.W_hh = nn.Linear(hidden_dim, hidden_dim, bias=bias)

    def forward(self, x: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        """
        Single forward step of RNN cell.

        Args:
            x: Input tensor (batch_size, input_dim)
            hidden: Previous hidden state (batch_size, hidden_dim)

        Returns:
            New hidden state (batch_size, hidden_dim)
        """
        # h_t = tanh(W_ih @ x_t + W_hh @ h_{t-1})
        h_new = torch.tanh(self.W_ih(x) + self.W_hh(hidden))

        return h_new


class MultiLayerRNN(nn.Module):
    """
    Multi-layer RNN using stacked RNNCell instances.

    Stacks multiple RNN cells to create a deep recurrent network.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        batch_first: bool = True,
    ):
        """
        Initialize multi-layer RNN.

        Args:
            input_dim: Dimension of input features
            hidden_dim: Dimension of hidden state per layer
            num_layers: Number of stacked RNN layers
            dropout: Dropout probability between layers (not applied to last layer)
            batch_first: If True, input is (batch, seq, feature). Otherwise (seq, batch, feature)
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.batch_first = batch_first

        # Create RNN cells for each layer
        self.cells = nn.ModuleList()
        for layer in range(num_layers):
            # First layer takes input_dim, others take hidden_dim
            layer_input_dim = input_dim if layer == 0 else hidden_dim
            self.cells.append(RNNCell(layer_input_dim, hidden_dim))

        # Dropout between layers (not after last layer)
        self.dropout = nn.Dropout(dropout) if dropout > 0 and num_layers > 1 else None

    def forward(
        self, x, hidden: torch.Tensor = None
    ) -> tuple:
        """
        Forward pass through multi-layer RNN.

        Args:
            x: Input tensor or PackedSequence
               - If batch_first=True: (batch_size, seq_len, input_dim)
               - If batch_first=False: (seq_len, batch_size, input_dim)
               - Can also be PackedSequence from pack_padded_sequence
            hidden: Initial hidden state (num_layers, batch_size, hidden_dim)
                   If None, initialized to zeros

        Returns:
            Tuple of (outputs, hidden) where:
            - outputs: All timestep outputs (or PackedSequence if input was packed)
              - If batch_first=True: (batch_size, seq_len, hidden_dim)
              - If batch_first=False: (seq_len, batch_size, hidden_dim)
            - hidden: Final hidden states (num_layers, batch_size, hidden_dim)
        """
        # Handle PackedSequence
        is_packed = isinstance(x, nn.utils.rnn.PackedSequence)
        if is_packed:
            # Unpack: automatically restores original batch order via stored unsorted_indices
            # seq_lengths will be in original order (before any pack_padded_sequence sorting)
            x, seq_lengths = nn.utils.rnn.pad_packed_sequence(x, batch_first=self.batch_first)

        # Handle batch_first convention
        if self.batch_first:
            batch_size, seq_len, _ = x.size()
            x = x.transpose(0, 1)  # Convert to (seq_len, batch_size, input_dim)
        else:
            seq_len, batch_size, _ = x.size()

        # Initialize hidden state if not provided
        if hidden is None:
            hidden = torch.zeros(
                self.num_layers, batch_size, self.hidden_dim,
                dtype=x.dtype, device=x.device
            )

        # Store outputs for each timestep
        outputs = []

        # Process each timestep
        for t in range(seq_len):
            x_t = x[t]  # (batch_size, input_dim) or (batch_size, hidden_dim)

            # Process through each layer
            new_hidden = []
            for layer in range(self.num_layers):
                # Get hidden state for this layer
                h_prev = hidden[layer]  # (batch_size, hidden_dim)

                # Apply RNN cell
                h_new = self.cells[layer](x_t, h_prev)

                # Apply dropout between layers (not on last layer)
                if self.dropout is not None and layer < self.num_layers - 1:
                    x_t = self.dropout(h_new)
                else:
                    x_t = h_new

                new_hidden.append(h_new)

            # Stack hidden states for all layers
            hidden = torch.stack(new_hidden, dim=0)  # (num_layers, batch_size, hidden_dim)

            # Save output from top layer
            outputs.append(x_t)

        # Stack outputs across time
        outputs = torch.stack(outputs, dim=0)  # (seq_len, batch_size, hidden_dim)

        # Convert back to batch_first if needed
        if self.batch_first:
            outputs = outputs.transpose(0, 1)  # (batch_size, seq_len, hidden_dim)

        # Re-pack if input was packed
        if is_packed:
            # Re-pack using restored seq_lengths (already in original order from pad_packed_sequence)
            # enforce_sorted=False allows unsorted batches
            outputs = nn.utils.rnn.pack_padded_sequence(
                outputs, seq_lengths.cpu(), batch_first=self.batch_first, enforce_sorted=False
            )

        return outputs, hidden
