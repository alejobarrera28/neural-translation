"""
LSTM cell implemented from scratch.

Implements the Long Short-Term Memory cell with forget, input, and output gates:
    f_t = σ(W_f @ [h_{t-1}, x_t] + b_f)  # Forget gate
    i_t = σ(W_i @ [h_{t-1}, x_t] + b_i)  # Input gate
    g_t = tanh(W_g @ [h_{t-1}, x_t] + b_g)  # Cell candidate
    o_t = σ(W_o @ [h_{t-1}, x_t] + b_o)  # Output gate
    c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t      # Cell state
    h_t = o_t ⊙ tanh(c_t)                # Hidden state

This solves the vanishing gradient problem by using gating mechanisms to
control information flow, allowing gradients to propagate through long sequences.

Reference: Hochreiter & Schmidhuber (1997) - "Long Short-Term Memory"
"""

import torch
import torch.nn as nn


class LSTMCell(nn.Module):
    """
    LSTM cell from scratch.

    Implements a single step of the LSTM computation with forget, input,
    and output gates to control information flow.

    The LSTM maintains two states:
        - c_t: Cell state (long-term memory)
        - h_t: Hidden state (short-term memory/output)
    """

    def __init__(self, input_dim: int, hidden_dim: int, bias: bool = True):
        """
        Initialize LSTM cell.

        Args:
            input_dim: Dimension of input features
            hidden_dim: Dimension of hidden state and cell state
            bias: Whether to include bias terms
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Forget gate: controls what to forget from cell state
        self.W_f = nn.Linear(input_dim + hidden_dim, hidden_dim, bias=bias)

        # Input gate: controls what new information to add to cell state
        self.W_i = nn.Linear(input_dim + hidden_dim, hidden_dim, bias=bias)

        # Cell candidate: new candidate values to add to cell state
        self.W_g = nn.Linear(input_dim + hidden_dim, hidden_dim, bias=bias)

        # Output gate: controls what to output from cell state
        self.W_o = nn.Linear(input_dim + hidden_dim, hidden_dim, bias=bias)

    def forward(
        self, x: torch.Tensor, hidden: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Single forward step of LSTM cell.

        Args:
            x: Input tensor (batch_size, input_dim)
            hidden: Tuple of (h_{t-1}, c_{t-1}) where:
                - h_{t-1}: Previous hidden state (batch_size, hidden_dim)
                - c_{t-1}: Previous cell state (batch_size, hidden_dim)

        Returns:
            Tuple of (h_t, c_t) where:
            - h_t: New hidden state (batch_size, hidden_dim)
            - c_t: New cell state (batch_size, hidden_dim)
        """
        h_prev, c_prev = hidden

        # Concatenate input and previous hidden state
        combined = torch.cat([h_prev, x], dim=1)  # (batch_size, hidden_dim + input_dim)

        # Compute gates
        f_t = torch.sigmoid(self.W_f(combined))  # Forget gate
        i_t = torch.sigmoid(self.W_i(combined))  # Input gate
        g_t = torch.tanh(self.W_g(combined))     # Cell candidate
        o_t = torch.sigmoid(self.W_o(combined))  # Output gate

        # Update cell state: forget old + remember new
        c_t = f_t * c_prev + i_t * g_t

        # Compute new hidden state
        h_t = o_t * torch.tanh(c_t)

        return h_t, c_t


class MultiLayerLSTM(nn.Module):
    """
    Multi-layer LSTM using stacked LSTMCell instances.

    Stacks multiple LSTM cells to create a deep recurrent network.
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
        Initialize multi-layer LSTM.

        Args:
            input_dim: Dimension of input features
            hidden_dim: Dimension of hidden state per layer
            num_layers: Number of stacked LSTM layers
            dropout: Dropout probability between layers (not applied to last layer)
            batch_first: If True, input is (batch, seq, feature). Otherwise (seq, batch, feature)
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.batch_first = batch_first

        # Create LSTM cells for each layer
        self.cells = nn.ModuleList()
        for layer in range(num_layers):
            # First layer takes input_dim, others take hidden_dim
            layer_input_dim = input_dim if layer == 0 else hidden_dim
            self.cells.append(LSTMCell(layer_input_dim, hidden_dim))

        # Dropout between layers (not after last layer)
        self.dropout = nn.Dropout(dropout) if dropout > 0 and num_layers > 1 else None

    def forward(
        self, x, hidden: tuple[torch.Tensor, torch.Tensor] = None
    ) -> tuple:
        """
        Forward pass through multi-layer LSTM.

        Args:
            x: Input tensor or PackedSequence
               - If batch_first=True: (batch_size, seq_len, input_dim)
               - If batch_first=False: (seq_len, batch_size, input_dim)
               - Can also be PackedSequence from pack_padded_sequence
            hidden: Initial hidden state tuple (h_0, c_0) where:
                   - h_0: (num_layers, batch_size, hidden_dim)
                   - c_0: (num_layers, batch_size, hidden_dim)
                   If None, initialized to zeros

        Returns:
            Tuple of (outputs, (h_n, c_n)) where:
            - outputs: All timestep outputs (or PackedSequence if input was packed)
              - If batch_first=True: (batch_size, seq_len, hidden_dim)
              - If batch_first=False: (seq_len, batch_size, hidden_dim)
            - h_n: Final hidden states (num_layers, batch_size, hidden_dim)
            - c_n: Final cell states (num_layers, batch_size, hidden_dim)
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
            h_0 = torch.zeros(
                self.num_layers, batch_size, self.hidden_dim,
                dtype=x.dtype, device=x.device
            )
            c_0 = torch.zeros(
                self.num_layers, batch_size, self.hidden_dim,
                dtype=x.dtype, device=x.device
            )
            hidden = (h_0, c_0)

        h_n, c_n = hidden

        # Store outputs for each timestep
        outputs = []

        # Process each timestep
        for t in range(seq_len):
            x_t = x[t]  # (batch_size, input_dim) or (batch_size, hidden_dim)

            # Process through each layer
            new_h = []
            new_c = []
            for layer in range(self.num_layers):
                # Get hidden and cell state for this layer
                h_prev = h_n[layer]  # (batch_size, hidden_dim)
                c_prev = c_n[layer]  # (batch_size, hidden_dim)

                # Apply LSTM cell
                h_new, c_new = self.cells[layer](x_t, (h_prev, c_prev))

                # Apply dropout between layers (not on last layer)
                if self.dropout is not None and layer < self.num_layers - 1:
                    x_t = self.dropout(h_new)
                else:
                    x_t = h_new

                new_h.append(h_new)
                new_c.append(c_new)

            # Stack hidden and cell states for all layers
            h_n = torch.stack(new_h, dim=0)  # (num_layers, batch_size, hidden_dim)
            c_n = torch.stack(new_c, dim=0)  # (num_layers, batch_size, hidden_dim)

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

        return outputs, (h_n, c_n)
