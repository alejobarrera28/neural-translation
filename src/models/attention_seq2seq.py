"""
Attention-based Seq2Seq model for neural machine translation.

Implements Bahdanau (additive) attention mechanism that allows the decoder
to focus on different parts of the source sequence at each decoding step.
This solves the information bottleneck of encoding the entire source into
a single fixed-length vector.

Architecture:
    Encoder: Bidirectional LSTM that processes source sequence
    Attention: Bahdanau attention that computes context vectors
    Decoder: LSTM with attention that generates target sequence
    Context: Decoder attends to all encoder outputs dynamically

Key innovation: Instead of a single context vector (final encoder state),
the decoder computes a new context vector at each step by attending to
all encoder hidden states.

Reference: Bahdanau et al. (2015) - "Neural Machine Translation by Jointly Learning to Align and Translate"
           https://arxiv.org/abs/1409.0473
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from typing import Tuple

from src.models.cells.attention import BahdanauAttention


class AttentionEncoder(nn.Module):
    """
    Bidirectional LSTM encoder for attention-based seq2seq.

    Uses bidirectional LSTM to capture both forward and backward context.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_layers: int = 1,
        dropout: float = 0.1,
        pad_idx: int = 0,
    ):
        """
        Initialize encoder.

        Args:
            vocab_size: Size of source vocabulary
            embedding_dim: Dimension of word embeddings
            hidden_dim: Dimension of LSTM hidden state (per direction)
            num_layers: Number of LSTM layers
            dropout: Dropout probability
            pad_idx: Index of padding token
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)

        # Project bidirectional outputs to decoder dimension
        self.fc_hidden = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc_cell = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(
        self, src: torch.Tensor, src_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Encode source sequences.

        Args:
            src: Source sequences (batch_size, src_len)
            src_lengths: Actual lengths before padding (batch_size,)

        Returns:
            Tuple of (outputs, (hidden, cell)) where:
            - outputs: Encoder outputs (batch_size, src_len, hidden_dim * 2)
            - hidden: Final hidden state (num_layers, batch_size, hidden_dim)
            - cell: Final cell state (num_layers, batch_size, hidden_dim)
        """
        # Embed tokens
        embedded = self.dropout(self.embedding(src))  # (batch_size, src_len, embedding_dim)

        # Pack padded sequences
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, src_lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        # Process through bidirectional LSTM
        packed_outputs, (hidden, cell) = self.lstm(packed)

        # Unpack outputs
        outputs, _ = nn.utils.rnn.pad_packed_sequence(
            packed_outputs, batch_first=True
        )  # (batch_size, src_len, hidden_dim * 2)

        # hidden: (num_layers * 2, batch_size, hidden_dim)
        # cell: (num_layers * 2, batch_size, hidden_dim)

        # Combine forward and backward final states
        # Take the last layer's forward and backward states
        # Forward: hidden[-2, :, :], Backward: hidden[-1, :, :]
        hidden_fwd = hidden[-2, :, :]  # (batch_size, hidden_dim)
        hidden_bwd = hidden[-1, :, :]  # (batch_size, hidden_dim)
        cell_fwd = cell[-2, :, :]
        cell_bwd = cell[-1, :, :]

        # Concatenate and project to decoder dimension
        hidden_combined = torch.cat([hidden_fwd, hidden_bwd], dim=1)  # (batch_size, hidden_dim * 2)
        cell_combined = torch.cat([cell_fwd, cell_bwd], dim=1)

        hidden_decoder = torch.tanh(self.fc_hidden(hidden_combined))  # (batch_size, hidden_dim)
        cell_decoder = torch.tanh(self.fc_cell(cell_combined))

        # Expand to num_layers for decoder
        # (batch_size, hidden_dim) -> (num_layers, batch_size, hidden_dim)
        hidden_decoder = hidden_decoder.unsqueeze(0).repeat(self.num_layers, 1, 1)
        cell_decoder = cell_decoder.unsqueeze(0).repeat(self.num_layers, 1, 1)

        return outputs, (hidden_decoder, cell_decoder)


class AttentionDecoder(nn.Module):
    """
    LSTM decoder with Bahdanau attention.

    At each step, computes attention over encoder outputs to create
    a context vector, then combines it with the current input.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        encoder_hidden_dim: int,
        num_layers: int = 1,
        dropout: float = 0.1,
        pad_idx: int = 0,
        attention_dim: int = 256,
    ):
        """
        Initialize attention decoder.

        Args:
            vocab_size: Size of target vocabulary
            embedding_dim: Dimension of word embeddings
            hidden_dim: Dimension of decoder LSTM hidden state
            encoder_hidden_dim: Dimension of encoder outputs (hidden_dim * 2 for bidirectional)
            num_layers: Number of LSTM layers
            dropout: Dropout probability
            pad_idx: Index of padding token
            attention_dim: Dimension of attention layer
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.attention = BahdanauAttention(hidden_dim, encoder_hidden_dim, attention_dim)

        # LSTM takes concatenation of embedding and context vector
        self.lstm = nn.LSTM(
            embedding_dim + encoder_hidden_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )

        # Output projection takes concatenation of LSTM output, context, and embedding
        self.fc_out = nn.Linear(hidden_dim + encoder_hidden_dim + embedding_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        tgt: torch.Tensor,
        hidden: Tuple[torch.Tensor, torch.Tensor],
        encoder_outputs: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Decode with attention.

        Args:
            tgt: Target sequences (batch_size, tgt_len)
            hidden: Tuple of (h_0, c_0) initial states (num_layers, batch_size, hidden_dim)
            encoder_outputs: All encoder outputs (batch_size, src_len, encoder_hidden_dim)
            mask: Optional padding mask (batch_size, src_len)

        Returns:
            Tuple of (logits, (hidden, cell), attention_weights) where:
            - logits: Output logits (batch_size, tgt_len, vocab_size)
            - hidden: Updated hidden state (num_layers, batch_size, hidden_dim)
            - cell: Updated cell state (num_layers, batch_size, hidden_dim)
            - attention_weights: Attention weights (batch_size, tgt_len, src_len)
        """
        batch_size = tgt.size(0)
        tgt_len = tgt.size(1)

        # Embed tokens
        embedded = self.dropout(self.embedding(tgt))  # (batch_size, tgt_len, embedding_dim)

        # Store outputs and attention weights
        outputs = []
        attention_weights_all = []

        h_t, c_t = hidden

        # Process each timestep
        for t in range(tgt_len):
            # Get current input embedding
            embedded_t = embedded[:, t, :].unsqueeze(1)  # (batch_size, 1, embedding_dim)

            # Get decoder hidden state from top layer for attention
            decoder_hidden = h_t[-1, :, :]  # (batch_size, hidden_dim)

            # Compute attention
            context, attn_weights = self.attention(
                decoder_hidden, encoder_outputs, mask
            )  # context: (batch_size, encoder_hidden_dim)

            # Concatenate embedding and context
            lstm_input = torch.cat([embedded_t, context.unsqueeze(1)], dim=2)
            # (batch_size, 1, embedding_dim + encoder_hidden_dim)

            # LSTM step
            lstm_output, (h_t, c_t) = self.lstm(lstm_input, (h_t, c_t))
            # lstm_output: (batch_size, 1, hidden_dim)

            # Concatenate LSTM output, context, and embedding for prediction
            prediction_input = torch.cat(
                [lstm_output.squeeze(1), context, embedded_t.squeeze(1)], dim=1
            )  # (batch_size, hidden_dim + encoder_hidden_dim + embedding_dim)

            # Project to vocabulary
            output = self.fc_out(prediction_input)  # (batch_size, vocab_size)

            outputs.append(output)
            attention_weights_all.append(attn_weights)

        # Stack outputs
        logits = torch.stack(outputs, dim=1)  # (batch_size, tgt_len, vocab_size)
        attention_weights = torch.stack(attention_weights_all, dim=1)  # (batch_size, tgt_len, src_len)

        return logits, (h_t, c_t), attention_weights


class AttentionSeq2Seq(nn.Module):
    """
    Complete attention-based seq2seq model.

    Uses bidirectional LSTM encoder and attention-based LSTM decoder.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 256,
        hidden_dim: int = 512,
        num_layers: int = 2,
        dropout: float = 0.1,
        pad_idx: int = 0,
        attention_dim: int = 256,
    ):
        """
        Initialize attention seq2seq model.

        Args:
            vocab_size: Size of vocabulary (shared for src and tgt)
            embedding_dim: Dimension of word embeddings
            hidden_dim: Dimension of LSTM hidden states
            num_layers: Number of LSTM layers
            dropout: Dropout probability
            pad_idx: Index of padding token
            attention_dim: Dimension of attention layer
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.pad_idx = pad_idx

        self.encoder = AttentionEncoder(
            vocab_size, embedding_dim, hidden_dim, num_layers, dropout, pad_idx
        )
        self.decoder = AttentionDecoder(
            vocab_size,
            embedding_dim,
            hidden_dim,
            hidden_dim * 2,  # Encoder is bidirectional
            num_layers,
            dropout,
            pad_idx,
            attention_dim,
        )

    def create_mask(self, src: torch.Tensor) -> torch.Tensor:
        """
        Create padding mask for source sequences.

        Args:
            src: Source sequences (batch_size, src_len)

        Returns:
            Mask tensor (batch_size, src_len) - True for padding positions
        """
        return src == self.pad_idx

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass with teacher forcing.

        Args:
            src: Source sequences (batch_size, src_len)
            tgt: Target sequences (batch_size, tgt_len)
            src_lengths: Source lengths (batch_size,)

        Returns:
            Output logits (batch_size, tgt_len, vocab_size)
        """
        # Create padding mask
        mask = self.create_mask(src)

        # Encode source
        encoder_outputs, hidden = self.encoder(src, src_lengths)

        # Decode with attention
        logits, _, _ = self.decoder(tgt, hidden, encoder_outputs, mask)

        return logits

    def encode(
        self, src: torch.Tensor, src_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Encode source sequences.

        Args:
            src: Source sequences (batch_size, src_len)
            src_lengths: Source lengths (batch_size,)

        Returns:
            Tuple of (encoder_outputs, (hidden, cell))
        """
        return self.encoder(src, src_lengths)

    def generate(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """
        Generate predictions (for inference/beam search).

        Args:
            src: Source sequences (batch_size, src_len)
            tgt: Target sequences so far (batch_size, current_tgt_len)
            src_lengths: Source lengths (batch_size,)

        Returns:
            Output logits (batch_size, current_tgt_len, vocab_size)
        """
        # Create padding mask
        mask = self.create_mask(src)

        # Encode source
        encoder_outputs, hidden = self.encoder(src, src_lengths)

        # Decode target
        logits, _, _ = self.decoder(tgt, hidden, encoder_outputs, mask)

        return logits
