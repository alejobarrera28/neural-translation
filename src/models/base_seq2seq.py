"""
Base encoder-decoder framework for sequence-to-sequence models.

Provides abstract classes for encoder, decoder, and complete seq2seq models
that can be instantiated with different recurrent cell types (RNN, LSTM, etc.).
"""

import torch
import torch.nn as nn
from typing import Tuple


class BaseEncoder(nn.Module):
    """
    Generic encoder that works with any recurrent cell.

    Embeds source sequences and processes them through a recurrent network.
    """

    def __init__(
        self,
        cell_module: nn.Module,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_layers: int = 1,
        dropout: float = 0.1,
        pad_idx: int = 0,
    ):
        """
        Initialize base encoder.

        Args:
            cell_module: Recurrent cell module (e.g., MultiLayerRNN)
            vocab_size: Size of source vocabulary
            embedding_dim: Dimension of word embeddings
            hidden_dim: Dimension of recurrent hidden state
            num_layers: Number of recurrent layers
            dropout: Dropout probability
            pad_idx: Index of padding token
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.recurrent = cell_module(
            input_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, src: torch.Tensor, src_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode source sequences.

        Args:
            src: Source sequences (batch_size, src_len)
            src_lengths: Actual lengths before padding (batch_size,)

        Returns:
            Tuple of (outputs, hidden) where:
            - outputs: Encoder outputs at each timestep (batch_size, src_len, hidden_dim)
            - hidden: Final hidden state (num_layers, batch_size, hidden_dim)
        """
        # Embed tokens
        embedded = self.dropout(self.embedding(src))  # (batch_size, src_len, embedding_dim)

        # Pack padded sequences for efficient processing
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, src_lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        # Process through recurrent network
        packed_outputs, hidden = self.recurrent(packed)

        # Unpack outputs
        outputs, _ = nn.utils.rnn.pad_packed_sequence(
            packed_outputs, batch_first=True
        )  # (batch_size, src_len, hidden_dim)

        return outputs, hidden


class BaseDecoder(nn.Module):
    """
    Generic decoder that works with any recurrent cell.

    Generates target sequences conditioned on encoder context.
    """

    def __init__(
        self,
        cell_module: nn.Module,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_layers: int = 1,
        dropout: float = 0.1,
        pad_idx: int = 0,
    ):
        """
        Initialize base decoder.

        Args:
            cell_module: Recurrent cell module (e.g., MultiLayerRNN)
            vocab_size: Size of target vocabulary
            embedding_dim: Dimension of word embeddings
            hidden_dim: Dimension of recurrent hidden state
            num_layers: Number of recurrent layers
            dropout: Dropout probability
            pad_idx: Index of padding token
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.recurrent = cell_module(
            input_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )
        self.fc_out = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, tgt: torch.Tensor, hidden: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decode one step or full sequence.

        Args:
            tgt: Target sequences (batch_size, tgt_len)
            hidden: Hidden state from encoder or previous step (num_layers, batch_size, hidden_dim)

        Returns:
            Tuple of (logits, hidden) where:
            - logits: Output logits (batch_size, tgt_len, vocab_size)
            - hidden: Updated hidden state (num_layers, batch_size, hidden_dim)
        """
        # Embed tokens
        embedded = self.dropout(self.embedding(tgt))  # (batch_size, tgt_len, embedding_dim)

        # Process through recurrent network
        output, hidden = self.recurrent(embedded, hidden)  # output: (batch_size, tgt_len, hidden_dim)

        # Project to vocabulary
        logits = self.fc_out(output)  # (batch_size, tgt_len, vocab_size)

        return logits, hidden


class BaseSeq2Seq(nn.Module):
    """
    Complete sequence-to-sequence model combining encoder and decoder.

    This base class provides the standard encoder-decoder architecture
    that can be instantiated with different cell types.
    """

    def __init__(
        self,
        cell_module: nn.Module,
        vocab_size: int,
        embedding_dim: int = 256,
        hidden_dim: int = 512,
        num_layers: int = 2,
        dropout: float = 0.1,
        pad_idx: int = 0,
    ):
        """
        Initialize base seq2seq model.

        Args:
            cell_module: Recurrent cell module to use (e.g., MultiLayerRNN)
            vocab_size: Size of vocabulary (shared for src and tgt)
            embedding_dim: Dimension of word embeddings
            hidden_dim: Dimension of recurrent hidden states
            num_layers: Number of recurrent layers in encoder and decoder
            dropout: Dropout probability
            pad_idx: Index of padding token
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.pad_idx = pad_idx

        self.encoder = BaseEncoder(
            cell_module, vocab_size, embedding_dim, hidden_dim, num_layers, dropout, pad_idx
        )
        self.decoder = BaseDecoder(
            cell_module, vocab_size, embedding_dim, hidden_dim, num_layers, dropout, pad_idx
        )

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass with maximum likelihood training (teacher forcing).

        Standard seq2seq training: decoder receives ground truth tokens as input
        and learns to predict the next token via cross-entropy loss.

        Args:
            src: Source sequences (batch_size, src_len)
            tgt: Target sequences (batch_size, tgt_len)
            src_lengths: Source lengths (batch_size,)

        Returns:
            Output logits (batch_size, tgt_len, vocab_size)

        Note:
            This always uses teacher forcing (ground truth as decoder input).
            For autoregressive generation without ground truth, use generate().
        """
        # Encode source
        _, hidden = self.encoder(src, src_lengths)

        # Decode with teacher forcing (ground truth as input)
        logits, _ = self.decoder(tgt, hidden)

        return logits

    def encode(
        self, src: torch.Tensor, src_lengths: torch.Tensor
    ) -> torch.Tensor:
        """
        Encode source sequences.

        Args:
            src: Source sequences (batch_size, src_len)
            src_lengths: Source lengths (batch_size,)

        Returns:
            Encoder hidden state (num_layers, batch_size, hidden_dim)
        """
        _, hidden = self.encoder(src, src_lengths)
        return hidden

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
        # Encode source
        hidden = self.encode(src, src_lengths)

        # Decode target
        logits, _ = self.decoder(tgt, hidden)

        return logits
