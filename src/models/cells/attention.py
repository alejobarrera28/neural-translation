"""
Bahdanau attention mechanism implemented from scratch.

Implements additive attention that computes alignment scores between
decoder hidden states and encoder outputs. This allows the decoder to
dynamically focus on different parts of the input sequence.

Attention score: score(h_decoder, h_encoder) = v^T tanh(W_1 h_decoder + W_2 h_encoder)
Attention weights: α = softmax(scores)
Context vector: c_t = Σ α_i * h_encoder_i

This mechanism solves the information bottleneck of fixed-length context vectors,
allowing the model to handle longer sequences more effectively.

Reference: Bahdanau et al. (2015) - "Neural Machine Translation by Jointly Learning to Align and Translate"
           https://arxiv.org/abs/1409.0473
"""

import torch
import torch.nn as nn
from typing import Tuple


class BahdanauAttention(nn.Module):
    """
    Bahdanau (additive) attention mechanism.

    Computes attention weights and context vector by scoring the alignment
    between decoder hidden state and each encoder hidden state.

    The attention mechanism allows the decoder to "look back" at the source
    sequence and focus on the most relevant parts for the current decoding step.
    """

    def __init__(self, decoder_hidden_dim: int, encoder_hidden_dim: int, attention_dim: int = 256):
        """
        Initialize Bahdanau attention.

        Args:
            decoder_hidden_dim: Dimension of decoder hidden states
            encoder_hidden_dim: Dimension of encoder hidden states (hidden_dim * 2 for bidirectional)
            attention_dim: Dimension of attention layer (intermediate projection size)
        """
        super().__init__()

        self.decoder_hidden_dim = decoder_hidden_dim
        self.encoder_hidden_dim = encoder_hidden_dim
        self.attention_dim = attention_dim

        # Attention layers
        # Project decoder hidden state to attention dimension
        self.W_decoder = nn.Linear(decoder_hidden_dim, attention_dim, bias=False)

        # Project encoder outputs to attention dimension
        self.W_encoder = nn.Linear(encoder_hidden_dim, attention_dim, bias=False)

        # Final projection to scalar scores
        self.v = nn.Linear(attention_dim, 1, bias=False)

    def forward(
        self,
        decoder_hidden: torch.Tensor,
        encoder_outputs: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute attention weights and context vector.

        Args:
            decoder_hidden: Current decoder hidden state (batch_size, decoder_hidden_dim)
            encoder_outputs: All encoder outputs (batch_size, src_len, encoder_hidden_dim)
            mask: Optional padding mask (batch_size, src_len) - True for padding positions

        Returns:
            Tuple of (context_vector, attention_weights) where:
            - context_vector: Weighted sum of encoder outputs (batch_size, encoder_hidden_dim)
            - attention_weights: Attention distribution (batch_size, src_len)
        """
        batch_size = encoder_outputs.size(0)
        src_len = encoder_outputs.size(1)

        # Project decoder hidden state
        # (batch_size, decoder_hidden_dim) -> (batch_size, 1, attention_dim)
        decoder_proj = self.W_decoder(decoder_hidden).unsqueeze(1)

        # Project encoder outputs
        # (batch_size, src_len, encoder_hidden_dim) -> (batch_size, src_len, attention_dim)
        encoder_proj = self.W_encoder(encoder_outputs)

        # Compute attention scores using additive (tanh) scoring
        # Broadcast decoder_proj across src_len and add to encoder_proj
        # (batch_size, src_len, attention_dim) -> (batch_size, src_len, 1) -> (batch_size, src_len)
        scores = self.v(torch.tanh(decoder_proj + encoder_proj))
        scores = scores.squeeze(2)  # (batch_size, src_len)

        # Apply mask to prevent attending to padding tokens
        if mask is not None:
            scores = scores.masked_fill(mask, -1e10)

        # Compute attention weights (normalize with softmax)
        attention_weights = torch.softmax(scores, dim=1)  # (batch_size, src_len)

        # Compute context vector as weighted sum of encoder outputs
        # (batch_size, 1, src_len) @ (batch_size, src_len, encoder_hidden_dim)
        # -> (batch_size, 1, encoder_hidden_dim) -> (batch_size, encoder_hidden_dim)
        context_vector = torch.bmm(
            attention_weights.unsqueeze(1), encoder_outputs
        ).squeeze(1)

        return context_vector, attention_weights
