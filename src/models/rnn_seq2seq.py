"""
Vanilla RNN Seq2Seq model for neural machine translation.

Uses from-scratch RNN cells wrapped in the base encoder-decoder framework.
This is the simplest seq2seq model, serving as a baseline.

Architecture:
    Encoder: Multi-layer RNN that processes source sequence
    Decoder: Multi-layer RNN that generates target sequence
    Context: Final encoder hidden state initializes decoder

Reference: Sutskever et al. (2014) - "Sequence to Sequence Learning with Neural Networks"
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.base_seq2seq import BaseSeq2Seq
from src.models.cells.rnn_cell import MultiLayerRNN


class RNNSeq2Seq(BaseSeq2Seq):
    """
    Complete RNN Seq2Seq model using from-scratch RNN cells.

    Inherits the encoder-decoder structure from BaseSeq2Seq and uses
    custom RNN cells for the recurrent computations.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 256,
        hidden_dim: int = 512,
        num_layers: int = 2,
        dropout: float = 0.1,
        pad_idx: int = 0,
    ):
        """
        Initialize RNN Seq2Seq model.

        Args:
            vocab_size: Size of vocabulary (shared for src and tgt)
            embedding_dim: Dimension of word embeddings
            hidden_dim: Dimension of RNN hidden states
            num_layers: Number of RNN layers in encoder and decoder
            dropout: Dropout probability
            pad_idx: Index of padding token
        """
        super().__init__(
            cell_module=MultiLayerRNN,
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            pad_idx=pad_idx,
        )
