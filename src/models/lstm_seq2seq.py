"""
LSTM Seq2Seq model for neural machine translation.

Uses from-scratch LSTM cells wrapped in the base encoder-decoder framework.
This model addresses the vanishing gradient problem of vanilla RNNs through
gating mechanisms, allowing it to capture longer-range dependencies.

Architecture:
    Encoder: Multi-layer LSTM that processes source sequence
    Decoder: Multi-layer LSTM that generates target sequence
    Context: Final encoder hidden state and cell state initialize decoder

Reference: Sutskever et al. (2014) - "Sequence to Sequence Learning with Neural Networks"
           Hochreiter & Schmidhuber (1997) - "Long Short-Term Memory"
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.base_seq2seq import BaseSeq2Seq
from src.models.cells.lstm_cell import MultiLayerLSTM


class LSTMSeq2Seq(BaseSeq2Seq):
    """
    Complete LSTM Seq2Seq model using from-scratch LSTM cells.

    Inherits the encoder-decoder structure from BaseSeq2Seq and uses
    custom LSTM cells for the recurrent computations.
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
        Initialize LSTM Seq2Seq model.

        Args:
            vocab_size: Size of vocabulary (shared for src and tgt)
            embedding_dim: Dimension of word embeddings
            hidden_dim: Dimension of LSTM hidden states
            num_layers: Number of LSTM layers in encoder and decoder
            dropout: Dropout probability
            pad_idx: Index of padding token
        """
        super().__init__(
            cell_module=MultiLayerLSTM,
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            pad_idx=pad_idx,
        )
