"""
PyTorch Dataset for parallel translation data with BPE tokenization.

Handles loading, tokenizing, and batching parallel sentence pairs for
seq2seq training and evaluation.
"""

from pathlib import Path
from typing import List, Tuple, Optional
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class TranslationDataset(Dataset):
    """
    Dataset for parallel translation pairs.

    Loads source and target sentences, tokenizes them using a BPE tokenizer,
    and provides batching utilities.
    """

    def __init__(
        self,
        src_path: Path,
        tgt_path: Path,
        tokenizer,  # BPETokenizer instance
        max_len: Optional[int] = None,
    ):
        """
        Initialize translation dataset.

        Args:
            src_path: Path to source language file (one sentence per line)
            tgt_path: Path to target language file (one sentence per line)
            tokenizer: Trained BPETokenizer instance (shared for both languages)
            max_len: Maximum sequence length (None for no limit)
        """
        self.tokenizer = tokenizer
        self.max_len = max_len

        # Load parallel sentences
        with open(src_path, "r", encoding="utf-8") as f:
            self.src_sentences = [line.strip() for line in f]
        with open(tgt_path, "r", encoding="utf-8") as f:
            self.tgt_sentences = [line.strip() for line in f]

        assert len(self.src_sentences) == len(
            self.tgt_sentences
        ), f"Mismatched lengths: {len(self.src_sentences)} vs {len(self.tgt_sentences)}"

        # Get special token indices
        self.pad_idx = tokenizer.token2idx[tokenizer.PAD_TOKEN]
        self.bos_idx = tokenizer.token2idx[tokenizer.BOS_TOKEN]
        self.eos_idx = tokenizer.token2idx[tokenizer.EOS_TOKEN]
        self.unk_idx = tokenizer.token2idx[tokenizer.UNK_TOKEN]

    def __len__(self) -> int:
        """Return number of sentence pairs."""
        return len(self.src_sentences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single source-target pair.

        Args:
            idx: Index of the sentence pair

        Returns:
            Tuple of (src_tensor, tgt_tensor) where:
            - src_tensor: Source indices with BOS/EOS tokens
            - tgt_tensor: Target indices with BOS/EOS tokens
        """
        src_text = self.src_sentences[idx]
        tgt_text = self.tgt_sentences[idx]

        # Encode with BOS/EOS tokens
        src_indices = self.tokenizer.encode(src_text, add_bos=True, add_eos=True)
        tgt_indices = self.tokenizer.encode(tgt_text, add_bos=True, add_eos=True)

        # Apply length limit if specified, ensuring EOS token is preserved
        if self.max_len is not None:
            if len(src_indices) > self.max_len:
                # Keep BOS and first (max_len - 2) tokens, then add EOS
                src_indices = src_indices[: self.max_len - 1] + [src_indices[-1]]
            if len(tgt_indices) > self.max_len:
                # Keep BOS and first (max_len - 2) tokens, then add EOS
                tgt_indices = tgt_indices[: self.max_len - 1] + [tgt_indices[-1]]

        return torch.tensor(src_indices, dtype=torch.long), torch.tensor(
            tgt_indices, dtype=torch.long
        )


def collate_fn(
    batch: List[Tuple[torch.Tensor, torch.Tensor]], pad_idx: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Collate function for DataLoader to batch variable-length sequences.

    Pads sequences to the maximum length in the batch and creates length tensors.

    Args:
        batch: List of (src, tgt) tensor pairs
        pad_idx: Index of PAD token for padding

    Returns:
        Tuple of (src_batch, src_lengths, tgt_batch, tgt_lengths) where:
        - src_batch: Padded source sequences (batch_size, max_src_len)
        - src_lengths: Actual lengths before padding (batch_size,)
        - tgt_batch: Padded target sequences (batch_size, max_tgt_len)
        - tgt_lengths: Actual lengths before padding (batch_size,)
    """
    src_batch, tgt_batch = zip(*batch)

    # Get lengths before padding
    src_lengths = torch.tensor([len(s) for s in src_batch], dtype=torch.long)
    tgt_lengths = torch.tensor([len(t) for t in tgt_batch], dtype=torch.long)

    # Pad sequences (pad_sequence pads to max length in batch)
    # batch_first=True gives (batch_size, seq_len) instead of (seq_len, batch_size)
    src_padded = pad_sequence(src_batch, batch_first=True, padding_value=pad_idx)
    tgt_padded = pad_sequence(tgt_batch, batch_first=True, padding_value=pad_idx)

    return src_padded, src_lengths, tgt_padded, tgt_lengths


class CollateFnWrapper:
    """Wrapper to make collate_fn picklable for multiprocessing."""

    def __init__(self, pad_idx: int):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        return collate_fn(batch, self.pad_idx)


def create_translation_dataloaders(
    train_src_path: Path,
    train_tgt_path: Path,
    val_src_path: Path,
    val_tgt_path: Path,
    tokenizer,
    batch_size: int = 32,
    max_len: Optional[int] = None,
    num_workers: int = 0,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create training and validation DataLoaders.

    Args:
        train_src_path: Path to training source file
        train_tgt_path: Path to training target file
        val_src_path: Path to validation source file
        val_tgt_path: Path to validation target file
        tokenizer: Trained BPETokenizer instance
        batch_size: Batch size for DataLoader
        max_len: Maximum sequence length (None for no limit)
        num_workers: Number of DataLoader workers

    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Create datasets
    train_dataset = TranslationDataset(
        train_src_path, train_tgt_path, tokenizer, max_len=max_len
    )
    val_dataset = TranslationDataset(
        val_src_path, val_tgt_path, tokenizer, max_len=max_len
    )

    # Get PAD index for collate function
    pad_idx = tokenizer.token2idx[tokenizer.PAD_TOKEN]

    # Create picklable collate function for multiprocessing
    collate = CollateFnWrapper(pad_idx)

    # Create DataLoaders with custom collate function
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate,
        num_workers=num_workers,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate,
        num_workers=num_workers,
    )

    return train_loader, val_loader
