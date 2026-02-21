"""
Training script for seq2seq models.

Supports all model architectures: RNN, LSTM, Attention, Transformer.
Handles training loop, validation, checkpointing, and logging.
"""

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.bpe_tokenizer import BPETokenizer
from src.data.dataset import create_translation_dataloaders
from src.utils import count_parameters, save_checkpoint


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    pad_idx: int,
    scaler=None,
    use_amp: bool = False,
) -> float:
    """
    Train for one epoch using maximum likelihood (teacher forcing).

    Args:
        model: Seq2Seq model
        dataloader: Training DataLoader
        optimizer: Optimizer
        criterion: Loss function
        device: Device for computation
        pad_idx: Padding token index
        scaler: GradScaler for mixed precision training
        use_amp: Whether to use automatic mixed precision

    Returns:
        Average loss for the epoch
    """
    model.train()
    epoch_loss = 0

    for src, src_lengths, tgt, tgt_lengths in tqdm(dataloader, desc="Training"):
        # Move to device
        src = src.to(device)
        tgt = tgt.to(device)
        src_lengths = src_lengths.to(device)

        # Zero gradients
        optimizer.zero_grad(set_to_none=True)

        # Forward pass
        # Target input: all tokens except last (for teacher forcing)
        # Target output: all tokens except first (BOS)
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        # Automatic Mixed Precision
        if use_amp:
            device_type = str(device).split(':')[0]
            with torch.autocast(device_type=device_type):
                logits = model(src, tgt_input, src_lengths)

                # Compute loss
                logits_flat = logits.reshape(-1, logits.size(-1))
                tgt_flat = tgt_output.reshape(-1)
                loss = criterion(logits_flat, tgt_flat)

            # Backward pass with gradient scaling (CUDA only)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                # MPS: no scaler needed
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
        else:
            logits = model(src, tgt_input, src_lengths)

            # Compute loss
            logits_flat = logits.reshape(-1, logits.size(-1))
            tgt_flat = tgt_output.reshape(-1)
            loss = criterion(logits_flat, tgt_flat)

            # Backward pass
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Update weights
            optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(dataloader)


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    pad_idx: int,
    use_amp: bool = False,
) -> float:
    """
    Evaluate model on validation set.

    Args:
        model: Seq2Seq model
        dataloader: Validation DataLoader
        criterion: Loss function
        device: Device for computation
        pad_idx: Padding token index
        use_amp: Whether to use automatic mixed precision

    Returns:
        Average validation loss
    """
    model.eval()
    epoch_loss = 0

    with torch.no_grad():
        for src, src_lengths, tgt, tgt_lengths in tqdm(dataloader, desc="Evaluating"):
            src = src.to(device)
            tgt = tgt.to(device)
            src_lengths = src_lengths.to(device)

            # Forward pass (using teacher forcing to compute validation loss)
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            if use_amp:
                with torch.autocast(device_type=str(device).split(':')[0]):
                    logits = model(src, tgt_input, src_lengths)

                    # Compute loss
                    logits_flat = logits.reshape(-1, logits.size(-1))
                    tgt_flat = tgt_output.reshape(-1)
                    loss = criterion(logits_flat, tgt_flat)
            else:
                logits = model(src, tgt_input, src_lengths)

                # Compute loss
                logits_flat = logits.reshape(-1, logits.size(-1))
                tgt_flat = tgt_output.reshape(-1)
                loss = criterion(logits_flat, tgt_flat)

            epoch_loss += loss.item()

    return epoch_loss / len(dataloader)


def train(
    model_name: str,
    train_src_path: Path,
    train_tgt_path: Path,
    val_src_path: Path,
    val_tgt_path: Path,
    tokenizer_path: Path,
    output_dir: Path,
    epochs: int = 10,
    batch_size: int = 128,
    learning_rate: float = 0.001,
    embedding_dim: int = 128,
    hidden_dim: int = 512,
    num_layers: int = 2,
    dropout: float = 0.1,
    max_len: int = 100,
    device: str = "mps",
    use_amp: bool = True,
):
    """
    Main training function.

    Args:
        model_name: Name of model architecture (rnn, lstm, attention, transformer)
        train_src_path: Path to training source file
        train_tgt_path: Path to training target file
        val_src_path: Path to validation source file
        val_tgt_path: Path to validation target file
        tokenizer_path: Path to BPE tokenizer
        output_dir: Directory to save checkpoints
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        embedding_dim: Embedding dimension
        hidden_dim: Hidden dimension
        num_layers: Number of layers
        dropout: Dropout probability
        max_len: Maximum sequence length
        device: Device for training (mps/cuda/cpu)
        use_amp: Whether to use automatic mixed precision training
    """
    # Setup device with MPS support for Apple Silicon
    if device == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup mixed precision training
    scaler = None
    device_type = str(device).split(':')[0]  # 'mps', 'cuda', or 'cpu'

    if use_amp and device_type != "cpu":
        if device_type == "cuda":
            scaler = torch.amp.GradScaler('cuda')
        elif device_type == "mps":
            # MPS doesn't use GradScaler, just autocast
            scaler = None
        amp_status = "Enabled"
    else:
        use_amp = False  # Disable AMP for CPU
        amp_status = "Disabled" if device_type == "cpu" else "Disabled (user)"

    print("\n" + "=" * 60)
    print(f"Training {model_name.upper()} Seq2Seq Model")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Mixed Precision (AMP): {amp_status}")

    # Load tokenizer
    print(f"\nLoading tokenizer from {tokenizer_path}...")
    tokenizer = BPETokenizer.load(tokenizer_path)
    vocab_size = len(tokenizer.token2idx)
    pad_idx = tokenizer.token2idx[tokenizer.PAD_TOKEN]
    print(f"Vocabulary size: {vocab_size:,}")

    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader, val_loader = create_translation_dataloaders(
        train_src_path,
        train_tgt_path,
        val_src_path,
        val_tgt_path,
        tokenizer,
        batch_size=batch_size,
        max_len=max_len,
        num_workers=4,  # Parallel data loading
    )
    print(f"Train batches: {len(train_loader):,}")
    print(f"Val batches: {len(val_loader):,}")

    # Initialize model
    print(f"\nInitializing {model_name} model...")
    if model_name == "rnn":
        from src.models.rnn_seq2seq import RNNSeq2Seq

        model = RNNSeq2Seq(
            vocab_size, embedding_dim, hidden_dim, num_layers, dropout, pad_idx
        )
    elif model_name == "lstm":
        # TODO: Import LSTM model when implemented
        raise NotImplementedError("LSTM model not yet implemented")
    elif model_name == "attention":
        # TODO: Import Attention model when implemented
        raise NotImplementedError("Attention model not yet implemented")
    elif model_name == "transformer":
        # TODO: Import Transformer model when implemented
        raise NotImplementedError("Transformer model not yet implemented")
    else:
        raise ValueError(f"Unknown model: {model_name}")

    model = model.to(device)
    print(f"Parameters: {count_parameters(model):,}")

    # Setup training
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    print("\nStarting training...\n")
    best_val_loss = float("inf")

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")

        # Train
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            pad_idx,
            scaler,
            use_amp,
        )

        # Validate
        val_loss = evaluate(model, val_loader, criterion, device, pad_idx, use_amp)

        print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # Save checkpoint
        checkpoint_path = output_dir / f"{model_name}_epoch_{epoch+1}.pt"
        save_checkpoint(
            model,
            optimizer,
            epoch,
            val_loss,
            checkpoint_path,
            vocab_size=vocab_size,
            model_name=model_name,
            model_config={
                "embedding_dim": embedding_dim,
                "hidden_dim": hidden_dim,
                "num_layers": num_layers,
                "dropout": dropout,
            },
        )
        print(f"  Checkpoint saved: {checkpoint_path}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = output_dir / f"{model_name}_best.pt"
            save_checkpoint(
                model,
                optimizer,
                epoch,
                val_loss,
                best_path,
                vocab_size=vocab_size,
                model_name=model_name,
                model_config={
                    "embedding_dim": embedding_dim,
                    "hidden_dim": hidden_dim,
                    "num_layers": num_layers,
                    "dropout": dropout,
                },
            )
            print(f"  Best model saved: {best_path}")

        print()

    print("=" * 60)
    print("Training complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train seq2seq translation model")

    # Data
    parser.add_argument(
        "--train-src",
        type=str,
        default="data/processed/opus-100/train.en",
        help="Training source file",
    )
    parser.add_argument(
        "--train-tgt",
        type=str,
        default="data/processed/opus-100/train.es",
        help="Training target file",
    )
    parser.add_argument(
        "--val-src",
        type=str,
        default="data/processed/opus-100/dev.en",
        help="Validation source file",
    )
    parser.add_argument(
        "--val-tgt",
        type=str,
        default="data/processed/opus-100/dev.es",
        help="Validation target file",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="data/vocab/bpe_vocab_shared.pkl",
        help="BPE tokenizer path",
    )

    # Model
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["rnn", "lstm", "attention", "transformer"],
        help="Model architecture",
    )
    parser.add_argument(
        "--embedding-dim", type=int, default=128, help="Embedding dimension"
    )
    parser.add_argument("--hidden-dim", type=int, default=512, help="Hidden dimension")
    parser.add_argument("--num-layers", type=int, default=2, help="Number of layers")
    parser.add_argument(
        "--dropout", type=float, default=0.1, help="Dropout probability"
    )

    # Training
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument(
        "--learning-rate", type=float, default=0.001, help="Learning rate"
    )
    parser.add_argument(
        "--max-len", type=int, default=100, help="Maximum sequence length"
    )

    # Output
    parser.add_argument(
        "--output-dir", type=str, default="models/checkpoints", help="Output directory"
    )
    parser.add_argument(
        "--device", type=str, default="mps", help="Device (mps/cuda/cpu)"
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable automatic mixed precision training",
    )

    args = parser.parse_args()

    train(
        model_name=args.model,
        train_src_path=Path(args.train_src),
        train_tgt_path=Path(args.train_tgt),
        val_src_path=Path(args.val_src),
        val_tgt_path=Path(args.val_tgt),
        tokenizer_path=Path(args.tokenizer),
        output_dir=Path(args.output_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        max_len=args.max_len,
        device=args.device,
        use_amp=not args.no_amp,
    )
