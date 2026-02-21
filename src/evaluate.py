"""
Evaluation script for seq2seq models.

Evaluates trained models on test data using BLEU score and other metrics.
Supports greedy decoding and beam search.
"""

import argparse
import torch
from pathlib import Path
from tqdm import tqdm

# Add src to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.bpe_tokenizer import BPETokenizer
from src.data.dataset import TranslationDataset, collate_fn
from src.utils import greedy_decode, beam_search_decode, compute_bleu, load_checkpoint


def translate_dataset(
    model,
    dataset: TranslationDataset,
    tokenizer: BPETokenizer,
    device: torch.device,
    max_len: int = 100,
    batch_size: int = 1,
    beam_width: int = 1,
) -> tuple[list[str], list[str]]:
    """
    Translate entire dataset.

    Args:
        model: Trained seq2seq model
        dataset: TranslationDataset to translate
        tokenizer: BPE tokenizer
        device: Device for computation
        max_len: Maximum generation length
        batch_size: Batch size (must be 1 for beam search)
        beam_width: Beam width (1 for greedy decoding)

    Returns:
        Tuple of (predictions, references)
    """
    model.eval()
    predictions = []
    references = []

    bos_idx = tokenizer.token2idx[tokenizer.BOS_TOKEN]
    eos_idx = tokenizer.token2idx[tokenizer.EOS_TOKEN]
    pad_idx = tokenizer.token2idx[tokenizer.PAD_TOKEN]

    # Create dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, pad_idx),
    )

    with torch.no_grad():
        for src, src_lengths, tgt, _ in tqdm(dataloader, desc="Translating"):
            src = src.to(device)
            src_lengths = src_lengths.to(device)

            # Generate translation
            if beam_width > 1:
                assert batch_size == 1, "Beam search requires batch_size=1"
                output = beam_search_decode(
                    model,
                    src,
                    src_lengths,
                    max_len,
                    bos_idx,
                    eos_idx,
                    beam_width,
                    device,
                )
            else:
                output = greedy_decode(
                    model, src, src_lengths, max_len, bos_idx, eos_idx, device
                )

            # Decode predictions
            for i in range(output.size(0)):
                pred_indices = output[i].cpu().tolist()
                pred_text = tokenizer.decode(pred_indices, remove_special=True)
                predictions.append(pred_text)

                # Get reference
                ref_indices = tgt[i].cpu().tolist()
                ref_text = tokenizer.decode(ref_indices, remove_special=True)
                references.append(ref_text)

    return predictions, references


def evaluate(
    model_path: Path,
    test_src_path: Path,
    test_tgt_path: Path,
    tokenizer_path: Path,
    output_path: Path = None,
    max_len: int = 100,
    beam_width: int = 1,
    device: str = "mps",
):
    """
    Evaluate trained model on test set.

    Args:
        model_path: Path to trained model checkpoint
        test_src_path: Path to test source file
        test_tgt_path: Path to test target file
        tokenizer_path: Path to BPE tokenizer
        output_path: Optional path to save translations
        max_len: Maximum generation length
        beam_width: Beam width (1 for greedy)
        device: Device for computation (mps/cuda/cpu)
    """
    # Setup device with MPS support for Apple Silicon
    if device == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print("\n" + "=" * 60)
    print("Evaluating Seq2Seq Model")
    print("=" * 60)
    print(f"Device: {device}")

    # Load tokenizer
    print(f"\nLoading tokenizer from {tokenizer_path}...")
    tokenizer = BPETokenizer.load(tokenizer_path)
    vocab_size = len(tokenizer.token2idx)
    pad_idx = tokenizer.token2idx[tokenizer.PAD_TOKEN]
    print(f"Vocabulary size: {vocab_size:,}")

    # Load test dataset
    print("\nLoading test dataset...")
    test_dataset = TranslationDataset(
        test_src_path, test_tgt_path, tokenizer, max_len=max_len
    )
    print(f"Test examples: {len(test_dataset):,}")

    # Load model
    print(f"\nLoading model from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device)
    model_name = checkpoint.get("model_name", "rnn")

    # Initialize model architecture
    if model_name == "rnn":
        from src.models.rnn_seq2seq import RNNSeq2Seq

        # Extract model hyperparameters from checkpoint
        model_config = checkpoint.get("model_config", {})
        model = RNNSeq2Seq(
            vocab_size=vocab_size,
            embedding_dim=model_config.get("embedding_dim", 256),
            hidden_dim=model_config.get("hidden_dim", 512),
            num_layers=model_config.get("num_layers", 2),
            dropout=model_config.get("dropout", 0.1),
            pad_idx=pad_idx,
        )
    elif model_name == "lstm":
        # TODO: Import LSTM model
        raise NotImplementedError("LSTM model not yet implemented")
    elif model_name == "attention":
        # TODO: Import Attention model
        raise NotImplementedError("Attention model not yet implemented")
    elif model_name == "transformer":
        # TODO: Import Transformer model
        raise NotImplementedError("Transformer model not yet implemented")
    else:
        raise ValueError(f"Unknown model: {model_name}")

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    print(f"Model loaded (epoch {checkpoint['epoch']})")

    # Translate
    print(f"\nTranslating test set (beam_width={beam_width})...")
    batch_size = 1 if beam_width > 1 else 32
    predictions, references = translate_dataset(
        model, test_dataset, tokenizer, device, max_len, batch_size, beam_width
    )

    # Compute BLEU
    print("\nComputing BLEU score...")
    bleu_score = compute_bleu(predictions, references)
    print(f"BLEU Score: {bleu_score:.2f}")

    # Save translations
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            for i, (pred, ref) in enumerate(zip(predictions, references)):
                f.write(f"SRC: {test_dataset.src_sentences[i]}\n")
                f.write(f"REF: {ref}\n")
                f.write(f"PRED: {pred}\n")
                f.write("\n")

        print(f"\nTranslations saved to: {output_path}")

    # Show sample translations
    print("\nSample Translations:")
    print("-" * 60)
    for i in range(min(5, len(predictions))):
        print(f"Example {i+1}:")
        print(f"  SRC:  {test_dataset.src_sentences[i]}")
        print(f"  REF:  {references[i]}")
        print(f"  PRED: {predictions[i]}")
        print()

    print("=" * 60)
    print(f"Evaluation complete! BLEU: {bleu_score:.2f}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate seq2seq translation model")

    parser.add_argument(
        "--model", type=str, required=True, help="Path to model checkpoint"
    )

    parser.add_argument(
        "--test-src",
        type=str,
        default="data/processed/opus-100/test.en",
        help="Test source file",
    )
    parser.add_argument(
        "--test-tgt",
        type=str,
        default="data/processed/opus-100/test.es",
        help="Test target file",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="data/vocab/bpe_vocab_shared.pkl",
        help="BPE tokenizer path",
    )

    parser.add_argument(
        "--output", type=str, default=None, help="Output file for translations"
    )
    parser.add_argument(
        "--max-len", type=int, default=100, help="Maximum generation length"
    )
    parser.add_argument(
        "--beam-width", type=int, default=1, help="Beam width (1=greedy)"
    )
    parser.add_argument(
        "--device", type=str, default="mps", help="Device (mps/cuda/cpu)"
    )

    args = parser.parse_args()

    evaluate(
        model_path=Path(args.model),
        test_src_path=Path(args.test_src),
        test_tgt_path=Path(args.test_tgt),
        tokenizer_path=Path(args.tokenizer),
        output_path=Path(args.output) if args.output else None,
        max_len=args.max_len,
        beam_width=args.beam_width,
        device=args.device,
    )
