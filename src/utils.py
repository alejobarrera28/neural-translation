"""
Utility functions for neural machine translation.

Includes helpers for masking, sequence generation, and metric computation.
"""

import torch
import torch.nn.functional as F
from typing import List, Optional


def greedy_decode(
    model,
    src: torch.Tensor,
    src_lengths: torch.Tensor,
    max_len: int,
    bos_idx: int,
    eos_idx: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Greedy decoding for sequence generation.

    Args:
        model: Seq2Seq model with generate() method
        src: Source sequences (batch_size, src_len)
        src_lengths: Source lengths (batch_size,)
        max_len: Maximum generation length
        bos_idx: Index of BOS token
        eos_idx: Index of EOS token
        device: Device for computation

    Returns:
        Generated sequences (batch_size, max_len)
    """
    batch_size = src.size(0)

    # Encode source once (avoid redundant encoding in loop)
    hidden = model.encode(src, src_lengths)

    # Initialize output with BOS token
    output = torch.full((batch_size, 1), bos_idx, dtype=torch.long, device=device)

    # Track which sequences have finished (encountered EOS)
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

    for _ in range(max_len - 1):
        # Decoder processes entire sequence so far [BOS, t1, t2, ...] autoregressively
        # Hidden state naturally updates each step as output grows
        logits, hidden = model.decoder(output, hidden)

        # Take last position and get most likely token
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)  # (batch_size, 1)

        # Append to output
        output = torch.cat([output, next_token], dim=1)

        # Mark sequences that generated EOS
        finished |= next_token.squeeze(-1) == eos_idx

        # Stop if all sequences finished
        if finished.all():
            break

    return output


def beam_search_decode(
    model,
    src: torch.Tensor,
    src_lengths: torch.Tensor,
    max_len: int,
    bos_idx: int,
    eos_idx: int,
    beam_width: int,
    device: torch.device,
    length_penalty: float = 1.0,
) -> torch.Tensor:
    """
    Beam search decoding for sequence generation.

    Args:
        model: Seq2Seq model with generate() method
        src: Source sequences (batch_size, src_len) - NOTE: batch_size must be 1
        src_lengths: Source lengths (batch_size,)
        max_len: Maximum generation length
        bos_idx: Index of BOS token
        eos_idx: Index of EOS token
        beam_width: Number of beams
        device: Device for computation
        length_penalty: Length penalty factor (alpha in Google NMT)

    Returns:
        Best generated sequence (1, seq_len)
    """
    assert src.size(0) == 1, "Beam search only supports batch_size=1"

    # Encode source once
    hidden = model.encode(src, src_lengths)

    # Initialize beams with BOS token
    # Each beam: (sequence, score, hidden_state)
    beams = [(torch.tensor([[bos_idx]], device=device), 0.0, hidden)]
    finished_beams = []

    for _ in range(max_len - 1):
        candidates = []

        for seq, score, h in beams:
            # Skip if this beam already generated EOS
            if seq[0, -1].item() == eos_idx:
                finished_beams.append((seq, score, h))
                continue

            # Get predictions (decode only)
            logits, new_hidden = model.decoder(seq, h)
            log_probs = F.log_softmax(logits[:, -1, :], dim=-1)  # (1, vocab_size)

            # Get top-k tokens
            top_log_probs, top_indices = log_probs.topk(beam_width, dim=-1)

            # Create new candidate beams
            for k in range(beam_width):
                token = top_indices[0, k].unsqueeze(0).unsqueeze(0)  # (1, 1)
                token_score = top_log_probs[0, k].item()

                new_seq = torch.cat([seq, token], dim=1)
                new_score = score + token_score

                candidates.append((new_seq, new_score, new_hidden))

        # Select top beam_width candidates
        # Apply length penalty: score / (length^alpha)
        candidates.sort(
            key=lambda x: x[1] / (x[0].size(1) ** length_penalty), reverse=True
        )
        beams = candidates[:beam_width]

        # Stop if all beams finished
        if len(beams) == 0:
            break

    # Add remaining beams to finished
    finished_beams.extend(beams)

    # Return best beam
    if finished_beams:
        best_beam = max(
            finished_beams, key=lambda x: x[1] / (x[0].size(1) ** length_penalty)
        )
        return best_beam[0]
    else:
        return beams[0][0]


def compute_bleu(
    predictions: List[str],
    references: List[str],
    max_n: int = 4,
) -> float:
    """
    BLEU (Bilingual Evaluation Understudy) measures translation quality by
    comparing n-gram overlap between predictions and references.

    Formula: BLEU = BP × exp(Σ(1/N) × Σ log(precision_n))
    Where:
        - precision_n: n-gram precision (clipped to reference counts per sentence)
        - BP: brevity penalty (penalizes short translations)
        - N: maximum n-gram order (typically 4)

    Uses corpus-level aggregation with per-sentence n-gram clipping (not sentence
    concatenation). N-grams are counted per sentence, clipped to reference counts,
    then statistics are aggregated. Add-epsilon smoothing handles zero matches.

    Args:
        predictions: List of predicted sentences
        references: List of reference sentences
        max_n: Maximum n-gram order

    Returns:
        BLEU score as float (0-100 scale)
    """
    from collections import Counter
    import math

    # Edge case: empty inputs
    if not predictions or not references:
        return 0.0

    if len(predictions) != len(references):
        raise ValueError(
            f"Length mismatch: {len(predictions)} predictions vs {len(references)} references"
        )

    # Corpus-level statistics with per-sentence clipping
    # This is the standard BLEU implementation approach
    total_pred_len = 0
    total_ref_len = 0

    # Track matches and counts per n-gram order
    n_gram_stats = {n: {'matches': 0, 'total': 0} for n in range(1, max_n + 1)}

    for pred, ref in zip(predictions, references):
        # Basic tokenization: split on whitespace
        pred_tokens = pred.strip().split()
        ref_tokens = ref.strip().split()

        # Update corpus length statistics
        total_pred_len += len(pred_tokens)
        total_ref_len += len(ref_tokens)

        # Compute n-gram matches for each order with per-sentence clipping
        for n in range(1, max_n + 1):
            # Extract n-grams for this sentence
            pred_ngrams = _get_ngrams(pred_tokens, n)
            ref_ngrams = _get_ngrams(ref_tokens, n)

            # Count clipped matches (clip to reference counts per sentence)
            for ngram, pred_count in pred_ngrams.items():
                ref_count = ref_ngrams.get(ngram, 0)
                n_gram_stats[n]['matches'] += min(pred_count, ref_count)

            # Total prediction n-grams
            n_gram_stats[n]['total'] += sum(pred_ngrams.values())

    # Edge case: empty after tokenization
    if total_pred_len == 0 or total_ref_len == 0:
        return 0.0

    # Compute n-gram precisions
    log_precisions = []

    for n in range(1, max_n + 1):
        matches = n_gram_stats[n]['matches']
        total = n_gram_stats[n]['total']

        # Edge case: no n-grams of this length (e.g., prediction too short)
        if total == 0:
            # Use small epsilon to avoid log(0)
            precision = 1e-10
        else:
            # Add-epsilon smoothing for zero matches
            precision = (matches + 1e-10) / (total + 1e-10)

        log_precisions.append(math.log(precision))

    # Geometric mean of n-gram precisions
    bleu = math.exp(sum(log_precisions) / max_n)

    # Brevity penalty (penalize predictions shorter than references)
    pred_len = total_pred_len
    ref_len = total_ref_len

    if pred_len < ref_len:
        bp = math.exp(1 - ref_len / pred_len)
    else:
        bp = 1.0

    # Scale to 0-100 and apply brevity penalty
    return bp * bleu * 100


def _get_ngrams(tokens: List[str], n: int) -> dict:
    """
    Extract n-grams from a list of tokens.

    Args:
        tokens: List of token strings
        n: N-gram order (1=unigram, 2=bigram, etc.)

    Returns:
        Counter mapping n-grams (as tuples) to their counts
    """
    from collections import Counter

    ngrams = []
    for i in range(len(tokens) - n + 1):
        ngram = tuple(tokens[i : i + n])
        ngrams.append(ngram)

    return Counter(ngrams)


def count_parameters(model: torch.nn.Module) -> int:
    """
    Count trainable parameters in a model.

    Args:
        model: PyTorch model

    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    path: str,
    **kwargs,
) -> None:
    """
    Save training checkpoint.

    Args:
        model: Model to save
        optimizer: Optimizer to save
        epoch: Current epoch
        loss: Current loss
        path: Path to save checkpoint
        **kwargs: Additional items to save
    """
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        **kwargs,
    }
    torch.save(checkpoint, path)


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> dict:
    """
    Load training checkpoint.

    Args:
        path: Path to checkpoint
        model: Model to load state into
        optimizer: Optional optimizer to load state into

    Returns:
        Checkpoint dictionary with epoch, loss, etc.
    """
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint
