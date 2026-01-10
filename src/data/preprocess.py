"""
Data preprocessing script for OPUS-100 EN-ES parallel corpus.

This script cleans the raw data by:
1. Normalizing Unicode (NFC form)
2. Cleaning HTML/XML tags and entities
3. Normalizing whitespace
4. Removing empty lines
5. Filtering punctuation-only pairs
6. Filtering by sentence length (min/max tokens and characters)
7. Filtering severe misalignments (length ratio)
8. Deduplicating exact pairs (training only)
9. Shuffling data (training only)
"""

from pathlib import Path
from typing import List, Tuple, Dict
import argparse
import unicodedata
import re
import html
import json
import random


def load_parallel_data(en_path: Path, es_path: Path) -> Tuple[List[str], List[str]]:
    """Load parallel text files."""
    with open(en_path, "r", encoding="utf-8") as f:
        en_lines = [line.strip() for line in f]
    with open(es_path, "r", encoding="utf-8") as f:
        es_lines = [line.strip() for line in f]

    assert len(en_lines) == len(
        es_lines
    ), f"Mismatched line counts: {len(en_lines)} vs {len(es_lines)}"
    return en_lines, es_lines


def save_parallel_data(
    en_lines: List[str], es_lines: List[str], en_path: Path, es_path: Path
) -> None:
    """Save parallel text files."""
    en_path.parent.mkdir(parents=True, exist_ok=True)
    es_path.parent.mkdir(parents=True, exist_ok=True)

    with open(en_path, "w", encoding="utf-8") as f:
        f.write("\n".join(en_lines) + "\n")
    with open(es_path, "w", encoding="utf-8") as f:
        f.write("\n".join(es_lines) + "\n")


def normalize_unicode(
    en_lines: List[str], es_lines: List[str]
) -> Tuple[List[str], List[str], int]:
    """
    Normalize Unicode to NFC form for consistent character representation.

    Returns:
        (normalized_en, normalized_es, num_changed)
    """
    normalized_en = []
    normalized_es = []
    num_changed = 0

    for en, es in zip(en_lines, es_lines):
        en_norm = unicodedata.normalize("NFC", en)
        es_norm = unicodedata.normalize("NFC", es)

        if en_norm != en or es_norm != es:
            num_changed += 1

        normalized_en.append(en_norm)
        normalized_es.append(es_norm)

    return normalized_en, normalized_es, num_changed


def clean_html(
    en_lines: List[str], es_lines: List[str]
) -> Tuple[List[str], List[str], int]:
    """
    Remove HTML/XML tags and decode HTML entities.

    Returns:
        (cleaned_en, cleaned_es, num_changed)
    """
    tag_pattern = re.compile(r"<[^>]+>")
    clean_en = []
    clean_es = []
    num_changed = 0

    for en, es in zip(en_lines, es_lines):
        # Decode HTML entities first
        en_decoded = html.unescape(en)
        es_decoded = html.unescape(es)

        # Remove HTML/XML tags
        en_clean = tag_pattern.sub(" ", en_decoded)
        es_clean = tag_pattern.sub(" ", es_decoded)

        if en_clean != en or es_clean != es:
            num_changed += 1

        clean_en.append(en_clean)
        clean_es.append(es_clean)

    return clean_en, clean_es, num_changed


def normalize_whitespace(
    en_lines: List[str], es_lines: List[str]
) -> Tuple[List[str], List[str], int]:
    """
    Normalize whitespace: collapse multiple spaces, strip leading/trailing.

    Returns:
        (normalized_en, normalized_es, num_changed)
    """
    whitespace_pattern = re.compile(r"\s+")
    clean_en = []
    clean_es = []
    num_changed = 0

    for en, es in zip(en_lines, es_lines):
        en_clean = whitespace_pattern.sub(" ", en).strip()
        es_clean = whitespace_pattern.sub(" ", es).strip()

        if en_clean != en.strip() or es_clean != es.strip():
            num_changed += 1

        clean_en.append(en_clean)
        clean_es.append(es_clean)

    return clean_en, clean_es, num_changed


def filter_empty_lines(
    en_lines: List[str], es_lines: List[str]
) -> Tuple[List[str], List[str], int]:
    """
    Remove pairs where either sentence is empty.

    Returns:
        (filtered_en, filtered_es, num_removed)
    """
    clean_en = []
    clean_es = []
    num_removed = 0

    for en, es in zip(en_lines, es_lines):
        if en.strip() and es.strip():
            clean_en.append(en)
            clean_es.append(es)
        else:
            num_removed += 1

    return clean_en, clean_es, num_removed


def filter_punctuation_only(
    en_lines: List[str], es_lines: List[str]
) -> Tuple[List[str], List[str], int]:
    """
    Remove pairs that contain only punctuation, numbers, or special characters.

    Returns:
        (filtered_en, filtered_es, num_removed)
    """
    clean_en = []
    clean_es = []
    num_removed = 0

    for en, es in zip(en_lines, es_lines):
        # Remove all punctuation, whitespace, and digits to check for alphabetic content
        en_alpha = "".join(c for c in en if c.isalpha())
        es_alpha = "".join(c for c in es if c.isalpha())

        # Keep only if both have at least some alphabetic characters
        if en_alpha and es_alpha:
            clean_en.append(en)
            clean_es.append(es)
        else:
            num_removed += 1

    return clean_en, clean_es, num_removed


def filter_by_length(
    en_lines: List[str],
    es_lines: List[str],
    min_tokens: int = 3,
    max_tokens: int = 100,
) -> Tuple[List[str], List[str], int]:
    """
    Remove pairs where either sentence is too short or too long.

    Args:
        min_tokens: Minimum tokens per sentence (default 3)
        max_tokens: Maximum tokens per sentence (default 100)

    Returns:
        (filtered_en, filtered_es, num_removed)
    """
    clean_en = []
    clean_es = []
    num_removed = 0

    for en, es in zip(en_lines, es_lines):
        en_len = len(en.split())
        es_len = len(es.split())

        if min_tokens <= en_len <= max_tokens and min_tokens <= es_len <= max_tokens:
            clean_en.append(en)
            clean_es.append(es)
        else:
            num_removed += 1

    return clean_en, clean_es, num_removed


def filter_misaligned(
    en_lines: List[str], es_lines: List[str], max_ratio: float = 3.0
) -> Tuple[List[str], List[str], int]:
    """
    Remove pairs with extreme length ratio (likely misaligned).

    Returns:
        (filtered_en, filtered_es, num_removed)
    """
    clean_en = []
    clean_es = []
    num_removed = 0

    for en, es in zip(en_lines, es_lines):
        en_len = len(en.split())
        es_len = len(es.split())

        # Empty sentences should already be filtered by filter_empty_lines
        # But add safety check
        if en_len == 0 or es_len == 0:
            num_removed += 1
            continue

        ratio = max(en_len, es_len) / min(en_len, es_len)

        if ratio <= max_ratio:
            clean_en.append(en)
            clean_es.append(es)
        else:
            num_removed += 1

    return clean_en, clean_es, num_removed


def deduplicate(
    en_lines: List[str], es_lines: List[str]
) -> Tuple[List[str], List[str], int]:
    """
    Remove exact duplicate pairs, keeping first occurrence.

    Returns:
        (deduplicated_en, deduplicated_es, num_removed)
    """
    seen_pairs = set()
    clean_en = []
    clean_es = []
    num_duplicates = 0

    for en, es in zip(en_lines, es_lines):
        pair = (en, es)
        if pair not in seen_pairs:
            seen_pairs.add(pair)
            clean_en.append(en)
            clean_es.append(es)
        else:
            num_duplicates += 1

    return clean_en, clean_es, num_duplicates


def shuffle_parallel(
    en_lines: List[str], es_lines: List[str]
) -> Tuple[List[str], List[str]]:
    """
    Shuffle parallel sentences while maintaining alignment.

    Returns:
        (shuffled_en, shuffled_es)
    """
    # Create indices and shuffle them
    indices = list(range(len(en_lines)))
    random.shuffle(indices)

    # Reorder both lists using shuffled indices
    shuffled_en = [en_lines[i] for i in indices]
    shuffled_es = [es_lines[i] for i in indices]

    return shuffled_en, shuffled_es


def preprocess_split(
    en_path: Path,
    es_path: Path,
    output_en_path: Path,
    output_es_path: Path,
    apply_dedup: bool = True,
    min_tokens: int = 3,
    max_tokens: int = 100,
    max_ratio: float = 3.0,
) -> Dict[str, int]:
    """
    Preprocess a single data split.

    Args:
        en_path: Input English file
        es_path: Input Spanish file
        output_en_path: Output English file
        output_es_path: Output Spanish file
        apply_dedup: Whether to deduplicate (only for train)
        min_tokens: Minimum tokens per sentence
        max_tokens: Maximum tokens per sentence
        max_ratio: Maximum length ratio for alignment

    Returns:
        Statistics dictionary
    """
    print(f"\nProcessing {en_path.stem}...")

    # Load data
    en_lines, es_lines = load_parallel_data(en_path, es_path)
    original_count = len(en_lines)
    print(f"  Original pairs: {original_count:,}")

    stats = {
        "original": original_count,
        "unicode_normalized": 0,
        "html_cleaned": 0,
        "whitespace_normalized": 0,
        "empty": 0,
        "punctuation_only": 0,
        "length_filtered": 0,
        "misaligned": 0,
        "language_id_filtered": 0,
        "duplicates": 0,
    }

    # Step 1: Normalize Unicode
    en_lines, es_lines, num_unicode = normalize_unicode(en_lines, es_lines)
    stats["unicode_normalized"] = num_unicode
    print(f"  After Unicode normalization: {num_unicode:,} pairs changed")

    # Step 2: Clean HTML/XML
    en_lines, es_lines, num_html = clean_html(en_lines, es_lines)
    stats["html_cleaned"] = num_html
    print(f"  After HTML cleaning: {num_html:,} pairs changed")

    # Step 3: Normalize whitespace
    en_lines, es_lines, num_ws = normalize_whitespace(en_lines, es_lines)
    stats["whitespace_normalized"] = num_ws
    print(f"  After whitespace normalization: {num_ws:,} pairs changed")

    # Step 4: Filter empty lines
    en_lines, es_lines, num_empty = filter_empty_lines(en_lines, es_lines)
    print(
        f"  After empty line removal: {len(en_lines):,} (-{num_empty:,}, {num_empty/original_count*100:.2f}%)"
    )

    # Step 5: Filter punctuation-only pairs
    en_lines, es_lines, num_punct = filter_punctuation_only(en_lines, es_lines)
    stats["punctuation_only"] = num_punct
    print(
        f"  After punctuation filtering: {len(en_lines):,} (-{num_punct:,}, {num_punct/original_count*100:.2f}%)"
    )

    # Step 6: Filter by length (min and max tokens)
    en_lines, es_lines, num_length = filter_by_length(
        en_lines,
        es_lines,
        min_tokens=min_tokens,
        max_tokens=max_tokens,
    )
    stats["length_filtered"] = num_length
    print(
        f"  After length filtering: {len(en_lines):,} (-{num_length:,}, {num_length/original_count*100:.2f}%)"
    )

    # Step 7: Filter misaligned pairs
    en_lines, es_lines, num_misaligned = filter_misaligned(
        en_lines, es_lines, max_ratio=max_ratio
    )
    stats["misaligned"] = num_misaligned
    print(
        f"  After misalignment removal: {len(en_lines):,} (-{num_misaligned:,}, {num_misaligned/original_count*100:.2f}%)"
    )

    # Step 8: Deduplicate (training only)
    if apply_dedup:
        en_lines, es_lines, num_dup = deduplicate(en_lines, es_lines)
        stats["duplicates"] = num_dup
        print(
            f"  After deduplication: {len(en_lines):,} (-{num_dup:,}, {num_dup/original_count*100:.2f}%)"
        )

    # Step 9: Shuffle (training only)
    if apply_dedup:  # Use same flag as dedup for training data
        en_lines, es_lines = shuffle_parallel(en_lines, es_lines)
        print(f"  Data shuffled")

    # Save cleaned data
    save_parallel_data(en_lines, es_lines, output_en_path, output_es_path)
    stats["final"] = len(en_lines)

    total_removed = original_count - len(en_lines)
    print(
        f"  Final: {len(en_lines):,} pairs ({total_removed:,} removed, {total_removed/original_count*100:.2f}%)"
    )

    return stats


def preprocess_all_splits():
    parser = argparse.ArgumentParser(description="Preprocess OPUS-100 EN-ES data")
    parser.add_argument(
        "--raw-dir",
        type=str,
        default="data/raw/opus-100",
        help="Directory containing raw data",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed/opus-100",
        help="Directory for cleaned data",
    )
    parser.add_argument(
        "--min-tokens",
        type=int,
        default=3,
        help="Minimum tokens per sentence (default: 3)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=100,
        help="Maximum tokens per sentence (default: 100)",
    )
    parser.add_argument(
        "--max-ratio",
        type=float,
        default=3.0,
        help="Maximum length ratio for alignment (default: 3.0)",
    )

    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)

    print("\nStarting OPUS-100 EN-ES Data Preprocessing\n")

    print(f"Min tokens:       {args.min_tokens}")
    print(f"Max tokens:       {args.max_tokens}")
    print(f"Max ratio:        {args.max_ratio}")

    # Process training data (with deduplication)
    train_stats = preprocess_split(
        en_path=raw_dir / "opus.en-es-train.en",
        es_path=raw_dir / "opus.en-es-train.es",
        output_en_path=output_dir / "train.en",
        output_es_path=output_dir / "train.es",
        apply_dedup=True,
        min_tokens=args.min_tokens,
        max_tokens=args.max_tokens,
        max_ratio=args.max_ratio,
    )

    # Process dev data (no deduplication)
    dev_stats = preprocess_split(
        en_path=raw_dir / "opus.en-es-dev.en",
        es_path=raw_dir / "opus.en-es-dev.es",
        output_en_path=output_dir / "dev.en",
        output_es_path=output_dir / "dev.es",
        apply_dedup=False,
        min_tokens=args.min_tokens,
        max_tokens=args.max_tokens,
        max_ratio=args.max_ratio,
    )

    # Process test data (no deduplication)
    test_stats = preprocess_split(
        en_path=raw_dir / "opus.en-es-test.en",
        es_path=raw_dir / "opus.en-es-test.es",
        output_en_path=output_dir / "test.en",
        output_es_path=output_dir / "test.es",
        apply_dedup=False,
        min_tokens=args.min_tokens,
        max_tokens=args.max_tokens,
        max_ratio=args.max_ratio,
    )

    # Save statistics to JSON
    all_stats = {
        "train": train_stats,
        "dev": dev_stats,
        "test": test_stats,
        "config": {
            "min_tokens": args.min_tokens,
            "max_tokens": args.max_tokens,
            "max_ratio": args.max_ratio,
        },
    }

    stats_file = output_dir.parent / "preprocessing_stats.json"
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(all_stats, f, indent=2)

    print(f"\nPreprocessed data and statistics saved to: {output_dir.parent}")
    print("\nPreprocessing complete!")


if __name__ == "__main__":
    preprocess_all_splits()
