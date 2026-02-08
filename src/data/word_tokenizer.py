"""
Word-level tokenizer for neural machine translation.

Builds vocabulary from training data using frequency-based filtering, where
each word is treated as an atomic unit.

HOW IT WORKS:
    Training: Count word frequencies, filter by min_freq, limit to max_size
    Encoding: Split text on whitespace, map words to vocabulary indices

Special tokens: PAD, UNK, BOS, EOS
"""

from pathlib import Path
from typing import List, Optional, Tuple
from collections import Counter
import pickle


class WordTokenizer:
    """
    Word-level tokenizer with integrated vocabulary.

    Treats each word as an atomic token with frequency-based vocabulary building.
    """

    # Special tokens
    PAD_TOKEN = "<PAD>"  # Padding token for batch processing
    UNK_TOKEN = "<UNK>"  # Unknown token for out-of-vocabulary words
    BOS_TOKEN = "<BOS>"  # Beginning-of-sequence token
    EOS_TOKEN = "<EOS>"  # End-of-sequence token

    def __init__(
        self,
        min_freq: int = 5,
        max_size: Optional[int] = None,
        lowercase: bool = True,
    ):
        """
        Initialize word tokenizer.

        Args:
            min_freq: Minimum frequency for word inclusion
            max_size: Maximum vocabulary size (None for unlimited)
            lowercase: Whether to convert text to lowercase
        """
        self.min_freq = min_freq
        self.max_size = max_size
        self.lowercase = lowercase

        self.special_tokens = [
            self.PAD_TOKEN,
            self.UNK_TOKEN,
            self.BOS_TOKEN,
            self.EOS_TOKEN,
        ]
        self.word2idx = {token: idx for idx, token in enumerate(self.special_tokens)}
        self.idx2word = {idx: token for idx, token in enumerate(self.special_tokens)}

    def _tokenize(self, text: str) -> List[str]:
        """
        Split text into words and strip punctuation.

        Args:
            text: Input text to tokenize

        Returns:
            List of word tokens with punctuation removed
        """
        if self.lowercase:
            text = text.lower()

        # Strip common punctuation from text
        import string
        translator = str.maketrans('', '', string.punctuation)
        text = text.translate(translator)

        return text.strip().split()

    def train(self, file_paths: List[Path]) -> None:
        """
        Build vocabulary from text files.

        Counts word frequencies, filters by min_freq, and optionally limits
        to max_size most frequent words.

        Args:
            file_paths: List of text files to build vocabulary from
        """
        # Count word frequencies across all files
        word_counts = Counter()
        for file_path in file_paths:
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    word_counts.update(self._tokenize(line))

        print(f"  Initial vocabulary: {len(word_counts):,} unique words")

        # Filter by frequency threshold
        filtered_words = [
            word for word, count in word_counts.items() if count >= self.min_freq
        ]
        num_filtered = len(word_counts) - len(filtered_words)
        print(f"  Filtered out: {num_filtered:,} words (below min_freq={self.min_freq})")

        # Sort by frequency (most frequent first)
        filtered_words.sort(key=lambda w: word_counts[w], reverse=True)

        # Limit vocabulary size if specified
        if self.max_size is not None:
            filtered_words = filtered_words[: self.max_size - len(self.special_tokens)]

        # Build word-to-index mappings
        for word in filtered_words:
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word

        # Calculate and display coverage statistics
        total_tokens = sum(word_counts.values())
        covered_tokens = sum(word_counts[word] for word in filtered_words)
        coverage = (covered_tokens / total_tokens * 100) if total_tokens > 0 else 0
        print(f"  Coverage: {coverage:.2f}% of training tokens")

    def encode(
        self, text: str, add_bos: bool = False, add_eos: bool = False
    ) -> List[int]:
        """
        Convert text to token indices.

        Args:
            text: Input text to encode
            add_bos: Whether to add beginning-of-sequence token
            add_eos: Whether to add end-of-sequence token

        Returns:
            List of token indices
        """
        tokens = self._tokenize(text)
        indices = []

        if add_bos:
            indices.append(self.word2idx[self.BOS_TOKEN])

        # Map words to indices (UNK for out-of-vocabulary)
        unk_idx = self.word2idx[self.UNK_TOKEN]
        indices.extend([self.word2idx.get(token, unk_idx) for token in tokens])

        if add_eos:
            indices.append(self.word2idx[self.EOS_TOKEN])

        return indices

    def encode_batch(
        self, texts: List[str], add_bos: bool = False, add_eos: bool = False
    ) -> List[List[int]]:
        """
        Encode multiple texts in batch.

        Args:
            texts: List of text strings to encode
            add_bos: Whether to add beginning-of-sequence token to each text
            add_eos: Whether to add end-of-sequence token to each text

        Returns:
            List of token index lists, one per input text
        """
        return [self.encode(text, add_bos=add_bos, add_eos=add_eos) for text in texts]

    def decode(self, indices: List[int], remove_special: bool = True) -> str:
        """
        Convert token indices back to text.

        Args:
            indices: List of token indices to decode
            remove_special: Whether to remove special tokens from output

        Returns:
            Decoded text string
        """
        special_set = set(self.special_tokens) if remove_special else set()
        words = []
        for idx in indices:
            word = self.idx2word.get(idx, self.UNK_TOKEN)
            if word not in special_set:
                words.append(word)
        return " ".join(words)

    def decode_batch(
        self, batch_indices: List[List[int]], remove_special: bool = True
    ) -> List[str]:
        """
        Decode multiple sequences in batch.

        Args:
            batch_indices: List of token index lists to decode
            remove_special: Whether to remove special tokens from each output

        Returns:
            List of decoded text strings
        """
        return [
            self.decode(indices, remove_special=remove_special)
            for indices in batch_indices
        ]

    def save(self, path: Path) -> None:
        """
        Save tokenizer to disk using pickle.

        Args:
            path: Path where the tokenizer will be saved
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "word2idx": self.word2idx,
            "idx2word": self.idx2word,
            "min_freq": self.min_freq,
            "max_size": self.max_size,
            "lowercase": self.lowercase,
            "special_tokens": self.special_tokens,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls, path: Path) -> "WordTokenizer":
        """
        Load tokenizer from disk.

        Args:
            path: Path to the saved tokenizer file

        Returns:
            Loaded WordTokenizer instance
        """
        with open(path, "rb") as f:
            state = pickle.load(f)

        tokenizer = cls(
            min_freq=state["min_freq"],
            max_size=state["max_size"],
            lowercase=state["lowercase"],
        )
        tokenizer.word2idx = state["word2idx"]
        tokenizer.idx2word = state["idx2word"]
        tokenizer.special_tokens = state["special_tokens"]

        return tokenizer


def build_word_tokenizers(
    train_en_path: Path,
    train_es_path: Path,
    output_dir: Path,
    min_freq: int = 5,
    max_size: Optional[int] = None,
) -> Tuple[WordTokenizer, WordTokenizer]:
    """
    Build separate word tokenizers for English and Spanish.

    Creates two independent tokenizers, one for each language, which allows
    each vocabulary to be optimized for its language's specific characteristics.

    Args:
        train_en_path: Path to English training data file
        train_es_path: Path to Spanish training data file
        output_dir: Directory where the trained tokenizers will be saved
        min_freq: Minimum word frequency for inclusion (default: 5)
        max_size: Maximum vocabulary size (default: None for unlimited)

    Returns:
        Tuple of (english_tokenizer, spanish_tokenizer)
    """
    print("\nBuilding word tokenizers\n")

    print(f"Min frequency: {min_freq}")
    print(f"Max vocab size: {max_size or 'unlimited'}")

    # Build English tokenizer
    print(f"\nProcessing {train_en_path.name}...")
    tokenizer_en = WordTokenizer(min_freq=min_freq, max_size=max_size)
    tokenizer_en.train([train_en_path])
    tokenizer_en.save(output_dir / "word_vocab_en.pkl")
    print(f"  Final vocabulary size: {len(tokenizer_en.word2idx):,}")

    # Build Spanish tokenizer
    print(f"\nProcessing {train_es_path.name}...")
    tokenizer_es = WordTokenizer(min_freq=min_freq, max_size=max_size)
    tokenizer_es.train([train_es_path])
    tokenizer_es.save(output_dir / "word_vocab_es.pkl")
    print(f"  Final vocabulary size: {len(tokenizer_es.word2idx):,}")

    print(f"\nVocabulary saved to: {output_dir}")
    print("\nTokenizer build complete!")

    return tokenizer_en, tokenizer_es


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build word-level tokenizers for NMT")
    parser.add_argument(
        "--train-en",
        type=str,
        default="data/processed/opus-100/train.en",
        help="Path to English training data",
    )
    parser.add_argument(
        "--train-es",
        type=str,
        default="data/processed/opus-100/train.es",
        help="Path to Spanish training data",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/vocab",
        help="Output directory for tokenizers",
    )
    parser.add_argument(
        "--min-freq",
        type=int,
        default=5,
        help="Minimum word frequency (default: 5)",
    )
    parser.add_argument(
        "--max-size",
        type=int,
        default=None,
        help="Maximum vocabulary size (default: None)",
    )

    args = parser.parse_args()

    build_word_tokenizers(
        train_en_path=Path(args.train_en),
        train_es_path=Path(args.train_es),
        output_dir=Path(args.output_dir),
        min_freq=args.min_freq,
        max_size=args.max_size,
    )
