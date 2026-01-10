"""
Byte Pair Encoding (BPE) tokenizer for neural machine translation.

Implements the BPE algorithm from scratch for subword tokenization, which helps
handle out-of-vocabulary words by breaking them into smaller units.

================================================================================
HOW IT WORKS
================================================================================

BPE learns a vocabulary of subword units by iteratively merging the most
frequent adjacent token pairs. This allows rare/unseen words to be represented
as sequences of known subwords.

TRAINING:
    1. Split words into characters, add end-of-word markers (</w>)
       "hello" → ('h', 'e', 'l', 'l', 'o', '</w>')

    2. Find most frequent pair and merge it everywhere
       Most frequent: ('h', 'e') → Merge to 'he'
       Result: ('he', 'l', 'l', 'o', '</w>')

    3. Update frequencies incrementally using inverted index (pair → words)
       This enables O(1) lookups instead of O(|V|) full vocabulary scans

    4. Repeat until vocab size reached or frequency threshold hit

ENCODING:
    1. Split text into characters with end-of-word markers
    2. Apply merges greedily in order learned (using merge rank for determinism)
    3. Convert tokens to indices (UNK for unknowns, special tokens: PAD/BOS/EOS)

DATA STRUCTURES:
    • merges: Ordered list of (pair, merged_token) operations learned in training
    • token2idx/idx2token: Bidirectional vocabulary mappings for encode/decode
    • vocab, pair_to_words: Training-only structures for learning merges

Reference: Sennrich et al. (2016) - https://arxiv.org/abs/1508.07909
================================================================================
"""

from pathlib import Path
from typing import List, Dict, Tuple, Set
from collections import Counter, defaultdict
import pickle
from tqdm import tqdm


class BPETokenizer:
    """
    BPE tokenizer with subword vocabulary.

    Uses byte pair encoding to learn a vocabulary of subword units that can
    represent any text by iteratively merging the most frequent character pairs.
    """

    # Special tokens
    PAD_TOKEN = "<PAD>"  # Padding token for batch processing
    UNK_TOKEN = "<UNK>"  # Unknown token for out-of-vocabulary items
    BOS_TOKEN = "<BOS>"  # Beginning-of-sequence token
    EOS_TOKEN = "<EOS>"  # End-of-sequence token
    EOW_TOKEN = "</w>"  # End-of-word marker to preserve word boundaries

    def __init__(
        self, vocab_size: int = 32000, min_freq: int = 2, lowercase: bool = True
    ):
        """
        Initialize BPE tokenizer.

        Args:
            vocab_size: Target vocabulary size (including special tokens)
            min_freq: Minimum frequency for merge operations
            lowercase: Convert text to lowercase
        """
        self.vocab_size = vocab_size
        self.min_freq = min_freq
        self.lowercase = lowercase

        # Special tokens at fixed indices
        self.special_tokens = [
            self.PAD_TOKEN,
            self.UNK_TOKEN,
            self.BOS_TOKEN,
            self.EOS_TOKEN,
        ]
        self.token2idx = {}
        self.idx2token = {}
        for idx, token in enumerate(self.special_tokens):
            self.token2idx[token] = idx
            self.idx2token[idx] = token

        # BPE merge operations (ordered) - single source of truth
        self.merges = []  # List of (pair, new_token) tuples

    # ==================== Training Methods ====================

    def _preprocess(self, text: str) -> str:
        """
        Normalize text before tokenization.

        Args:
            text: Input text to normalize

        Returns:
            Normalized text (lowercased if configured, whitespace stripped)
        """
        if self.lowercase:
            text = text.lower()
        text = text.strip()
        return text

    def _get_initial_vocab(self, texts: List[str]) -> Dict[Tuple[str, ...], int]:
        """
        Build character-level vocabulary from texts.

        Each word is split into individual characters and an end-of-word marker
        is appended. Word frequencies are counted.

        Args:
            texts: List of text strings to process

        Returns:
            Dict mapping character tuples to their frequencies
            Example: {('h', 'e', 'l', 'l', 'o', '</w>'): 5}
        """
        word_freqs = Counter()

        for text in texts:
            text = self._preprocess(text)
            words = text.split()
            for word in words:
                # Split into characters and add end-of-word marker
                char_word = tuple(word) + (self.EOW_TOKEN,)
                word_freqs[char_word] += 1

        return word_freqs

    # Training helper methods

    def _iter_pairs(self, word: Tuple[str, ...]) -> range:
        """
        Get range for iterating over adjacent pair indices in a word.

        Args:
            word: Tuple of tokens

        Returns:
            Range object for iterating pair indices
        """
        return range(len(word) - 1)

    def _get_pairs_from_word(self, word: Tuple[str, ...]) -> List[Tuple[str, str]]:
        """
        Extract all adjacent pairs from a word.

        Args:
            word: Tuple of tokens

        Returns:
            List of adjacent token pairs
        """
        return [(word[i], word[i + 1]) for i in self._iter_pairs(word)]

    def _add_word_to_index(
        self,
        word: Tuple[str, ...],
        pair_to_words: Dict[Tuple[str, str], Set[Tuple[str, ...]]],
    ) -> None:
        """
        Add a word to the inverted index.

        Updates the inverted index so each pair in the word points to this word.
        This enables O(1) lookup of all words containing a given pair.

        Args:
            word: Word as tuple of tokens to add
            pair_to_words: Inverted index mapping pairs to words containing them
        """
        for pair in self._get_pairs_from_word(word):
            pair_to_words[pair].add(word)

    def _remove_word_from_index(
        self,
        word: Tuple[str, ...],
        pair_to_words: Dict[Tuple[str, str], Set[Tuple[str, ...]]],
    ) -> None:
        """
        Remove a word from the inverted index.

        Cleans up empty entries to keep the index memory-efficient.

        Args:
            word: Word as tuple of tokens to remove
            pair_to_words: Inverted index mapping pairs to words containing them
        """
        for pair in self._get_pairs_from_word(word):
            pair_to_words[pair].discard(word)
            if not pair_to_words[pair]:
                del pair_to_words[pair]

    def _is_merge_pair(
        self, word: Tuple[str, ...], i: int, first_token: str, second_token: str
    ) -> bool:
        """
        Check if position i contains the target pair to merge.

        Args:
            word: Word as tuple of tokens
            i: Position to check
            first_token: First token of the pair
            second_token: Second token of the pair

        Returns:
            True if the pair exists at position i, False otherwise
        """
        return (
            i < len(word) - 1 and word[i] == first_token and word[i + 1] == second_token
        )

    def _merge_word(
        self,
        word: Tuple[str, ...],
        freq: int,
        first_token: str,
        second_token: str,
        replacement: str,
    ) -> Tuple[Tuple[str, ...], Dict[Tuple[str, str], int]]:
        """
        Apply merge rule to a single word and compute pair frequency deltas.

        This is a pure function that doesn't modify any global state.

        Args:
            word: Original word as tuple of tokens
            freq: Frequency of this word
            first_token: First token of merge pair
            second_token: Second token of merge pair
            replacement: Merged token to replace (first_token, second_token)

        Returns:
            Tuple of (new_word, pair_frequency_changes) where:
            - new_word: Word with merges applied
            - pair_frequency_changes: Dict mapping pairs to frequency changes
        """
        pair_frequency_changes = defaultdict(int)
        new_word = []
        i = 0

        while i < len(word):
            if self._is_merge_pair(word, i, first_token, second_token):
                # Found pair to merge - update frequency statistics
                # Remove old pairs: left context + pair + right context
                if i > 0:
                    pair_frequency_changes[(word[i - 1], first_token)] -= freq
                if i + 2 < len(word):
                    pair_frequency_changes[(second_token, word[i + 2])] -= freq
                pair_frequency_changes[(first_token, second_token)] -= freq

                # Add new pairs around the merged token
                if i > 0:
                    pair_frequency_changes[(word[i - 1], replacement)] += freq
                if i + 2 < len(word):
                    pair_frequency_changes[(replacement, word[i + 2])] += freq

                new_word.append(replacement)
                i += 2  # Skip both tokens in the pair
            else:
                new_word.append(word[i])
                i += 1

        return tuple(new_word), pair_frequency_changes

    # Training pipeline methods

    def _read_training_data(self, file_paths: List[Path]) -> List[str]:
        """
        Read and combine all training files.

        Args:
            file_paths: List of text files to read

        Returns:
            List of text lines from all files
        """
        texts = []
        for file_path in file_paths:
            with open(file_path, "r", encoding="utf-8") as f:
                texts.extend(f.readlines())
        return texts

    def _build_initial_vocab(self, texts: List[str]) -> Dict[Tuple[str, ...], int]:
        """
        Build character-level vocabulary from texts.

        This is the starting point for BPE training. Each unique word is
        represented as a tuple of characters plus an end-of-word marker.

        Args:
            texts: List of text lines

        Returns:
            Dict mapping character-level words to their frequencies
        """
        vocab = self._get_initial_vocab(texts)

        # Calculate and display statistics
        unique_tokens = set()
        for word in vocab.keys():
            unique_tokens.update(word)

        print(f"  Character set size: {len(unique_tokens):,}")
        print(f"  Initial vocabulary: {len(vocab):,} unique words")

        return vocab

    def _initialize_pair_statistics(
        self, vocab: Dict[Tuple[str, ...], int]
    ) -> Tuple[Counter, Dict[Tuple[str, str], Set[Tuple[str, ...]]]]:
        """
        Initialize pair frequencies and inverted index for efficient merging.

        The pair frequencies track how often each adjacent token pair appears,
        while the inverted index allows O(1) lookup of all words containing a pair.

        Args:
            vocab: Character-level vocabulary with word frequencies

        Returns:
            Tuple of (pair_frequencies, inverted_index) where:
            - pair_frequencies: Counter mapping pairs to total frequency
            - inverted_index: Dict mapping pairs to sets of words containing them
        """
        # Count how often each adjacent pair appears across all words
        pairs = Counter()
        for word, freq in vocab.items():
            for pair in self._get_pairs_from_word(word):
                pairs[pair] += freq

        # Build inverted index for fast lookup: which words contain each pair?
        pair_to_words = defaultdict(set)
        for word in vocab.keys():
            for pair in self._get_pairs_from_word(word):
                pair_to_words[pair].add(word)

        return pairs, pair_to_words

    def _learn_merges(
        self,
        vocab: Dict[Tuple[str, ...], int],
        pairs: Counter,
        pair_to_words: Dict[Tuple[str, str], Set[Tuple[str, ...]]],
    ) -> None:
        """
        Learn BPE merge operations by iteratively merging most frequent pairs.

        This is the core BPE algorithm: repeatedly find the most frequent pair
        of adjacent tokens and merge them into a single token. Stops when the
        target vocabulary size is reached or frequency threshold is hit.

        Modifies self.merges in-place.

        Args:
            vocab: Character-level vocabulary (modified in-place)
            pairs: Pair frequency counter (modified in-place)
            pair_to_words: Inverted index (modified in-place)
        """
        # Calculate how many merges we need to reach target vocab size
        unique_tokens = set()
        for word in vocab.keys():
            unique_tokens.update(word)

        num_merges = self.vocab_size - len(self.special_tokens) - len(unique_tokens)
        print(f"\nLearning {num_merges:,} merge operations...")

        # Iteratively learn merges with progress tracking
        pbar = tqdm(range(num_merges), desc="Learning merges", unit="merge")
        for _ in pbar:
            if not pairs:
                pbar.set_postfix_str(f"Stopped: no more pairs")
                break

            # Find the most frequent pair
            best_pair = max(pairs, key=pairs.get)
            best_freq = pairs[best_pair]

            # Stop if frequency falls below threshold
            if best_freq < self.min_freq:
                pbar.set_postfix_str(f"Stopped: frequency threshold")
                break

            # Merge the best pair across all affected words
            first_token, second_token = best_pair
            replacement = first_token + second_token
            changes = defaultdict(int)

            # Use inverted index to find all words with this pair (O(1) lookup)
            words_to_update = [(word, vocab[word]) for word in pair_to_words[best_pair]]

            # Process each affected word
            for word, freq in words_to_update:
                # Remove old word from vocabulary and index
                del vocab[word]
                self._remove_word_from_index(word, pair_to_words)

                # Apply merge and get frequency changes for adjacent pairs
                new_word, pair_frequency_changes = self._merge_word(
                    word, freq, first_token, second_token, replacement
                )

                # Add updated word to vocabulary and index
                vocab[new_word] = freq
                self._add_word_to_index(new_word, pair_to_words)

                # Accumulate frequency changes across all words
                for pair, delta in pair_frequency_changes.items():
                    changes[pair] += delta

            # Update global pair frequencies based on accumulated changes
            for pair, delta in changes.items():
                pairs[pair] += delta
                if pairs[pair] <= 0:
                    del pairs[pair]

            # Record this merge operation for later use during encoding
            self.merges.append((best_pair, replacement))

            # Update progress bar
            pbar.set_postfix_str(f"freq={best_freq}")

    def _build_final_vocabulary(self, vocab: Dict[Tuple[str, ...], int]) -> None:
        """
        Build final token-to-index mappings from all learned tokens.

        Extracts all unique tokens from the vocabulary (including merged tokens)
        and creates bidirectional mappings for encoding and decoding.

        Args:
            vocab: Vocabulary after all merges have been applied
        """
        print("\nBuilding final vocabulary...")

        # Extract all unique tokens from the final vocabulary
        final_tokens = set()
        for word in vocab.keys():
            final_tokens.update(word)

        # Add tokens to mappings (special tokens already added in __init__)
        for token in sorted(final_tokens):
            if token not in self.token2idx:
                idx = len(self.token2idx)
                self.token2idx[token] = idx
                self.idx2token[idx] = token

    def train(self, file_paths: List[Path]) -> None:
        """
        Learn BPE merges from training data.

        Args:
            file_paths: List of text files to train on
        """
        # Phase 1: Load all training texts from files
        texts = self._read_training_data(file_paths)

        # Phase 2: Build character-level vocabulary with word frequencies
        vocab = self._build_initial_vocab(texts)

        # Phase 3: Initialize pair statistics and inverted index for efficient lookup
        pairs, pair_to_words = self._initialize_pair_statistics(vocab)

        # Phase 4: Iteratively learn merge operations by combining most frequent pairs
        self._learn_merges(vocab, pairs, pair_to_words)

        # Phase 5: Build final token-to-index mappings from all learned tokens
        self._build_final_vocabulary(vocab)

    # ==================== Encoding/Decoding Methods ====================

    def _find_best_merge(
        self, tokens: List[str], merge_rank: Dict[Tuple[str, str], int]
    ) -> Tuple[Tuple[str, str], int]:
        """
        Find the pair with the lowest merge rank (earliest learned) in a sequence.

        During encoding, we want to apply merges in the same order they were
        learned during training, so we find the pair with the lowest rank.

        Args:
            tokens: List of token strings to scan
            merge_rank: Dict mapping pairs to their merge rank (lower = earlier)

        Returns:
            Tuple of (best_pair, best_idx) where:
            - best_pair: The pair to merge, or None if no valid pair found
            - best_idx: Position of the pair, or None if no valid pair found
        """
        best_pair = None
        best_idx = None
        best_rank = float("inf")

        # Scan all adjacent pairs and find the one with lowest rank
        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i + 1])
            rank = merge_rank.get(pair)
            if rank is not None and rank < best_rank:
                best_rank = rank
                best_pair = pair
                best_idx = i

        return best_pair, best_idx

    def _apply_merges(self, tokens: List[str]) -> List[str]:
        """
        Apply learned merges to token sequence using greedy algorithm.

        Iteratively finds the pair with the lowest merge rank (earliest in training)
        and merges it, repeating until no more valid pairs exist. This ensures
        merges are applied in the same order they were learned during training.

        Time complexity: O(k * n) where k is number of merges applied and
        n is the sequence length.

        Args:
            tokens: List of token strings to merge

        Returns:
            List of tokens after applying all applicable merges
        """
        if len(tokens) <= 1:
            return tokens

        # Build lookup tables for efficient merge application
        pair_to_replacement = {pair: replacement for pair, replacement in self.merges}
        merge_rank = {pair: rank for rank, (pair, _) in enumerate(self.merges)}

        # Repeatedly apply the highest-priority merge until none remain
        while len(tokens) > 1:
            best_pair, best_idx = self._find_best_merge(tokens, merge_rank)

            # Stop if no mergeable pair exists in current sequence
            if best_pair is None:
                break

            # Replace the pair with its merged token
            tokens[best_idx : best_idx + 2] = [pair_to_replacement[best_pair]]

        return tokens

    def encode(
        self, text: str, add_bos: bool = False, add_eos: bool = False
    ) -> List[int]:
        """
        Convert text to token indices using learned BPE vocabulary.

        The text is split into words, each word is tokenized using learned
        merges, and tokens are converted to their vocabulary indices.

        Args:
            text: Input text to encode
            add_bos: Whether to add beginning-of-sequence token
            add_eos: Whether to add end-of-sequence token

        Returns:
            List of token indices representing the encoded text
        """
        text = self._preprocess(text)
        words = text.split()

        indices = []
        if add_bos:
            indices.append(self.token2idx[self.BOS_TOKEN])

        for word in words:
            # Start with character-level representation
            tokens = list(word) + [self.EOW_TOKEN]

            # Apply learned BPE merges
            tokens = self._apply_merges(tokens)

            # Convert tokens to vocabulary indices
            for token in tokens:
                if token in self.token2idx:
                    indices.append(self.token2idx[token])
                else:
                    # Use UNK token for unknown tokens
                    indices.append(self.token2idx[self.UNK_TOKEN])

        if add_eos:
            indices.append(self.token2idx[self.EOS_TOKEN])

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
        return [self.encode(text, add_bos, add_eos) for text in texts]

    def decode(self, indices: List[int], remove_special: bool = True) -> str:
        """
        Convert token indices back to text.

        Args:
            indices: List of token indices to decode
            remove_special: Whether to remove special tokens (PAD, UNK, BOS, EOS)

        Returns:
            Decoded text string
        """
        tokens = []

        for idx in indices:
            # Look up token, use UNK if index not found
            token = self.idx2token.get(idx, self.UNK_TOKEN)

            # Skip special tokens if requested
            if remove_special and token in self.special_tokens:
                continue

            tokens.append(token)

        # Reconstruct text by joining tokens and converting EOW markers to spaces
        text = "".join(tokens)
        text = text.replace(self.EOW_TOKEN, " ")
        return text.strip()

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
        return [self.decode(indices, remove_special) for indices in batch_indices]

    # ==================== Persistence Methods ====================

    def save(self, path: Path) -> None:
        """
        Save tokenizer to disk using pickle.

        Args:
            path: Path where the tokenizer will be saved
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path) -> "BPETokenizer":
        """
        Load tokenizer from disk.

        Args:
            path: Path to the saved tokenizer file

        Returns:
            Loaded BPETokenizer instance
        """
        with open(path, "rb") as f:
            return pickle.load(f)


def build_bpe_tokenizer(
    train_en_path: Path,
    train_es_path: Path,
    output_dir: Path,
    vocab_size: int = 32000,
    min_freq: int = 2,
) -> BPETokenizer:
    """
    Build shared BPE tokenizer for English and Spanish data.

    Creates a single tokenizer trained on both languages, enabling better
    representation of cognates and shared vocabulary between the languages.

    Args:
        train_en_path: Path to English training data file
        train_es_path: Path to Spanish training data file
        output_dir: Directory where the trained tokenizer will be saved
        vocab_size: Target vocabulary size (default: 32000)
        min_freq: Minimum frequency threshold for merge operations (default: 2)

    Returns:
        Trained BPETokenizer instance
    """
    print("\nBuilding BPE tokenizer\n")

    print(f"Target vocab size: {vocab_size:,}")
    print(f"Min merge frequency: {min_freq}")

    print(f"\nProcessing {train_en_path.name}, {train_es_path.name}...")
    tokenizer = BPETokenizer(vocab_size=vocab_size, min_freq=min_freq)
    tokenizer.train([train_en_path, train_es_path])

    # Save the trained tokenizer
    tokenizer.save(output_dir / "bpe_vocab_shared.pkl")
    print(f"  Final vocabulary size: {len(tokenizer.token2idx):,}")
    print(f"  Merge operations: {len(tokenizer.merges):,}")

    print(f"\nVocabulary saved to: {output_dir}")
    print("\nBPE tokenizer build complete!")

    return tokenizer


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build BPE tokenizer for NMT")
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
        help="Output directory for tokenizer",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=32000,
        help="Target vocabulary size (default: 32000)",
    )
    parser.add_argument(
        "--min-freq",
        type=int,
        default=2,
        help="Minimum frequency for merge operations (default: 2)",
    )

    args = parser.parse_args()

    build_bpe_tokenizer(
        train_en_path=Path(args.train_en),
        train_es_path=Path(args.train_es),
        output_dir=Path(args.output_dir),
        vocab_size=args.vocab_size,
        min_freq=args.min_freq,
    )
