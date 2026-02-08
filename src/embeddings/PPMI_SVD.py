"""
PPMI + SVD word embeddings implementation.

Builds embeddings via Positive Pointwise Mutual Information (PPMI) on
co-occurrence counts followed by truncated SVD dimensionality reduction.

METHODOLOGY:
    1. Use pre-trained WordTokenizer vocabulary
    2. Tokenize corpus using WordTokenizer (ensures consistency with vocabulary)
    3. Build co-occurrence matrix with sliding window and distance weighting
    4. Compute PMI and clip negative values (PPMI)
    5. Apply truncated SVD to get dense embeddings

REQUIRED COMPONENTS:
    - Vocabulary cutoff: Limit to top-N words (prevents matrix explosion)
    - Context window weighting: Weight nearby words more than distant ones
    - Truncated SVD: Keep only top-d dimensions
"""

from pathlib import Path
from typing import List, Optional, Dict
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import svds
import pickle
import sys
from tqdm import tqdm

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.data.word_tokenizer import WordTokenizer
from src.embeddings.base import WordEmbedding


class PPMI_SVD(WordEmbedding):
    """
    PPMI + SVD word embeddings with distance-weighted co-occurrence.

    Uses a pre-trained WordTokenizer vocabulary, builds co-occurrence matrix
    with sliding window, and applies PPMI transformation followed by SVD.
    """

    def __init__(
        self,
        tokenizer: WordTokenizer,
        window_size: int = 5,
        embedding_dim: int = 300,
        weighting: str = "harmonic",  # "uniform", "harmonic", or "linear"
    ):
        """
        Initialize PPMI-SVD model.

        Args:
            tokenizer: Pre-trained WordTokenizer with vocabulary
            window_size: Context window radius (words on each side)
            embedding_dim: Target embedding dimensionality
            weighting: Distance weighting scheme for context words
                      "uniform": All context words weighted equally
                      "harmonic": Weight = 1/distance
                      "linear": Weight = (window_size - distance + 1) / window_size
        """
        super().__init__(tokenizer, embedding_dim, window_size)
        self.weighting = weighting

        # Import vocabulary from tokenizer (exclude special tokens)
        # PPMI_SVD requires re-indexing for dense matrix operations
        self.word2idx: Dict[str, int] = {}
        self.idx2word: Dict[int, str] = {}
        self._import_vocabulary()

        # Embeddings matrix
        self.embeddings: Optional[np.ndarray] = None

    def _import_vocabulary(self) -> None:
        """
        Import vocabulary from WordTokenizer, excluding special tokens.
        """
        special_tokens = set(self.tokenizer.special_tokens)

        idx = 0
        for word, _ in self.tokenizer.word2idx.items():
            if word not in special_tokens:
                self.word2idx[word] = idx
                self.idx2word[idx] = word
                idx += 1

        print(
            f"Imported vocabulary: {len(self.word2idx):,} words (excluding special tokens)"
        )

    def _get_distance_weight(self, distance: int) -> float:
        """
        Calculate weight for context word based on distance.

        Args:
            distance: Distance from target word (1 to window_size)

        Returns:
            Weight value for this distance
        """
        if self.weighting == "uniform":
            return 1.0
        elif self.weighting == "harmonic":
            return 1.0 / distance
        elif self.weighting == "linear":
            return (self.window_size - distance + 1) / self.window_size
        else:
            raise ValueError(f"Unknown weighting scheme: {self.weighting}")

    def _build_cooccurrence_matrix(self, file_paths: List[Path]) -> csr_matrix:
        """
        Build weighted co-occurrence matrix from corpus.

        Uses sliding window with distance-based weighting.

        Args:
            file_paths: List of text files to process

        Returns:
            Sparse co-occurrence matrix (vocab_size x vocab_size)
        """
        print("\nBuilding co-occurrence matrix...")

        vocab_len = len(self.word2idx)
        cooc_matrix = lil_matrix((vocab_len, vocab_len), dtype=np.float32)

        total_pairs = 0

        for file_path in file_paths:
            # Count lines for progress bar
            with open(file_path, "r", encoding="utf-8") as f:
                num_lines = sum(1 for _ in f)

            with open(file_path, "r", encoding="utf-8") as f:
                for line in tqdm(f, total=num_lines, desc="  Processing lines"):
                    # Use tokenizer for consistent preprocessing with vocabulary
                    words = self.tokenizer._tokenize(line.strip())

                    # Convert to indices, skip out-of-vocab words
                    indices = [
                        self.word2idx[word] for word in words if word in self.word2idx
                    ]

                    # Sliding window over sentence
                    for i, target_idx in enumerate(indices):
                        # Look at context words within window
                        start = max(0, i - self.window_size)
                        end = min(len(indices), i + self.window_size + 1)

                        for j in range(start, end):
                            if i == j:  # Skip the target word itself
                                continue

                            context_idx = indices[j]
                            distance = abs(i - j)
                            weight = self._get_distance_weight(distance)

                            cooc_matrix[target_idx, context_idx] += weight
                            total_pairs += 1

        print(f"  Total weighted pairs: {total_pairs:,}")

        # Convert to CSR format for efficient operations
        return cooc_matrix.tocsr()

    def _compute_ppmi(self, cooc_matrix: csr_matrix) -> csr_matrix:
        """
        Compute Positive PMI from co-occurrence matrix.

        PMI(w, c) = log( P(w,c) / (P(w) * P(c)) )
        PPMI(w, c) = max(0, PMI(w, c))

        Args:
            cooc_matrix: Co-occurrence count matrix

        Returns:
            PPMI matrix (same shape as input)
        """
        print("\nComputing PPMI...")

        # Calculate probabilities
        total_counts = cooc_matrix.sum()
        word_counts = np.array(cooc_matrix.sum(axis=1)).flatten()
        context_counts = np.array(cooc_matrix.sum(axis=0)).flatten()

        # Work with sparse matrix directly - only process non-zero entries
        # Convert to COO format for efficient iteration over non-zero entries
        cooc_coo = cooc_matrix.tocoo()

        total_entries = cooc_matrix.shape[0] * cooc_matrix.shape[1]
        density = 100 * cooc_coo.nnz / total_entries
        print(f"  Non-zero entries: {cooc_coo.nnz:,} ({density:.4f}% dense)")
        print("  Computing PMI values...")

        # Get arrays of row indices, column indices, and co-occurrence counts
        rows = cooc_coo.row
        cols = cooc_coo.col
        data = cooc_coo.data

        # Vectorized probability calculations
        # P(w, c) = count(w, c) / total_counts
        p_wc = data / total_counts

        # P(w) = count(w) / total_counts
        p_w = word_counts[rows] / total_counts

        # P(c) = count(c) / total_counts
        p_c = context_counts[cols] / total_counts

        # PMI(w, c) = log( P(w,c) / (P(w) * P(c)) )
        pmi_values = np.log(p_wc / (p_w * p_c))

        # PPMI(w, c) = max(0, PMI(w, c))
        ppmi_values = np.maximum(0, pmi_values)

        # Build sparse PPMI matrix from positive PMI values only
        print("  Building sparse PPMI matrix...")
        mask = ppmi_values > 0
        ppmi_csr = csr_matrix(
            (ppmi_values[mask], (rows[mask], cols[mask])),
            shape=cooc_matrix.shape,
            dtype=np.float32,
        )

        # Calculate final sparsity
        total_entries = ppmi_csr.shape[0] * ppmi_csr.shape[1]
        sparsity = 100 * (1 - ppmi_csr.nnz / total_entries)
        print(f"  PPMI sparsity: {sparsity:.2f}%")

        return ppmi_csr

    def _apply_svd(self, ppmi_matrix: csr_matrix) -> np.ndarray:
        """
        Convert a PPMI co-occurrence matrix into dense word embeddings using truncated SVD.

        Each row of the returned embeddings represents a word projected into a latent space
        capturing the main co-occurrence patterns in the vocabulary.

        Steps correspond to the SVD decomposition X ≈ U Σ V^T:
          - V^T: principal co-occurrence directions in word-feature space
          - Σ: magnitude of each latent direction (importance / variance)
          - U: normalized coordinates of each word along these directions
        """

        print("\nApplying truncated SVD...")
        print(f"  Target embedding dimension: {self.embedding_dim}")

        # STEP 1: Truncated SVD
        # Decompose the PPMI matrix X ≈ U Σ V^T
        # - U: words in terms of latent patterns (unit vectors)
        # - s: singular values, σ_i = strength/importance of each latent pattern
        # - Vt: latent patterns in feature space (how patterns are composed from words)
        U, s, Vt = svds(ppmi_matrix, k=self.embedding_dim)

        # STEP 2: Sort singular values and corresponding vectors descending
        # Why: svds returns σ_i in ascending order; largest singular values correspond
        # to the most important latent patterns (largest variance / strongest co-occurrence)
        idx = np.argsort(s)[::-1]
        U = U[:, idx]  # words along most important latent patterns first
        s = s[idx]  # magnitude of each pattern

        # STEP 3: Construct embeddings by scaling words along latent patterns
        # Why sqrt? Because in PCA-style reasoning, variance along a direction = σ_i^2
        # Scaling by sqrt(σ_i) preserves the geometry of distances between words
        # Result: each row = a word vector in latent space; each column = a latent pattern
        embeddings = U * np.sqrt(s)

        # STEP 4: Interpretation of resulting embeddings
        # - embeddings.shape = (vocab_size, embedding_dim)
        # - Row i: word i's coordinates in latent space (high-level co-occurrence patterns)
        # - Column j: latent dimension j representing a principal co-occurrence pattern
        # - Larger σ_i (stronger patterns) dominate the first dimensions
        print(f"  Embedding shape: {embeddings.shape}")
        print(f"  Top singular values (strength of latent patterns): {s[:10]}")

        return embeddings

    def train(self, file_paths: List[Path]) -> None:
        """
        Train PPMI-SVD embeddings on corpus.

        Args:
            file_paths: List of text files for training
        """
        self._print_training_header(
            "PPMI-SVD",
            **{
                "Vocab size": f"{len(self.word2idx):,}",
                "Window size": self.window_size,
                "Embedding dim": self.embedding_dim,
                "Weighting": self.weighting,
            },
        )

        # Step 1: Build weighted co-occurrence matrix
        cooc_matrix = self._build_cooccurrence_matrix(file_paths)

        # Step 2: Compute PPMI
        ppmi_matrix = self._compute_ppmi(cooc_matrix)

        # Step 3: Apply truncated SVD
        self.embeddings = self._apply_svd(ppmi_matrix)

        self._print_training_footer()

    def get_embedding(self, word: str) -> Optional[np.ndarray]:
        """
        Get embedding vector for a word.

        Args:
            word: Word to look up

        Returns:
            Embedding vector, or None if word not in vocabulary
        """
        word = word.lower()
        if word not in self.word2idx:
            return None

        idx = self.word2idx[word]
        return self.embeddings[idx]

    def most_similar(self, word: str, top_k: int = 10) -> List[tuple]:
        """
        Find most similar words by cosine similarity.

        Args:
            word: Query word
            top_k: Number of similar words to return

        Returns:
            List of (word, similarity) tuples
        """
        embedding = self.get_embedding(word)
        if embedding is None:
            return []

        # Use base class template method
        word_idx = self.word2idx[word.lower()]
        results = self._find_top_k_similar(
            embedding, self.embeddings, top_k, exclude_indices={word_idx}
        )

        # Convert indices to words
        return [(self.idx2word[idx], similarity) for idx, similarity in results]

    def _get_save_state(self) -> dict:
        """
        Return PPMI_SVD-specific state for serialization.

        Returns:
            Dictionary of model-specific state
        """
        return {
            "weighting": self.weighting,
            "word2idx": self.word2idx,
            "idx2word": self.idx2word,
            "embeddings": self.embeddings,
        }

    @classmethod
    def load(cls, path: Path) -> "PPMI_SVD":
        """
        Load model from disk.

        Args:
            path: Path to saved model

        Returns:
            Loaded PPMI_SVD model
        """
        with open(path, "rb") as f:
            state = pickle.load(f)

        model = cls(
            tokenizer=state["tokenizer"],
            window_size=state["window_size"],
            embedding_dim=state["embedding_dim"],
            weighting=state["weighting"],
        )

        model.word2idx = state["word2idx"]
        model.idx2word = state["idx2word"]
        model.embeddings = state["embeddings"]

        return model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train PPMI-SVD word embeddings")
    parser.add_argument(
        "--train-file",
        type=str,
        default="data/processed/opus-100/train.en",
        help="Path to training corpus file",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="data/vocab/word_vocab_en.pkl",
        help="Path to pre-trained WordTokenizer (.pkl)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/embeddings/ppmi_svd.pkl",
        help="Output path for trained model",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=5,
        help="Context window size (default: 5)",
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=300,
        help="Embedding dimensionality (default: 300)",
    )
    parser.add_argument(
        "--weighting",
        type=str,
        choices=["uniform", "harmonic", "linear"],
        default="harmonic",
        help="Distance weighting scheme (default: harmonic)",
    )

    args = parser.parse_args()

    # Load tokenizer
    print(f"Loading tokenizer from: {args.tokenizer}")
    tokenizer = WordTokenizer.load(Path(args.tokenizer))

    # Train model
    model = PPMI_SVD(
        tokenizer=tokenizer,
        window_size=args.window_size,
        embedding_dim=args.embedding_dim,
        weighting=args.weighting,
    )

    model.train([Path(args.train_file)])
    model.save(Path(args.output))

    # Test similarity queries
    print("\n" + "=" * 60)
    print("Testing similarity queries")
    print("=" * 60)

    test_words = ["the", "good", "king", "city"]
    for word in test_words:
        if word in model.word2idx:
            print(f"\nMost similar to '{word}':")
            similar = model.most_similar(word, top_k=5)
            for w, score in similar:
                print(f"  {w}: {score:.4f}")
