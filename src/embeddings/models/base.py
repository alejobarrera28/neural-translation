"""
Abstract base class for word embedding models.

Provides common infrastructure for embedding storage, similarity computation,
and serialization. Subclasses implement specific training algorithms and
vocabulary management strategies.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional
import numpy as np
import pickle


class WordEmbedding(ABC):
    """
    Abstract base class for word embedding models.

    Provides infrastructure for embedding storage, similarity computation,
    and serialization. Subclasses define their own vocabulary management
    and training logic.

    All embedding models use a WordTokenizer but may manage vocabulary
    differently (e.g., re-indexing for dense matrix operations).
    """

    def __init__(
        self,
        tokenizer,  # WordTokenizer instance
        embedding_dim: int,
        window_size: int,
    ):
        """
        Initialize base embedding model.

        Args:
            tokenizer: Pre-trained WordTokenizer (may be used differently by subclasses)
            embedding_dim: Dimensionality of embeddings
            window_size: Context window size for training
        """
        self.tokenizer = tokenizer
        self.embedding_dim = embedding_dim
        self.window_size = window_size

    def _print_training_header(self, model_name: str, **config) -> None:
        """
        Print training header with model configuration.

        Args:
            model_name: Name of the model (e.g., "PPMI-SVD", "Word2Vec")
            **config: Configuration parameters to display
        """
        print("\n" + "=" * 60)
        print(f"Training {model_name} Embeddings")
        print("=" * 60)
        for key, value in config.items():
            print(f"{key}: {value}")

    def _print_training_footer(self) -> None:
        """Print training completion footer."""
        print("\n" + "=" * 60)
        print("Training complete!")
        print("=" * 60)

    def _normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Normalize embeddings for cosine similarity.

        Args:
            embeddings: Embedding matrix (n, dim)

        Returns:
            Normalized embeddings with unit norm per row
        """
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)  # Avoid division by zero
        return embeddings / norms

    def _cosine_similarity(
        self, query: np.ndarray, embeddings: np.ndarray
    ) -> np.ndarray:
        """
        Compute cosine similarity between query and all embeddings.

        Args:
            query: Query vector (dim,)
            embeddings: Embedding matrix (n, dim)

        Returns:
            Similarity scores (n,)
        """
        # Normalize query
        query_norm = np.linalg.norm(query)
        if query_norm < 1e-10:
            return np.zeros(embeddings.shape[0])
        query_normed = query / query_norm

        # Normalize embeddings
        embeddings_normed = self._normalize_embeddings(embeddings)

        # Compute similarities
        return embeddings_normed @ query_normed

    def _find_top_k_similar(
        self,
        query_embedding: np.ndarray,
        embeddings: np.ndarray,
        top_k: int,
        exclude_indices: Optional[set] = None,
    ) -> list:
        """
        Find top-k most similar embeddings using cosine similarity.

        This is a template method that performs the core similarity search.
        Subclasses call this and convert the results to their format.

        Args:
            query_embedding: Query vector (dim,)
            embeddings: Embedding matrix to search (n, dim)
            top_k: Number of results to return
            exclude_indices: Set of indices to exclude from results

        Returns:
            List of (index, similarity) tuples, sorted by similarity descending
        """
        if exclude_indices is None:
            exclude_indices = set()

        # Compute similarities
        similarities = self._cosine_similarity(query_embedding, embeddings)

        # Mask excluded indices
        for idx in exclude_indices:
            similarities[idx] = -np.inf

        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]

        return [(int(idx), float(similarities[idx])) for idx in top_indices]

    def save(self, path: Path) -> None:
        """
        Save model to disk.

        Args:
            path: Path to save model
        """
        path.parent.mkdir(parents=True, exist_ok=True)

        base_state = {
            "tokenizer": self.tokenizer,
            "embedding_dim": self.embedding_dim,
            "window_size": self.window_size,
        }

        # Merge with subclass-specific state
        state = {**base_state, **self._get_save_state()}

        with open(path, "wb") as f:
            pickle.dump(state, f)

        print(f"\nModel saved to: {path}")
