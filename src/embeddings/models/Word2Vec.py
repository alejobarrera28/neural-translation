"""
Word2Vec implementation using Skip-Gram with Negative Sampling (SGNS).

This is the canonical Word2Vec algorithm as used in practice:
- Skip-gram architecture predicting context from target words
- Negative sampling instead of full softmax (practical necessity)
- Unigram^0.75 noise distribution (definition-level requirement)
- Subsampling of frequent words (prevents common words from dominating)

Architecture:
    Two embedding matrices: word embeddings W and context embeddings C
    Scoring: dot(W[target], C[context])
    Loss: Binary cross-entropy over positive and negative samples

References:
    Mikolov et al. (2013): "Efficient Estimation of Word Representations in Vector Space"
    Mikolov et al. (2013): "Distributed Representations of Words and Phrases and their Compositionality"
"""

import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import pickle
from collections import Counter
from tqdm import tqdm
import math


class Word2Vec:
    """
    Skip-Gram with Negative Sampling (SGNS) implementation.

    This is the practical Word2Vec algorithm that learns word embeddings
    by predicting context words from target words using negative sampling.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 300,
        window_size: int = 5,
        num_negative_samples: int = 5,
        subsample_threshold: float = 1e-5,
        learning_rate: float = 0.025,
        min_learning_rate: float = 0.0001,
        batch_size: int = 512,
    ):
        """
        Initialize Word2Vec model.

        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimensionality of word embeddings
            window_size: Maximum distance between target and context words
            num_negative_samples: Number of negative samples per positive sample
            subsample_threshold: Threshold for subsampling frequent words (t in paper)
            learning_rate: Initial learning rate
            min_learning_rate: Minimum learning rate for decay
            batch_size: Number of training pairs to process in parallel
        """
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.window_size = window_size
        self.num_negative_samples = num_negative_samples
        self.subsample_threshold = subsample_threshold
        self.learning_rate = learning_rate
        self.initial_learning_rate = learning_rate
        self.min_learning_rate = min_learning_rate
        self.batch_size = batch_size

        # Initialize embedding matrices with small random values
        # W: word embeddings (what we ultimately want)
        # C: context embeddings (auxiliary matrix for training)
        self.W = (np.random.rand(vocab_size, embedding_dim) - 0.5) / embedding_dim
        self.C = np.zeros((vocab_size, embedding_dim))

        # Placeholders for training-specific data
        self.word_counts = None
        self.subsample_probs = None
        self.negative_sampling_table = None

    def _build_negative_sampling_table(
        self, word_counts: Counter, table_size: int = 1_000_000
    ):
        """
        Build sampling table using unigram^0.75 distribution.

        The 0.75 exponent smooths the distribution, giving rare words more representation.
        """
        SMOOTHING_EXPONENT = 0.75
        norm = sum(count**SMOOTHING_EXPONENT for count in word_counts.values())

        table = []
        for word_idx in range(self.vocab_size):
            count = word_counts.get(word_idx, 0)
            if count > 0:
                proportion = (count**SMOOTHING_EXPONENT) / norm
                num_entries = int(proportion * table_size)
                table.extend([word_idx] * num_entries)

        self.negative_sampling_table = np.array(table, dtype=np.int32)

    def _build_subsampling_probs(self, word_counts: Counter, total_words: int):
        """
        Compute subsampling probabilities to downsample frequent words.

        P(keep) = sqrt(t / f(w)) for words with frequency f(w) > threshold t
        """
        self.subsample_probs = np.ones(self.vocab_size)

        for word_idx, count in word_counts.items():
            freq = count / total_words
            if freq > self.subsample_threshold:
                self.subsample_probs[word_idx] = np.sqrt(
                    self.subsample_threshold / freq
                )

    def _generate_training_pairs(
        self,
        corpus: List[List[int]],
        special_token_ids: set,
    ) -> List[Tuple[int, int]]:
        """
        Generate (target, context) training pairs from corpus.

        For each target word, generate pairs with words in its context window.

        Args:
            corpus: List of sentences, where each sentence is a list of word indices
            special_token_ids: Set of special token IDs to skip (e.g., {0, 1, 2, 3})

        Returns:
            List of (target_idx, context_idx) pairs
        """

        pairs = []

        for sentence in corpus:
            # Filter out special tokens
            sentence = [word for word in sentence if word not in special_token_ids]

            # Apply subsampling
            sentence = [
                word
                for word in sentence
                if np.random.rand() < self.subsample_probs[word]
            ]

            # Skip empty sentences
            if len(sentence) == 0:
                continue

            # Generate pairs within context window
            for i, target in enumerate(sentence):
                # Dynamic window size (sample from 1 to window_size)
                actual_window = np.random.randint(1, self.window_size + 1)

                # Context words within window
                start = max(0, i - actual_window)
                end = min(len(sentence), i + actual_window + 1)

                for j in range(start, end):
                    if i != j:  # Don't pair word with itself
                        context = sentence[j]
                        pairs.append((target, context))

        return pairs

    def _sample_negative(
        self, num_samples: int, exclude: Optional[set] = None
    ) -> np.ndarray:
        """
        Sample negative examples from the noise distribution.

        Args:
            num_samples: Number of negative samples to draw
            exclude: Set of word indices to exclude from sampling

        Returns:
            Array of negative sample indices
        """
        if exclude is None:
            exclude = set()

        samples = []
        while len(samples) < num_samples:
            # Sample from precomputed table
            idx = np.random.randint(0, len(self.negative_sampling_table))
            sample = self.negative_sampling_table[idx]

            # Skip if in exclusion set
            if sample not in exclude:
                samples.append(sample)
                exclude.add(sample)  # Avoid duplicates

        return np.array(samples, dtype=np.int32)

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Numerically stable sigmoid function."""
        # Use np.clip to prevent overflow in exp
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def _train_batch(
        self, target_indices: np.ndarray, context_indices: np.ndarray
    ) -> float:
        """
        Train on a batch using binary cross-entropy with negative sampling.

        Word2Vec frames word prediction as binary classification:
        - Positive pairs (label=1): Real (target, context) pairs from corpus
        - Negative pairs (label=0): Random (target, noise_word) pairs

        For each pair we compute:
            score = dot(W[target], C[context])
            prediction = sigmoid(score)
            loss = -log(prediction) for positives, -log(1-prediction) for negatives
            gradient = (prediction - label) for both cases

        Loss: -log(σ(target·context)) - ∑ log(σ(-target·negative_i))
        """
        batch_size = len(target_indices)
        target_embs = self.W[target_indices]  # (batch_size, dim)
        context_embs = self.C[context_indices]  # (batch_size, dim)

        # ============================================================================
        # POSITIVE SAMPLES: Real context pairs (label = 1)
        # ============================================================================
        # Goal: Maximize dot(W[target], C[context]) → push embeddings together

        # Forward pass
        pos_scores = np.sum(target_embs * context_embs, axis=1)  # dot products
        pos_predictions = self._sigmoid(pos_scores)  # σ(score) ∈ [0,1]
        pos_loss = -np.sum(np.log(pos_predictions + 1e-10))  # cross-entropy

        # Backward pass: Compute gradients
        # For binary cross-entropy with label=1: ∂L/∂score = (prediction - 1)
        pos_error = (pos_predictions - 1) * self.learning_rate  # (batch_size,)

        # Chain rule: ∂L/∂W[target] = error * ∂score/∂W = error * C[context]
        pos_grad_target = pos_error[:, np.newaxis] * context_embs  # (batch_size, dim)

        # Chain rule: ∂L/∂C[context] = error * ∂score/∂C = error * W[target]
        pos_grad_context = pos_error[:, np.newaxis] * target_embs  # (batch_size, dim)

        # ============================================================================
        # NEGATIVE SAMPLES: Random noise pairs (label = 0)
        # ============================================================================
        # Goal: Minimize dot(W[target], C[negative]) → push embeddings apart

        # Sample random negative words (excluding target and positive context)
        neg_indices = np.array(
            [
                self._sample_negative(
                    self.num_negative_samples,
                    exclude={int(target_indices[i]), int(context_indices[i])},
                )
                for i in range(batch_size)
            ]
        )  # (batch_size, num_negative_samples)

        # Forward pass
        neg_embs = self.C[neg_indices]  # (batch_size, num_neg, dim)
        neg_scores = np.sum(target_embs[:, np.newaxis, :] * neg_embs, axis=2)  # (batch_size, num_neg)

        # Use σ(-score) instead of (1 - σ(score)) for numerical stability
        neg_predictions = self._sigmoid(-neg_scores)  # Want this → 1 (score → 0)
        neg_loss = -np.sum(np.log(neg_predictions + 1e-10))  # cross-entropy

        # Backward pass: Compute gradients
        # For -log(σ(-score)): ∂L/∂score = -(1 - σ(-score)) = -(1 - prediction)
        # Equivalently with label=0: gradient = (prediction - 0) but we negate score,
        # so final form is: (1 - σ(-score))
        neg_error = (1 - neg_predictions) * self.learning_rate  # (batch_size, num_neg)

        # Accumulate gradients for target embeddings from all negative samples
        neg_grad_target = np.sum(neg_error[:, :, np.newaxis] * neg_embs, axis=1)  # (batch_size, dim)

        # ============================================================================
        # UPDATE EMBEDDINGS
        # ============================================================================
        # Gradient descent: embedding -= learning_rate * gradient

        # Update W[target]: Accumulate gradients from positive + negative samples
        total_grad_target = pos_grad_target + neg_grad_target
        np.add.at(self.W, target_indices, -total_grad_target)

        # Update C[context]: Only from positive samples
        np.add.at(self.C, context_indices, -pos_grad_context)

        # Update C[negatives]: Each negative word gets its own gradient
        for i in range(batch_size):
            for j in range(self.num_negative_samples):
                neg_word_idx = neg_indices[i, j]
                neg_grad_context = neg_error[i, j] * target_embs[i]
                self.C[neg_word_idx] -= neg_grad_context

        return pos_loss + neg_loss

    def train(
        self,
        corpus: List[List[int]],
        word_counts: Counter,
        epochs: int = 5,
        special_token_ids: Optional[set] = None,
    ):
        """
        Train Word2Vec model on corpus.

        Args:
            corpus: List of sentences (each sentence is list of word indices)
            word_counts: Counter mapping word indices to their frequencies
            epochs: Number of training epochs
            special_token_ids: Set of special token IDs to exclude (e.g., {0, 1, 2, 3} for PAD/UNK/BOS/EOS)
        """
        self.word_counts = word_counts
        total_words = sum(word_counts.values())

        # Build sampling structures
        self._build_negative_sampling_table(word_counts)
        self._build_subsampling_probs(word_counts, total_words)

        # Generate training pairs
        training_pairs = self._generate_training_pairs(
            corpus, special_token_ids=special_token_ids
        )
        print(f"Training on {len(training_pairs):,} pairs for {epochs} epochs")

        # Training loop
        for epoch in range(epochs):
            np.random.shuffle(training_pairs)
            training_pairs_array = np.array(training_pairs, dtype=np.int32)
            num_pairs = len(training_pairs_array)
            num_batches = math.ceil(num_pairs / self.batch_size)

            epoch_loss = 0
            for batch_idx in tqdm(range(num_batches), desc=f"Epoch {epoch+1}/{epochs}"):
                # Linear learning rate decay
                progress = (epoch * num_pairs + batch_idx * self.batch_size) / (
                    epochs * num_pairs
                )
                self.learning_rate = max(
                    self.min_learning_rate, self.initial_learning_rate * (1 - progress)
                )

                # Get batch and train
                start_idx = batch_idx * self.batch_size
                end_idx = min(start_idx + self.batch_size, num_pairs)
                batch_pairs = training_pairs_array[start_idx:end_idx]
                batch_loss = self._train_batch(batch_pairs[:, 0], batch_pairs[:, 1])
                epoch_loss += batch_loss

            avg_loss = epoch_loss / num_pairs
            print(
                f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}, LR: {self.learning_rate:.6f}"
            )

    def get_embedding(self, word_idx: int) -> np.ndarray:
        """
        Get embedding vector for a word.

        Args:
            word_idx: Index of word

        Returns:
            Embedding vector (dim,)
        """
        return self.W[word_idx]

    def get_embeddings(self) -> np.ndarray:
        """
        Get all word embeddings.

        Returns:
            Embedding matrix (vocab_size, dim)
        """
        return self.W

    def most_similar(
        self, word_idx: int, top_k: int = 10, exclude: Optional[set] = None
    ) -> List[Tuple[int, float]]:
        """
        Find most similar words using cosine similarity.

        Args:
            word_idx: Index of query word
            top_k: Number of similar words to return
            exclude: Set of word indices to exclude from results

        Returns:
            List of (word_idx, similarity) tuples, sorted by similarity
        """
        if exclude is None:
            exclude = set()
        exclude.add(word_idx)  # Exclude the word itself

        # Get query embedding and normalize
        query = self.W[word_idx]
        query_norm = query / (np.linalg.norm(query) + 1e-10)

        # Compute cosine similarities
        W_norm = self.W / (np.linalg.norm(self.W, axis=1, keepdims=True) + 1e-10)
        similarities = np.dot(W_norm, query_norm)

        # Mask excluded words
        for idx in exclude:
            similarities[idx] = -np.inf

        # Get top-k
        top_indices = np.argsort(similarities)[::-1][:top_k]
        results = [(idx, similarities[idx]) for idx in top_indices]

        return results

    def save(self, path: Path):
        """
        Save model to disk.

        Args:
            path: Path where model will be saved
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "W": self.W,
            "C": self.C,
            "vocab_size": self.vocab_size,
            "embedding_dim": self.embedding_dim,
            "window_size": self.window_size,
            "num_negative_samples": self.num_negative_samples,
            "subsample_threshold": self.subsample_threshold,
            "learning_rate": self.learning_rate,
            "initial_learning_rate": self.initial_learning_rate,
            "min_learning_rate": self.min_learning_rate,
            "batch_size": self.batch_size,
            "word_counts": self.word_counts,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls, path: Path) -> "Word2Vec":
        """
        Load model from disk.

        Args:
            path: Path to saved model

        Returns:
            Loaded Word2Vec instance
        """
        with open(path, "rb") as f:
            state = pickle.load(f)

        model = cls(
            vocab_size=state["vocab_size"],
            embedding_dim=state["embedding_dim"],
            window_size=state["window_size"],
            num_negative_samples=state["num_negative_samples"],
            subsample_threshold=state["subsample_threshold"],
            learning_rate=state["learning_rate"],
            min_learning_rate=state["min_learning_rate"],
            batch_size=state.get(
                "batch_size", 512
            ),  # Default for backward compatibility
        )
        model.W = state["W"]
        model.C = state["C"]
        model.initial_learning_rate = state["initial_learning_rate"]
        model.word_counts = state["word_counts"]

        return model


def train_word2vec_from_tokenizer(
    tokenizer,  # WordTokenizer instance
    corpus_paths: List[Path],
    embedding_dim: int = 300,
    window_size: int = 5,
    num_negative_samples: int = 5,
    subsample_threshold: float = 1e-5,
    epochs: int = 5,
    learning_rate: float = 0.025,
    min_learning_rate: float = 0.0001,
    batch_size: int = 512,
    save_path: Optional[Path] = None,
) -> Word2Vec:
    """
    Train Word2Vec model from a WordTokenizer and corpus files.

    Args:
        tokenizer: Trained WordTokenizer instance
        corpus_paths: List of text file paths to train on
        embedding_dim: Dimensionality of embeddings
        window_size: Maximum context window size
        num_negative_samples: Number of negative samples per positive
        subsample_threshold: Threshold for frequent word subsampling
        epochs: Number of training epochs
        learning_rate: Initial learning rate
        min_learning_rate: Minimum learning rate
        batch_size: Number of training pairs to process in parallel
        save_path: Optional path to save trained model

    Returns:
        Trained Word2Vec model
    """
    # Load and encode corpus
    print("Loading corpus...")
    corpus = []
    word_counts = Counter()

    for corpus_path in corpus_paths:
        with open(corpus_path, "r", encoding="utf-8") as f:
            for line in f:
                encoded = tokenizer.encode(line.strip(), add_bos=False, add_eos=False)
                if len(encoded) > 0:
                    corpus.append(encoded)
                    word_counts.update(encoded)

    print(f"Loaded {len(corpus):,} sentences, {sum(word_counts.values()):,} tokens")

    # Initialize and train model
    special_token_ids = {
        tokenizer.word2idx[token] for token in tokenizer.special_tokens
    }

    model = Word2Vec(
        vocab_size=len(tokenizer.word2idx),
        embedding_dim=embedding_dim,
        window_size=window_size,
        num_negative_samples=num_negative_samples,
        subsample_threshold=subsample_threshold,
        learning_rate=learning_rate,
        min_learning_rate=min_learning_rate,
        batch_size=batch_size,
    )

    model.train(
        corpus=corpus,
        word_counts=word_counts,
        epochs=epochs,
        special_token_ids=special_token_ids,
    )

    # Save if path provided
    if save_path is not None:
        print(f"Saving model to {save_path}")
        model.save(save_path)

    return model


if __name__ == "__main__":
    import argparse
    import sys

    sys.path.append(str(Path(__file__).parent.parent.parent.parent))
    from src.data.word_tokenizer import WordTokenizer

    parser = argparse.ArgumentParser(
        description="Train Word2Vec embeddings using Skip-Gram with Negative Sampling"
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="data/vocab/word_vocab_en.pkl",
        help="Path to trained WordTokenizer (.pkl file)",
    )
    parser.add_argument(
        "--corpus",
        type=str,
        nargs="+",
        default="data/processed/opus-100/train.en",
        help="Path(s) to corpus text file(s)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/word2vec.pkl",
        help="Path to save trained Word2Vec model",
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=300,
        help="Dimensionality of word embeddings (default: 300)",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=5,
        help="Maximum context window size (default: 5)",
    )
    parser.add_argument(
        "--negative-samples",
        type=int,
        default=5,
        help="Number of negative samples (default: 5)",
    )
    parser.add_argument(
        "--subsample-threshold",
        type=float,
        default=1e-5,
        help="Threshold for subsampling frequent words (default: 1e-5)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs (default: 5)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.025,
        help="Initial learning rate (default: 0.025)",
    )
    parser.add_argument(
        "--min-learning-rate",
        type=float,
        default=0.0001,
        help="Minimum learning rate (default: 0.0001)",
    )

    args = parser.parse_args()

    # Load tokenizer
    print(f"Loading tokenizer from {args.tokenizer}...")
    tokenizer = WordTokenizer.load(Path(args.tokenizer))
    print(f"Vocabulary size: {len(tokenizer.word2idx):,}")

    # Convert corpus paths
    corpus_paths = [Path(args.corpus)]

    # Train model
    model = train_word2vec_from_tokenizer(
        tokenizer=tokenizer,
        corpus_paths=corpus_paths,
        embedding_dim=args.embedding_dim,
        window_size=args.window_size,
        num_negative_samples=args.negative_samples,
        subsample_threshold=args.subsample_threshold,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        min_learning_rate=args.min_learning_rate,
        save_path=Path(args.output),
    )

    print("Training complete!")
