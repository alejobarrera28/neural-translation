# Neural Translation & Embeddings from Scratch

A deep learning project implementing neural machine translation (English-Spanish) and word embeddings from scratch using PyTorch. This project explores progressive architectural improvements from vanilla RNNs to attention-based models, alongside traditional embedding methods.

> **Note**: This project was designed as two complementary studies: word embeddings and neural translation. The translation component is partially complete due to computational constraints with large vocabulary sizes on local hardware.

## Table of Contents
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Implemented Components](#implemented-components)
- [Notebooks](#notebooks)
- [Key Findings](#key-findings)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Technical Details](#technical-details)
- [Limitations](#limitations)
- [License](#license)

## Overview

This project demonstrates the implementation of fundamental NLP architectures from scratch:

**Word Embeddings:**
- PPMI + SVD (count-based embeddings)
- Word2Vec with Skip-gram and Negative Sampling

**Neural Machine Translation:**
- Vanilla RNN Seq2Seq
- LSTM Seq2Seq (addressing vanishing gradients)
- Attention-based Seq2Seq (Bahdanau attention)

**Tokenization:**
- Word-level tokenizers (language-specific)
- Byte Pair Encoding (BPE) tokenizer (shared bilingual vocabulary)

## Project Structure

```
neural-translation/
├── src/
│   ├── embeddings/
│   │   ├── base.py              # Base embedding interface
│   │   ├── PPMI_SVD.py          # Count-based embeddings
│   │   └── Word2Vec.py          # Skip-gram with negative sampling
│   ├── models/
│   │   ├── cells/
│   │   │   ├── rnn_cell.py      # Vanilla RNN from scratch
│   │   │   ├── lstm_cell.py     # LSTM with gates from scratch
│   │   │   └── attention.py     # Bahdanau attention mechanism
│   │   ├── base_seq2seq.py      # Base encoder-decoder framework
│   │   ├── rnn_seq2seq.py       # RNN-based translator
│   │   ├── lstm_seq2seq.py      # LSTM-based translator
│   │   └── attention_seq2seq.py # Attention-based translator
│   ├── data/
│   │   ├── word_tokenizer.py    # Word-level tokenization
│   │   └── bpe_tokenizer.py     # BPE tokenization
│   ├── train.py                 # Training script
│   ├── evaluate.py              # Evaluation utilities
│   └── utils.py                 # Helper functions
├── notebooks/
│   ├── data_exploration.ipynb   # Dataset analysis
│   ├── embeddings_analysis.ipynb # Embedding quality tests
│   ├── vanishing_gradient.ipynb # Gradient flow analysis
│   └── bpe_analysis.ipynb       # Tokenizer deep dive
├── data/
│   ├── raw/                     # Original OPUS-100 data
│   ├── processed/               # Preprocessed parallel corpus
│   └── vocab/                   # Trained tokenizers
└── models/                      # Saved model checkpoints

```

## Implemented Components

### 1. Word Embeddings

#### PPMI + SVD ([src/embeddings/PPMI_SVD.py](src/embeddings/PPMI_SVD.py))
- Co-occurrence matrix construction with sliding window
- Distance-weighted context (harmonic/linear/uniform)
- Positive Pointwise Mutual Information transformation
- Truncated SVD for dimensionality reduction

#### Word2Vec ([src/embeddings/Word2Vec.py](src/embeddings/Word2Vec.py))
- Skip-gram architecture
- Negative sampling (unigram^0.75 distribution)
- Frequent word subsampling
- Efficient training on large corpora

### 2. Neural Translation Models

#### Vanilla RNN Seq2Seq ([src/models/rnn_seq2seq.py](src/models/rnn_seq2seq.py))
- Multi-layer encoder-decoder architecture
- Custom RNN cells from scratch
- Baseline for comparison

#### LSTM Seq2Seq ([src/models/lstm_seq2seq.py](src/models/lstm_seq2seq.py))
- Gated recurrent units (forget, input, output gates)
- Cell state for long-term memory
- Addresses vanishing gradient problem

#### Attention Seq2Seq ([src/models/attention_seq2seq.py](src/models/attention_seq2seq.py))
- Bahdanau (additive) attention mechanism
- Bidirectional LSTM encoder
- Dynamic context vectors at each decoding step
- Solves information bottleneck of fixed-length context

### 3. Tokenization

#### Word Tokenizer ([src/data/word_tokenizer.py](src/data/word_tokenizer.py))
- Frequency-based vocabulary building
- Language-specific vocabularies (separate for EN/ES)
- Treats each word as atomic unit
- Simple whitespace splitting with punctuation handling
- Configurable min frequency and max vocab size

**Characteristics:**
- **Vocab Size**: ~50K words per language (configurable)
- **Coverage**: Higher OOV rate due to morphological variants
- **Use case**: Baseline for comparison, simpler training

#### BPE Tokenizer ([src/data/bpe_tokenizer.py](src/data/bpe_tokenizer.py))
- Byte Pair Encoding with learned merge operations
- Shared bilingual vocabulary (EN-ES)
- Subword segmentation for better coverage
- Morphologically-aware tokenization

**Characteristics:**
- **Vocab Size**: 32,000 tokens (shared EN-ES)
- **Compression**: ~4.8 chars/token for both languages
- **Coverage**: <0.02% UNK rate on dev set
- **Balance**: ES/EN token ratio of 1.074 (nearly balanced)
- **Morphology**: Captures common prefixes/suffixes effectively

See [notebooks/bpe_analysis.ipynb](notebooks/bpe_analysis.ipynb) for comprehensive BPE tokenizer analysis.

## Notebooks

### [data_exploration.ipynb](notebooks/data_exploration.ipynb)
- OPUS-100 corpus statistics
- Language distribution analysis
- Sentence length distributions

### [embeddings_analysis.ipynb](notebooks/embeddings_analysis.ipynb)
- Embedding quality evaluation
- Analogy tasks
- Nearest neighbor analysis
- Comparison of PPMI-SVD vs Word2Vec

### [vanishing_gradient.ipynb](notebooks/vanishing_gradient.ipynb)
- Gradient flow analysis in RNNs vs LSTMs
- Visualization of gradient magnitudes across layers
- Empirical demonstration of vanishing gradient problem

### [bpe_analysis.ipynb](notebooks/bpe_analysis.ipynb)
- Vocabulary distribution by token length
- Merge operation analysis
- Morphological coverage testing
- Bilingual balance metrics
- Compression ratios

## Key Findings

### Tokenization
- BPE achieves excellent bilingual balance (ES/EN ratio: 1.074)
- Near-zero OOV rate (<0.02%) on held-out data
- Efficient compression (~5 characters per token)
- Captures morphological patterns (prefixes, suffixes, roots)

### Gradient Flow
- Vanilla RNNs suffer from exponential gradient decay
- LSTMs maintain stable gradients through gating mechanisms
- Attention mechanisms reduce dependency on long-range gradient propagation

### Vocabulary Size Challenge
The primary limitation encountered was **computational infeasibility of training with large vocabularies on local hardware**. A 32K vocab size creates:
- Large embedding matrices (32K × embedding_dim)
- Memory-intensive softmax computations
- Slow training iterations

This made full translation model training impractical without distributed computing or GPU acceleration.

## Dataset

**OPUS-100 English-Spanish Parallel Corpus**
- Source: [OPUS-100](https://opus.nlpl.eu/opus-100.php)
- Download: [opus-100-corpus-en-es-v1.0.tar.gz](https://object.pouta.csc.fi/OPUS-100/v1.0/opus-100-corpus-en-es-v1.0.tar.gz)
- Language pair: English ↔ Spanish
- Parallel sentences for translation tasks

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/neural-translation.git
cd neural-translation

# Install dependencies
pip install torch numpy scipy scikit-learn tqdm jupyter matplotlib seaborn pandas
```

**Requirements:**
- Python 3.8+
- PyTorch
- NumPy
- SciPy (for PPMI-SVD)
- scikit-learn
- tqdm
- Jupyter (for notebooks)
- matplotlib, seaborn (for visualizations)

## Usage

### Training Word Embeddings

```python
from src.embeddings.Word2Vec import Word2Vec
from src.data.word_tokenizer import WordTokenizer

# Load tokenizer
tokenizer = WordTokenizer.load('data/vocab/word_vocab_en.pkl')

# Initialize Word2Vec
w2v = Word2Vec(
    vocab_size=len(tokenizer),
    embedding_dim=300,
    window_size=5,
    num_negative_samples=5
)

# Train on corpus
corpus_paths = ['data/processed/opus-100/train.en']
w2v.train(corpus_paths, epochs=5)

# Save embeddings
w2v.save('models/word2vec_en.pkl')
```

### Training Translation Models

```python
from src.models.attention_seq2seq import AttentionSeq2Seq
from src.data.bpe_tokenizer import BPETokenizer
import torch

# Load tokenizer
tokenizer = BPETokenizer.load('data/vocab/bpe_vocab_shared.pkl')

# Initialize model
model = AttentionSeq2Seq(
    vocab_size=len(tokenizer),
    embedding_dim=256,
    hidden_dim=512,
    num_layers=2,
    dropout=0.1
)

# Training loop (simplified)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_idx)

# See src/train.py for complete training pipeline
```

### Using BPE Tokenizer

```python
from src.data.bpe_tokenizer import BPETokenizer

# Load pre-trained tokenizer
tokenizer = BPETokenizer.load('data/vocab/bpe_vocab_shared.pkl')

# Encode text
text = "Hello, how are you?"
indices = tokenizer.encode(text)
print(f"Tokens: {[tokenizer.idx2token[i] for i in indices]}")

# Decode
decoded = tokenizer.decode(indices)
print(f"Decoded: {decoded}")
```

## Technical Details

### RNN Cell ([src/models/cells/rnn_cell.py](src/models/cells/rnn_cell.py))
```
h_t = tanh(W_ih * x_t + b_ih + W_hh * h_{t-1} + b_hh)
```

### LSTM Cell ([src/models/cells/lstm_cell.py](src/models/cells/lstm_cell.py))
```
f_t = σ(W_if * x_t + b_if + W_hf * h_{t-1} + b_hf)  # Forget gate
i_t = σ(W_ii * x_t + b_ii + W_hi * h_{t-1} + b_hi)  # Input gate
g_t = tanh(W_ig * x_t + b_ig + W_hg * h_{t-1} + b_hg)  # Cell candidate
o_t = σ(W_io * x_t + b_io + W_ho * h_{t-1} + b_ho)  # Output gate

c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t
h_t = o_t ⊙ tanh(c_t)
```

### Bahdanau Attention ([src/models/cells/attention.py](src/models/cells/attention.py))
```
e_ij = v^T * tanh(W_1 * h_i + W_2 * s_j)  # Energy scores
α_ij = softmax(e_ij)                       # Attention weights
c_j = Σ α_ij * h_i                        # Context vector
```

### Word2Vec Skip-Gram ([src/embeddings/Word2Vec.py](src/embeddings/Word2Vec.py))
```
score = dot(W[target], C[context])
P(label=1) = σ(score)
Loss = -[log P(pos) + Σ log P(neg)]
```

## Limitations

1. **Computational Constraints**: Training seq2seq models with 32K vocabulary requires significant GPU resources
2. **Incomplete Training**: Translation models not fully trained due to hardware limitations
3. **Single Language Pair**: Only English-Spanish implemented
4. **No Beam Search**: Greedy decoding only (beam search not implemented)
5. **Limited Evaluation**: BLEU scores not computed due to incomplete training

## Future Work

- [ ] Train models on GPU cluster or cloud infrastructure
- [ ] Implement beam search decoding
- [ ] Add transformer architecture (self-attention)
- [ ] Subword regularization (BPE-dropout)
- [ ] Multi-language support
- [ ] Proper BLEU/METEOR evaluation
- [ ] Model compression techniques

## References

- Sutskever et al. (2014) - *Sequence to Sequence Learning with Neural Networks*
- Bahdanau et al. (2015) - *Neural Machine Translation by Jointly Learning to Align and Translate*
- Hochreiter & Schmidhuber (1997) - *Long Short-Term Memory*
- Mikolov et al. (2013) - *Efficient Estimation of Word Representations in Vector Space*
- Sennrich et al. (2016) - *Neural Machine Translation of Rare Words with Subword Units*

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Author

Alejandro Barrera

---

*This project was created as an educational exploration of neural machine translation and word embedding techniques, implementing core architectures from scratch to understand their inner workings.*
