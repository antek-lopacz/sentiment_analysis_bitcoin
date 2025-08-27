# Quantifying Investor Sentiment: A Machine Learning Analysis of Bitcoin-Related Reddit Posts
**Bachelor thesis repository (Python 3.11)**

## Summary
Cryptocurrencies move on sentiment as much as fundamentals. This project quantifies market sentiment from **Bitcoin Reddit posts** and compares two embedding methods (**Word2Vec** and **BERT**) paired with two classifiers (**SVM** and **MLP**).

**Key findings**
- Fine-tuned **BERT-MLP** performs best.
- Simpler **W2V** models are competitive and far cheaper to run.
- Tokenization/embeddings matter a lot; **BERT embeddings without fine-tuning worked poorly with SVMs**.
- Highlights trade-offs between **accuracy** and **compute**.

---

## Quick setup

### 1) Create & activate a virtual environment
```bash
# Create a virtual environment
python -m venv .venv

# Windows activation
.venv\Scripts\activate

# macOS/Linux activation
source .venv/bin/activate
```

### 2) Install runtime dependencies

Install Python dependencies:
```bash
pip install -r requirements.txt
```

Install PyTorch (choose one):

#### CPU-only
```bash
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision torchaudio
```

#### NVIDIA CUDA 12.4
```bash
pip install --index-url https://download.pytorch.org/whl/cu124 torch torchvision torchaudio
```

If unsure, see: https://pytorch.org/get-started/locally/