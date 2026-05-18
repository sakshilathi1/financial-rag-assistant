# Financial RAG Assistant

A production-quality **Retrieval-Augmented Generation (RAG)** pipeline for SEC 10-K filings that compares **dense**, **sparse**, and **hybrid** retrieval strategies with rigorous evaluation metrics.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Financial RAG Assistant                     │
├──────────────┬──────────────────────────┬───────────────────────┤
│  Data Layer  │     Retrieval Layer      │   Generation Layer    │
│              │                          │                       │
│  SEC EDGAR   │  ┌────────┐ ┌────────┐  │  ┌─────────────────┐  │
│  Loader      │  │ Dense  │ │Sparse  │  │  │ AnswerGenerator │  │
│              │  │(FAISS) │ │(BM25)  │  │  │ (OpenAI / tmpl) │  │
│  Text        │  └───┬────┘ └───┬────┘  │  └────────┬────────┘  │
│  Preprocessor│      └────┬─────┘       │           │           │
│              │       ┌───▼────┐        │           │           │
│              │       │Hybrid  │        │           │           │
│              │       │ (RRF)  │        │           │           │
│              │       └───┬────┘        │           │           │
├──────────────┴───────────┼─────────────┴───────────┼───────────┤
│               RAGPipeline│                          │           │
│               query() ───┘──── retrieved_chunks ───┘           │
├─────────────────────────────────────────────────────────────────┤
│                    Evaluation Layer                             │
│   Recall@K · Precision@K · MRR · NDCG@K · ROUGE-L              │
└─────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
financial-rag-assistant/
├── README.md
├── requirements.txt
├── .gitignore
├── .env.example
├── config.yaml
├── src/
│   ├── data/
│   │   ├── sec_loader.py       # Download SEC 10-K filings via EDGAR API
│   │   └── preprocessor.py    # Text cleaning, word-based chunking
│   ├── retrieval/
│   │   ├── dense.py            # Dense: sentence-transformers + FAISS
│   │   ├── sparse.py           # Sparse: BM25Okapi
│   │   └── hybrid.py           # Hybrid: Reciprocal Rank Fusion
│   ├── generation/
│   │   └── generator.py        # LLM generation (OpenAI + template fallback)
│   ├── pipeline/
│   │   └── rag_pipeline.py     # End-to-end RAG pipeline
│   └── evaluation/
│       └── evaluator.py        # Retrieval + generation metrics
├── scripts/
│   ├── download_data.py        # CLI: download SEC filings
│   ├── build_index.py          # CLI: build dense + sparse indices
│   └── run_evaluation.py       # CLI: dense vs hybrid evaluation
└── tests/
    ├── test_preprocessor.py
    ├── test_retrieval.py
    └── test_evaluation.py
```

---

## Installation

```bash
# Clone and enter the repo
git clone https://github.com/sakshilathi1/financial-rag-assistant.git
cd financial-rag-assistant

# Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy and edit environment variables
cp .env.example .env
# Set OPENAI_API_KEY if you want LLM-generated answers (optional)
```

---

## Quick Start (3 steps)

### Step 1 — Download SEC 10-K filings

```bash
python scripts/download_data.py \
    --tickers AAPL,MSFT,GOOGL \
    --output-dir data/raw \
    --limit 2
```

### Step 2 — Build retrieval indices

```bash
python scripts/build_index.py \
    --input-dir data/raw \
    --output-dir data/index
```

### Step 3 — Evaluate dense vs hybrid retrieval

```bash
python scripts/run_evaluation.py \
    --index-dir data/index \
    --output-dir results/ \
    --n-queries 20
```

---

## Retrieval Methods

### Dense Retrieval (`src/retrieval/dense.py`)

Encodes all chunks and the query with **`BAAI/bge-small-en-v1.5`** (a sentence-transformer model), normalises embeddings to unit length, and performs cosine-similarity search via FAISS `IndexFlatIP`.

- **Strengths:** Captures semantic similarity; finds relevant passages even when exact query words are absent.
- **Weaknesses:** Slower to build; may miss exact keyword matches.

### Sparse Retrieval (`src/retrieval/sparse.py`)

Uses **BM25Okapi** (from `rank-bm25`) with configurable `k1` and `b` parameters over simple whitespace-tokenised text.

- **Strengths:** Fast; excellent at exact keyword matching; no GPU needed.
- **Weaknesses:** No semantic understanding; fails on paraphrases.

### Hybrid Retrieval (`src/retrieval/hybrid.py`)

Combines ranked lists from dense and sparse retrieval with **Reciprocal Rank Fusion (RRF)**:

```
RRF_score(d) = Σ  1 / (rrf_k + rank_i(d))
```

RRF is parameter-free (just one constant `rrf_k = 60`) and robust to score-scale differences between retrievers.

- **Strengths:** Gets the best of both worlds; generally outperforms either retriever alone.
- **Weaknesses:** Requires both indices to be built.

---

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **Recall@K** | Fraction of relevant docs found in top-K results |
| **Precision@K** | Fraction of top-K results that are relevant |
| **MRR** | Mean Reciprocal Rank — rewards finding relevant docs early |
| **NDCG@K** | Normalised Discounted Cumulative Gain — position-sensitive |
| **ROUGE-L** | Longest common subsequence F1 for generation quality |

### Example output

```
----------------------------------------------------------------------
  Dense vs Hybrid Retrieval — Evaluation Results
----------------------------------------------------------------------
Metric                  Dense@1  Dense@3  Dense@5 Dense@10 Hybrid@1 Hybrid@3 Hybrid@5 Hybrid@10
----------------------------------------------------------------------
RECALL                  0.4500   0.6200   0.7100   0.8300   0.5000   0.6800   0.7600   0.8700
PRECISION               0.4500   0.2067   0.1420   0.0830   0.5000   0.2267   0.1520   0.0870
NDCG                    0.4500   0.5510   0.5990   0.6540   0.5000   0.5920   0.6340   0.6810
MRR                          0.5210                              0.5870
----------------------------------------------------------------------

— Generation Quality (ROUGE-L F1) —
  Dense  : 0.3140
  Hybrid : 0.3380
```

---

## Configuration (`config.yaml`)

```yaml
data:
  chunk_size: 512        # Words per chunk
  chunk_overlap: 64      # Overlap between consecutive chunks
  min_chunk_length: 100  # Minimum words required to keep a chunk

retrieval:
  dense:
    model_name: "BAAI/bge-small-en-v1.5"
    top_k: 10
  sparse:
    top_k: 10
    b: 0.75    # BM25 length normalisation
    k1: 1.5    # BM25 term-frequency saturation
  hybrid:
    top_k: 10
    rrf_k: 60  # RRF smoothing constant

generation:
  model: "gpt-3.5-turbo"
  max_tokens: 512
  temperature: 0.0
  top_k_context: 5     # Number of chunks sent to LLM

evaluation:
  k_values: [1, 3, 5, 10]
```

---

## Running Tests

```bash
python -m pytest tests/ -v
```

Tests use synthetic corpora and mocked HTTP calls — no internet access required.

---

## Environment Variables

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | OpenAI API key (optional; falls back to template answer) |
| `SEC_USER_AGENT` | User-Agent for SEC EDGAR requests (required by SEC policy) |