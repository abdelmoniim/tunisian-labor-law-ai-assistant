# ⚖️ Tunisian labor law ai assistant

An AI assistant that answers questions about the Tunisian Labor Law (Code du Travail, 2016 edition) in French, grounded strictly in the official text — no external knowledge, no hallucinated articles.

Built from scratch with a custom hybrid-retrieval RAG pipeline — deliberately without LangChain, in order to understand what happens under the hood at every stage (chunking, embeddings, hybrid retrieval, reranking, generation) — as a portfolio project demonstrating retrieval-augmented generation fundamentals end to end.

> ⚠️ This is a technical demonstration project and does **not** constitute legal advice.

## How it works

```
PDF (Code du Travail 2016)
        │
        ▼
 Ingestion  → parses raw text into structured articles
        │      (article number, livre/titre/chapitre/section,
        │       modification history)
        ▼
 Chunking   → builds two indexes from the parsed articles:
        │       • BM25 (lexical/keyword index)
        │       • ChromaDB (dense vector index, bge-m3 embeddings)
        ▼
 Retrieve   → hybrid search at query time:
        │       1. BM25 search + vector search (top 20 each)
        │       2. Reciprocal Rank Fusion (RRF) merges both lists
        │       3. Cross-encoder reranks the fused candidates
        │       4. Top 5 most relevant articles are kept
        ▼
 Generate   → the retrieved articles are passed as context to
        │       Groq/Llama, which answers strictly from that
        │       context and cites the article numbers used
        ▼
   Gradio chat UI
```

### Why hybrid retrieval?

- **BM25** catches exact legal terminology and article-number matches that embeddings can miss.
- **Dense vectors (bge-m3)** catch semantically related questions phrased differently from the text.
- **RRF** combines both ranked lists without needing to tune a weighting scheme.
- **Cross-encoder reranking (bge-reranker-base)** does a final, more expensive but more accurate pass over the fused candidates before they reach the LLM.

### Grounding & anti-hallucination

The system prompt instructs the model to answer *only* from the retrieved articles, to always cite the article number(s) it relies on, and to explicitly say it has no relevant provision rather than guess when the retrieved context doesn't cover the question.

## Project structure

```
.
├── Data/
│   └── code_de_travail_2016_6.pdf     # Source document (official Code du Travail, 2016)
├── Ingestion/
│   └── parse_articles.py              # PDF → cleaned, structured JSON (one entry per article)
├── Chunking/
│   └── Build_Index.py                 # Builds the BM25 index and the ChromaDB vector index
└── Retrieve/
    ├── Retrieve.py                    # HybridRetriever: BM25 + vector search, RRF, reranking
    ├── generate.py                    # LaborLawRAG: retrieval + Groq/Llama generation
    └── app.py                         # Gradio chat interface (entry point)
```

## Tech stack

| Component | Choice |
|---|---|
| PDF parsing | PyMuPDF (`fitz`) |
| Lexical search | BM25 (`rank_bm25`) |
| Embeddings | `BAAI/bge-m3` (`sentence-transformers`) |
| Vector store | ChromaDB (HNSW) |
| Fusion | Reciprocal Rank Fusion (RRF) |
| Reranking | `BAAI/bge-reranker-base` (cross-encoder) |
| Generation | Groq API (Llama) |
| UI | Gradio |

## Setup

### 1. Install dependencies

```bash
pip install pymupdf rank_bm25 sentence-transformers chromadb groq gradio python-dotenv
```

### 2. Set your Groq API key

```bash
# Linux/Mac
export GROQ_API_KEY="your_key_here"

# Windows
setx GROQ_API_KEY "your_key_here"
```

Or create a `.env` file in `Retrieve/`:

```
GROQ_API_KEY=your_key_here
```

### 3. Build the indexes (first run only)

```bash
python Ingestion/parse_articles.py   # PDF → output/code_travail.json
python Chunking/Build_Index.py       # builds BM25 index + ChromaDB collection
```

### 4. Launch the assistant

```bash
cd Retrieve
python app.py
```

This starts a local Gradio chat interface. Ask a question in French about the Tunisian labor low (e.g. *"Quelle est la durée de la période d'essai ?"*) and the assistant will answer with the article(s) it used, shown alongside the conversation.

## Example

**Q:** Quelle est la durée de la période d'essai ?

**A:** The assistant retrieves the relevant article(s) via hybrid search + reranking, then answers based only on that text, citing the article number(s) — with the cited articles and their relevance scores displayed in the sidebar.

