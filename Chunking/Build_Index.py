import json
import re
import pickle
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

############################################################
# CONFIG
############################################################

INPUT_JSON = Path("output/code_travail.json")

CHROMA_DIR = Path("chroma_db")
CHROMA_COLLECTION_NAME = "code_travail"

BM25_INDEX_PATH = Path("output/bm25_index.pkl")

EMBEDDING_MODEL_NAME = "BAAI/bge-m3"

# Only ~1 article out of 507 exceeds this (Article 4, 795
# words), so 500 keeps virtually every article as a single
# chunk while still catching genuine outliers.
MAX_CHUNK_WORDS = 500


############################################################
# STEP 1
# Load parsed articles
############################################################

def load_articles(json_path):

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data["articles"]


############################################################
# STEP 2
# Article-aware chunking
############################################################

def split_sentences(text):
    """
    Split text into sentences without cutting mid-sentence.
    Splits on '.', ';' or ':' followed by whitespace, while
    avoiding common French abbreviations (art., n°, etc.) so
    legal references aren't broken apart.
    """

    # Protect common abbreviations from being treated as
    # sentence boundaries.
    protected = text
    protected = re.sub(r"\bart\.\s", "art_DOT_ ", protected, flags=re.I)
    protected = re.sub(r"\bn°\s", "n_DEGREE_ ", protected)

    raw_sentences = re.split(r"(?<=[\.;:])\s+", protected)

    sentences = []

    for s in raw_sentences:
        s = s.replace("art_DOT_", "art.").replace("n_DEGREE_", "n°")
        s = s.strip()
        if s:
            sentences.append(s)

    return sentences


def split_long_text(text, max_words):
    """
    Split long text into meaning-preserving chunks: first try
    paragraph breaks (natural alinéa boundaries preserved by
    the parser), then fall back to sentence boundaries.
    Sentences/paragraphs are packed greedily up to max_words —
    never cut inside a sentence.
    """

    words = text.split()

    if len(words) <= max_words:
        return [text]

    # Prefer splitting on existing paragraph breaks first, since
    # these usually correspond to alinéas (numbered sub-points)
    # in the article and are the most natural chunk boundaries.
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]

    if len(paragraphs) <= 1:
        paragraphs = [text]

    units = []

    for para in paragraphs:
        if len(para.split()) > max_words:
            # Paragraph itself is too long — break it down further
            # by sentence so we still never cut mid-sentence.
            units.extend(split_sentences(para))
        else:
            units.append(para)

    # Greedily pack units (paragraphs/sentences) into chunks,
    # respecting max_words and never splitting a unit itself.
    chunks = []
    current_chunk_units = []
    current_word_count = 0

    for unit in units:

        unit_word_count = len(unit.split())

        if current_word_count + unit_word_count > max_words and current_chunk_units:
            chunks.append(" ".join(current_chunk_units))
            current_chunk_units = []
            current_word_count = 0

        current_chunk_units.append(unit)
        current_word_count += unit_word_count

    if current_chunk_units:
        chunks.append(" ".join(current_chunk_units))

    return chunks


def build_chunks(articles):
    """
    Turn parsed articles into retrieval-ready chunks.
    Each chunk carries full hierarchical metadata (livre,
    titre, chapitre, section) plus modification history, so
    the UI can display provenance and the retriever can filter
    by structure if needed later.
    """

    chunks = []

    for art in articles:

        text_parts = split_long_text(art["text"], MAX_CHUNK_WORDS)

        for part_idx, part_text in enumerate(text_parts):

            chunk_id = f"article_{art['id']}"

            if len(text_parts) > 1:
                chunk_id += f"_part{part_idx + 1}"

            # Prepend the article label to the chunk text itself.
            # This helps both BM25 (exact article number lookup)
            # and embeddings (the model sees which article the
            # text belongs to, improving retrieval when queries
            # reference an article number directly).
            embedded_text = f"{art['article']}. {part_text}"

            chunks.append({
                "id": chunk_id,
                "article_id": art["id"],
                "article": art["article"],
                "livre": art["livre"],
                "titre": art["titre"],
                "chapitre": art["chapitre"],
                "section": art["section"],
                "modifications": art.get("modifications", []),
                "text": embedded_text,
            })

    return chunks


############################################################
# STEP 3
# Embedding
############################################################

def embed_chunks(chunks, model_name):

    model = SentenceTransformer(model_name)

    texts = [c["text"] for c in chunks]

    # normalize_embeddings=True gives cosine-similarity-ready
    # vectors, which matches ChromaDB's default distance setup
    # when the collection is created with cosine space.
    embeddings = model.encode(
        texts,
        batch_size=16,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    return embeddings


############################################################
# STEP 4
# ChromaDB indexing
############################################################

def build_chroma_index(chunks, embeddings, persist_dir, collection_name):

    client = chromadb.PersistentClient(path=str(persist_dir))

    # Drop any existing collection with the same name so reruns
    # don't silently duplicate or stack on top of old vectors.
    existing = [c.name for c in client.list_collections()]

    if collection_name in existing:
        client.delete_collection(collection_name)

    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    ids = [c["id"] for c in chunks]

    documents = [c["text"] for c in chunks]

    # Chroma metadata values must be str/int/float/bool — lists
    # (like modifications) get JSON-serialized to a string, and
    # None values get replaced with empty strings, since Chroma
    # rejects None in metadata.
    metadatas = []

    for c in chunks:

        metadatas.append({
            "article_id": c["article_id"],
            "article": c["article"] or "",
            "livre": c["livre"] or "",
            "titre": c["titre"] or "",
            "chapitre": c["chapitre"] or "",
            "section": c["section"] or "",
            "modifications": json.dumps(c["modifications"], ensure_ascii=False),
        })

    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings.tolist(),
        metadatas=metadatas,
    )

    return collection


############################################################
# STEP 5
# BM25 index (for hybrid retrieval)
############################################################

def tokenize(text):
    # Simple lowercase word tokenizer. Good enough for BM25
    # over French legal text; swap in a proper French tokenizer
    # (e.g. spaCy's fr_core_news_sm) later if recall on
    # inflected forms turns out to matter.
    return re.findall(r"\w+", text.lower())


def build_bm25_index(chunks):

    tokenized_corpus = [tokenize(c["text"]) for c in chunks]

    bm25 = BM25Okapi(tokenized_corpus)

    return bm25


def save_bm25_index(bm25, chunks, path):

    # Store the BM25 object alongside the chunk id/text list, so
    # a BM25 result index can be mapped back to a chunk id at
    # retrieval time without recomputing anything.
    payload = {
        "bm25": bm25,
        "chunk_ids": [c["id"] for c in chunks],
        "chunk_texts": [c["text"] for c in chunks],
    }

    with open(path, "wb") as f:
        pickle.dump(payload, f)


############################################################
# MAIN
############################################################

def main():

    articles = load_articles(INPUT_JSON)

    chunks = build_chunks(articles)

    print(f"{len(articles)} articles -> {len(chunks)} chunks")

    embeddings = embed_chunks(chunks, EMBEDDING_MODEL_NAME)

    build_chroma_index(
        chunks,
        embeddings,
        CHROMA_DIR,
        CHROMA_COLLECTION_NAME,
    )

    print(f"ChromaDB index built at {CHROMA_DIR} ({CHROMA_COLLECTION_NAME})")

    bm25 = build_bm25_index(chunks)

    Path("output").mkdir(exist_ok=True)

    save_bm25_index(bm25, chunks, BM25_INDEX_PATH)

    print(f"BM25 index saved to {BM25_INDEX_PATH}")


if __name__ == "__main__":
    main()
