import json
import pickle
import re
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder

############################################################
# CONFIG
############################################################

CHROMA_DIR = Path("chroma_db")
CHROMA_COLLECTION_NAME = "code_travail"

BM25_INDEX_PATH = Path("output/bm25_index.pkl")

EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
RERANKER_MODEL_NAME = "BAAI/bge-reranker-base"

# How many candidates each retriever (BM25, vector) contributes
# before fusion. Wider than the final top-k so RRF has enough
# signal to work with, and so a good match that's mediocre in
# one retriever but strong in the other doesn't get dropped
# too early.
CANDIDATES_PER_RETRIEVER = 20

# RRF constant — standard default from the original RRF paper.
# Higher k flattens the influence of rank position; lower k
# rewards top ranks more aggressively. 60 is the well-tested
# default and rarely needs tuning.
RRF_K = 60

# Final number of chunks handed to the reranker, and the final
# number returned after reranking.
FUSED_TOP_N = 15
FINAL_TOP_N = 5


############################################################
# STEP 1
# Load indexes (built by build_index.py)
############################################################

def load_bm25_index(path):

    with open(path, "rb") as f:
        payload = pickle.load(f)

    return payload["bm25"], payload["chunk_ids"], payload["chunk_texts"]


def load_chroma_collection(persist_dir, collection_name):

    client = chromadb.PersistentClient(path=str(persist_dir))

    return client.get_collection(collection_name)


############################################################
# STEP 2
# BM25 retrieval
############################################################

def tokenize(text):
    return re.findall(r"\w+", text.lower())


def bm25_search(bm25, chunk_ids, query, top_n):

    tokenized_query = tokenize(query)

    scores = bm25.get_scores(tokenized_query)

    # Pair each chunk id with its score, sort descending, keep
    # top_n. This mirrors what a vector search .query() call
    # returns, so both retrievers produce a comparable ranked
    # list going into RRF.
    ranked = sorted(
        zip(chunk_ids, scores),
        key=lambda x: x[1],
        reverse=True,
    )

    return [chunk_id for chunk_id, score in ranked[:top_n]]


############################################################
# STEP 3
# Vector (dense) retrieval
############################################################

def vector_search(collection, embed_model, query, top_n):

    # e5-style prefixing isn't needed for bge-m3, but if you
    # switch EMBEDDING_MODEL_NAME to an e5 model later, prefix
    # the query with "query: " here to match how e5 was trained.
    query_embedding = embed_model.encode(
        [query],
        normalize_embeddings=True,
    )[0]

    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_n,
    )

    return results["ids"][0]


############################################################
# STEP 4
# Reciprocal Rank Fusion
############################################################

def reciprocal_rank_fusion(ranked_lists, k=RRF_K):
    """
    ranked_lists: list of ranked chunk-id lists, one per
    retriever (e.g. [bm25_ids, vector_ids]).
    Returns chunk ids sorted by fused RRF score, descending.
    """

    scores = {}

    for ranked_list in ranked_lists:
        for rank, chunk_id in enumerate(ranked_list):
            # rank is 0-indexed here; RRF formula conventionally
            # uses 1-indexed rank, hence rank + 1 below.
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)

    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    return [chunk_id for chunk_id, score in fused]


############################################################
# STEP 5
# Cross-encoder reranking
############################################################

def rerank(reranker, query, chunk_id_to_text, chunk_ids, top_n):

    pairs = [(query, chunk_id_to_text[cid]) for cid in chunk_ids]

    scores = reranker.predict(pairs)

    reranked = sorted(
        zip(chunk_ids, scores),
        key=lambda x: x[1],
        reverse=True,
    )

    return reranked[:top_n]


############################################################
# STEP 6
# Full pipeline
############################################################

class HybridRetriever:

    def __init__(self):

        print("Loading BM25 index...")
        self.bm25, self.bm25_chunk_ids, self.bm25_chunk_texts = load_bm25_index(
            BM25_INDEX_PATH
        )

        print("Loading ChromaDB collection...")
        self.collection = load_chroma_collection(
            CHROMA_DIR, CHROMA_COLLECTION_NAME
        )

        print("Loading embedding model...")
        self.embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

        print("Loading reranker model...")
        self.reranker = CrossEncoder(RERANKER_MODEL_NAME)

        # Map chunk_id -> text and chunk_id -> full metadata, so
        # reranking and final display don't need to re-fetch
        # from Chroma one id at a time.
        self.chunk_id_to_text = dict(
            zip(self.bm25_chunk_ids, self.bm25_chunk_texts)
        )

        all_data = self.collection.get()
        self.chunk_id_to_metadata = dict(
            zip(all_data["ids"], all_data["metadatas"])
        )

    def retrieve(self, query, final_top_n=FINAL_TOP_N):

        bm25_ids = bm25_search(
            self.bm25, self.bm25_chunk_ids, query, CANDIDATES_PER_RETRIEVER
        )

        vector_ids = vector_search(
            self.collection, self.embed_model, query, CANDIDATES_PER_RETRIEVER
        )

        fused_ids = reciprocal_rank_fusion([bm25_ids, vector_ids])

        fused_top = fused_ids[:FUSED_TOP_N]

        reranked = rerank(
            self.reranker,
            query,
            self.chunk_id_to_text,
            fused_top,
            final_top_n,
        )

        results = []

        for chunk_id, score in reranked:

            metadata = self.chunk_id_to_metadata.get(chunk_id, {})

            results.append({
                "chunk_id": chunk_id,
                "score": float(score),
                "text": self.chunk_id_to_text[chunk_id],
                "article": metadata.get("article"),
                "livre": metadata.get("livre"),
                "titre": metadata.get("titre"),
                "chapitre": metadata.get("chapitre"),
                "section": metadata.get("section"),
                "modifications": json.loads(metadata.get("modifications", "[]")),
            })

        return results


############################################################
# MAIN (quick manual test)
############################################################

def main():

    retriever = HybridRetriever()

    query = input("\nEntrez votre question sur le Code du Travail: ")

    results = retriever.retrieve(query)

    print(f"\nTop {len(results)} résultats pour: {query!r}\n")

    for i, r in enumerate(results, start=1):
        print(f"{i}. {r['article']} (score={r['score']:.4f})")
        print(f"   {r['text'][:200]}...")
        print()


if __name__ == "__main__":
    main()
