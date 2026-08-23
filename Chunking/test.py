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


articles=load_articles(INPUT_JSON)
#print(articles[125])


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


sent=split_sentences(articles[125]['text'])
print(sent)