import os
import json
from pathlib import Path

from groq import Groq

from Retrieve import HybridRetriever
from dotenv import load_dotenv
load_dotenv()
############################################################
# CONFIG
############################################################

# Set your key via environment variable rather than hardcoding
# it in the script, so it never ends up committed to GitHub:
#   export GROQ_API_KEY="your_key_here"   (Linux/Mac)
#   setx GROQ_API_KEY "your_key_here"     (Windows)
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

GROQ_MODEL_NAME = "llama-3.3-70b-versatile"

# Reranker score below this is treated as "not relevant enough"
# — tune this after a few manual tests once you see the actual
# score distribution for good vs bad matches on your data.
MIN_RELEVANCE_SCORE = 0.0

SYSTEM_PROMPT = """Tu es un assistant juridique spécialisé dans le Code du Travail tunisien.

Règles strictes à respecter :
1. Réponds UNIQUEMENT à partir des articles fournis dans le contexte ci-dessous. N'utilise aucune connaissance externe.
2. Cite systématiquement le ou les numéros d'article sur lesquels ta réponse s'appuie (ex: "Selon l'Article 18...").
3. Si le contexte fourni ne contient pas d'information permettant de répondre à la question, dis-le clairement : "Je ne trouve pas de disposition pertinente dans le Code du Travail pour répondre à cette question." Ne devine pas et n'invente rien.
4. Reste précis et factuel. N'ajoute pas d'interprétation personnelle ou de conseil juridique au-delà de ce que dit le texte.
5. Réponds dans la même langue que la question posée (français ou arabe selon le cas).
"""


############################################################
# STEP 1
# Build the context block from retrieved chunks
############################################################

def format_context(results):

    blocks = []

    for r in results:

        header_parts = [r["article"]]

        if r.get("chapitre"):
            header_parts.append(r["chapitre"])

        if r.get("titre"):
            header_parts.append(r["titre"])

        header = " — ".join(p for p in header_parts if p)

        blocks.append(f"[{header}]\n{r['text']}")

    return "\n\n---\n\n".join(blocks)


############################################################
# STEP 2
# Build the full prompt sent to the model
############################################################

def build_user_prompt(question, context):

    return f"""Contexte (articles du Code du Travail) :

{context}

---

Question : {question}

Réponds à la question en te basant uniquement sur le contexte ci-dessus, en citant les articles utilisés."""


############################################################
# STEP 3
# Call Groq
############################################################

def call_groq(client, system_prompt, user_prompt, model_name):

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,  # low temperature: factual/legal answers,
                           # not creative ones
    )

    return response.choices[0].message.content


############################################################
# STEP 4
# Full RAG pipeline
############################################################

class LaborLawRAG:

    def __init__(self):

        if not GROQ_API_KEY:
            raise RuntimeError(
                "GROQ_API_KEY not set. Export it as an environment "
                "variable before running this script."
            )

        print("Initialisation du retriever hybride...")
        self.retriever = HybridRetriever()

        self.groq_client = Groq(api_key=GROQ_API_KEY)

    def answer(self, question, top_n=5):

        results = self.retriever.retrieve(question, final_top_n=top_n)

        # Filter out weakly relevant chunks so the model isn't
        # fed noise it might try to force an answer out of.
        relevant_results = [
            r for r in results if r["score"] >= MIN_RELEVANCE_SCORE
        ]

        if not relevant_results:
            return {
                "answer": (
                    "Je ne trouve pas de disposition pertinente dans le "
                    "Code du Travail pour répondre à cette question."
                ),
                "sources": [],
            }

        context = format_context(relevant_results)

        user_prompt = build_user_prompt(question, context)

        answer_text = call_groq(
            self.groq_client,
            SYSTEM_PROMPT,
            user_prompt,
            GROQ_MODEL_NAME,
        )

        sources = [
            {
                "article": r["article"],
                "chapitre": r["chapitre"],
                "titre": r["titre"],
                "score": r["score"],
            }
            for r in relevant_results
        ]

        return {
            "answer": answer_text,
            "sources": sources,
        }


############################################################
# MAIN (quick manual test)
############################################################

def main():

    rag = LaborLawRAG()

    question = input("\nEntrez votre question sur le Code du Travail: ")

    result = rag.answer(question)

    print(f"\n--- Réponse ---\n{result['answer']}")

    print(f"\n--- Sources ({len(result['sources'])}) ---")

    for s in result["sources"]:
        print(f"  {s['article']} (score={s['score']:.4f}) — {s['chapitre']}")


if __name__ == "__main__":
    main()
