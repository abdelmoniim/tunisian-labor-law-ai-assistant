import os

import gradio as gr
from dotenv import load_dotenv

from generate import LaborLawRAG

############################################################
# SETUP
############################################################

load_dotenv()

print("Chargement du pipeline RAG (cela peut prendre une minute)...")
rag = LaborLawRAG()
print("Pipeline prêt.")


############################################################
# FORMAT SOURCES FOR DISPLAY
############################################################

def format_sources_markdown(sources):

    if not sources:
        return "_Aucune source pertinente trouvée._"

    lines = []

    for s in sources:

        line = f"**{s['article']}**"

        if s.get("chapitre"):
            line += f" — {s['chapitre']}"

        if s.get("titre"):
            line += f" — {s['titre']}"

        line += f" _(score: {s['score']:.3f})_"

        lines.append(line)

    return "\n\n".join(lines)


############################################################
# MAIN CHAT FUNCTION
############################################################

def respond(question, history):

    if not question.strip():
        return "", history, "_Posez une question pour voir les sources._"

    result = rag.answer(question)

    history = history + [
        {"role": "user", "content": question},
        {"role": "assistant", "content": result["answer"]},
    ]

    sources_md = format_sources_markdown(result["sources"])

    return "", history, sources_md


############################################################
# BUILD UI
############################################################

def build_interface():

    with gr.Blocks(title="Assistant Code du Travail - Tunisie") as demo:

        gr.Markdown(
            """
            # ⚖️ Assistant Code du Travail Tunisien
            Posez une question en français sur le Code du Travail tunisien.
            Les réponses sont générées uniquement à partir du texte officiel,
            avec citation des articles utilisés.

            _Cet outil est un projet de démonstration technique et ne
            constitue pas un conseil juridique._
            """
        )

        with gr.Row():

            with gr.Column(scale=2):

                chatbot = gr.Chatbot(
                    label="Conversation",
                    height=500,
                )

                question_box = gr.Textbox(
                    label="Votre question",
                    placeholder="Ex: Quelle est la durée de la période d'essai ?",
                    lines=2,
                )

                with gr.Row():
                    submit_btn = gr.Button("Envoyer", variant="primary")
                    clear_btn = gr.Button("Effacer la conversation")

            with gr.Column(scale=1):

                gr.Markdown("### 📚 Articles cités")

                sources_display = gr.Markdown(
                    "_Les sources apparaîtront ici après votre question._"
                )

        submit_btn.click(
            fn=respond,
            inputs=[question_box, chatbot],
            outputs=[question_box, chatbot, sources_display],
        )

        question_box.submit(
            fn=respond,
            inputs=[question_box, chatbot],
            outputs=[question_box, chatbot, sources_display],
        )

        clear_btn.click(
            fn=lambda: (None, "_Les sources apparaîtront ici après votre question._"),
            inputs=[],
            outputs=[chatbot, sources_display],
        )

    return demo


############################################################
# MAIN
############################################################

def main():

    demo = build_interface()

    demo.launch()


if __name__ == "__main__":
    main()