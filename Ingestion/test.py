import fitz
import re
import json
import unicodedata
from pathlib import Path

############################################################
# STEP 1
# Read PDF
############################################################

def extract_pdf_text(pdf_path):

    doc = fitz.open(pdf_path)

    pages = []

    for page in doc:
        pages.append(page.get_text())

    return "\n".join(pages)


############################################################
# STEP 2
# Clean text
############################################################

def clean_text(text):

    # Unicode normalization FIRST, before any regex matching,
    # so accented characters (é, à, ç...) are in a consistent
    # form. Prevents silent embedding/retrieval degradation
    # from mixed precomposed/decomposed accents.
    text = unicodedata.normalize("NFC", text)

    # Rejoin words that were hyphenated across a line break by
    # the PDF layout (e.g. "informa-\ntion" -> "information").
    # Must run on the raw multi-line text, before we strip/join
    # lines below.
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

    lines = []

    for line in text.splitlines():

        line = line.strip()

        if not line:
            continue

        # remove page numbers
        if re.fullmatch(r"\d+", line):
            continue

        # remove repeating header
        if "Imprimerie Officielle" in line:
            continue

        lines.append(line)

    text = "\n".join(lines)

    # remove duplicated spaces (but not newlines)
    text = re.sub(r"[ \t]+", " ", text)

    return text


#clean_pages=clean_text(pages)

#print(clean_pages[:20000])

############################################################
# STEP 3
# Parse structure
############################################################

def parse_code(text):

    livre = None
    titre = None
    chapitre = None
    section = None

    articles = []
    current_article = None

    lines = text.split("\n")

    for line in lines:
        # ---------------- STOP CONDITION ----------------
        # If we reach "Table de Matières", stop parsing
        if re.search(r"TABLE DES MATIERES", line, re.I):
            break

        # ---------------- LIVRE ----------------
        if re.match(r"^LIVRE", line, re.I):
            livre = line
            continue

        # ---------------- TITRE ----------------
        if re.match(r"^TITRE", line, re.I):
            titre = line
            continue

        # ---------------- CHAPITRE ----------------
        if re.match(r"^Chapitre", line, re.I):
            chapitre = line
            continue

        # ---------------- SECTION ----------------
        if re.match(r"^Section", line, re.I):
            section = line
            continue

        # ---------------- ARTICLE ----------------
        if re.match(r"^Article", line, re.I):

            # finalize previous article
            if current_article:
                current_article["text"] = current_article["text"].strip()
                articles.append(current_article)

            # capture article name (allow hyphenated numbers and .-)
            # capture article name (allow bis/ter/quater, hyphenated numbers, and optional .-)
            # capture article name (numbers + optional suffixes + optional .-)
            m = re.match(r"^(Article\s+\d+(?:[-\.]\d+)?(?:\s+(?:bis|ter|quater|quinter|sexies|septies|octies|nonies|decies))?)",line,re.I)
            article_name = m.group(1) if m else line

            # normalize article name: remove trailing punctuation like "." or ".-"
            #article_name = re.sub(r"[.\-]+$", "", article_name).strip()


            # extract remaining text after article name
            article_text = line[len(article_name):].strip()


            modifications = []
            # detect modifications inside parentheses (anywhere in the text)
            for mod in re.findall(r"\((.*?)\)", article_text):
                law_match = re.search(r"loi n°\s*([0-9\-]+)", mod, re.I)
                date_match = re.search(r"(\d{1,2}\s+\w+\s+\d{4})", mod)

                law_number = law_match.group(1) if law_match else None
                date = date_match.group(1) if date_match else None

                if "Modifié par la loi" in mod:
                    modifications.append({"type": "modifié", "law_number": law_number, "date": date, "note": mod})
                elif "Ajouté par la loi" in mod:
                    modifications.append({"type": "ajouté", "law_number": law_number, "date": date, "note": mod})
                elif "Abrogé par la loi" in mod:
                    modifications.append({"type": "abrogé", "law_number": law_number, "date": date, "note": mod})
                else:
                    modifications.append({"type": "autre", "law_number": law_number, "date": date, "note": mod})

            # clean article text: remove ALL parenthetical notes
            article_text_clean = re.sub(r"\(.*?\)", "", article_text).strip(" .-")




            current_article = {
                "article": article_name,
                "livre": livre,
                "titre": titre,
                "chapitre": chapitre,
                "section": section,
                "text": article_text_clean,
                "modifications": modifications
            }

            continue

        # ---------------- CONTINUE ARTICLE ----------------
        if current_article:
            if current_article["text"]:
                current_article["text"] += "\n" + line
            else:
                current_article["text"] = line

    # finalize last article
    if current_article:
        current_article["text"] = current_article["text"].strip()
        articles.append(current_article)

    return articles


#articls=parse_code(clean_pages)
#print(articls[35])



############################################################
# STEP 4
# Create final JSON
############################################################

def create_json(articles):

    output = {
        "metadata": {
            "title": "Code du Travail",
            "version": 2016
        },
        "articles": []
    }

    for i, art in enumerate(articles, start=1):

        output["articles"].append({

            "id": i,

            "article": art["article"],

            "livre": art["livre"],

            "titre": art["titre"],

            "chapitre": art["chapitre"],

            "section": art["section"],

            "text": art["text"],
            # NEW: include modifications if present
            "modifications": art.get("modifications", [])

        })

    return output


#final_articles=create_json(articls)
#print(final_articles["articles"][500])


############################################################
# MAIN
############################################################

def main():

    pdf = Path("./data/code_de_travail_2016_6.pdf")

    text = extract_pdf_text(pdf)

    text = clean_text(text)

    articles = parse_code(text)

    result = create_json(articles)

    Path("output").mkdir(exist_ok=True)

    with open(
        "output/code_travail.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            result,
            f,
            ensure_ascii=False,
            indent=2
        )

    print(f"{len(articles)} articles extracted.")


if __name__ == "__main__":
    main()