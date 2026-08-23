import json
from pathlib import Path

INPUT_JSON = Path("output/code_travail.json")


def main():

    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    articles = data["articles"]

    word_counts = [len(art["text"].split()) for art in articles]

    over_300 = [
        (art["article"], wc)
        for art, wc in zip(articles, word_counts)
        if wc > 300
    ]

    over_500 = [
        (art["article"], wc)
        for art, wc in zip(articles, word_counts)
        if wc > 500
    ]

    print(f"Total articles: {len(articles)}")
    print(f"Articles > 300 words: {len(over_300)}")
    print(f"Articles > 500 words: {len(over_500)}")
    print(f"Max word count: {max(word_counts)}")
    print(f"Average word count: {sum(word_counts) / len(word_counts):.1f}")

    if over_300:
        print("\nArticles over 300 words:")
        for name, wc in sorted(over_300, key=lambda x: -x[1]):
            print(f"  {name}: {wc} words")


if __name__ == "__main__":
    main()