import pandas as pd
import a_CONSTANTS as C
import spacy
from spacy.tokens import Token
from spacy.language import Language
import emoji


## Useless now, but keeping incase ##

# def add_padding(
#     sentence: list[str], n: int, left: str = "#", right: str = "#"
# ) -> list[str]:
#     return [left] * (n - 1) + sentence + [right] * (n - 1)

# def pad_sentences(sentences: list[list[str]], n: int) -> list[list[str]]:
#     results: list[list[str]] = []
#     for sentence in sentences:
#         result = add_padding(sentence, n)
#         print(result)
#         results.append(result)
#     return results


def process_emojis(emoji_str: str) -> list[str]:
    ej_list: list[str] = []
    for ej in emoji.emoji_list(emoji_str):
        ej_list.append(ej["emoji"])
    return ej_list


def clean_spaCy_single(text: str, nlp: Language) -> list[Token]:
    doc = nlp(text)
    tokens = [
        token
        for token in doc
        if not any([token.is_punct, token.is_space, token.is_stop])
    ]
    return tokens


def extract_ngram(tokens: list[Token], n: int) -> list[tuple[str, int, int]]:

    n_grams: list[tuple[str, int, int]] = []
    for idx in range(len(tokens) - (n - 1)):
        ngram: list[Token] = tokens[idx : idx + n]
        n_grams.append(
            (
                " ".join([word.text for word in ngram]),
                ngram[0].idx,
                ngram[-1].idx + len(ngram[-1].text),
            )
        )
    return n_grams


def extract_ngrams(merged: pd.DataFrame, n: int) -> list[list[tuple[str, int, int]]]:
    results: list[list[tuple[str, int, int]]] = []
    nlp: Language = spacy.load("en_core_web_sm")
    for _, row in merged.iterrows():
        result: list[Token] = clean_spaCy_single(str(row[C.CSV_COLUMN]), nlp)
        results.append(extract_ngram(result, n))

    return results
