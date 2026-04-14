from nltk import download  # type: ignore
from nltk.tokenize import word_tokenize  # type: ignore
from nltk.corpus import stopwords  # type: ignore
from nltk.stem import WordNetLemmatizer  # type: ignore
import pandas as pd
import a_CONSTANTS as C
from nltk.tokenize import RegexpTokenizer  # type: ignore


download("punkt_tab")
download("stopwords")
download("wordnet")
download("omw-1.4")


def tokenize_merged(merged: pd.DataFrame) -> list[list[str]]:
    results: list[list[str]] = []

    for _, row in merged.iterrows():
        results.append(word_tokenize(str(row[C.CSV_COLUMN])))

    return results


def clean_stopwords(tokenized: list[list[str]]) -> list[list[str]]:
    stop_words: set[str] = set(stopwords.words("english"))  # type: ignore
    results: list[list[str]] = []

    for tokens in tokenized:
        results.append([token for token in tokens if token not in stop_words])

    return results


def remove_punc(merged: pd.DataFrame) -> list[list[str]]:
    results: list[list[str]] = []

    for _, row in merged.iterrows():
        results.append(
            [
                token
                for token in RegexpTokenizer(r"\w+").tokenize(  # type: ignore
                    str(row[C.CSV_COLUMN])
                )
            ]
        )

    return results


def lemmatize_tokens(tokenized: list[list[str]]) -> list[list[str]]:
    lemmatizer = WordNetLemmatizer()
    results: list[list[str]] = []

    for tokens in tokenized:
        results.append([lemmatizer.lemmatize(token) for token in tokens])

    return results
