# import a_CONSTANTS as C
import tokenizer as T
import load_data as D
import pandas as pd
from nltk.corpus import stopwords  # type: ignore


def main() -> None:
    D.make_data_dir()

    merged: pd.DataFrame = D.process_dataframes(*D.get_dataset_contents())

    # tokenized: list[list[str]] = T.tokenize_merged(merged)

    # stop_words_removed = T.clean_stopwords(tokenized)

    tokenized_no_punc: list[list[str]] = T.remove_punc(merged)

    no_stop_words_no_punc: list[list[str]] = T.clean_stopwords(tokenized_no_punc)

    lemmatized: list[list[str]] = T.lemmatize_tokens(no_stop_words_no_punc)

    for text in lemmatized[:100]:
        print(" ".join(text))


if __name__ == "__main__":
    main()
