# import a_CONSTANTS as C
import tokenizer as T
import load_data as D
import pandas as pd


def main() -> None:
    D.make_data_dir()

    merged: pd.DataFrame = D.process_dataframes(*D.get_dataset_contents())

    for sentence in T.extract_ngrams(merged, 2):
        print(sentence)


if __name__ == "__main__":
    main()
