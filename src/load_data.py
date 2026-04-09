from huggingface_hub import hf_hub_download
import pandas as pd
import a_CONSTANTS as C
from pathlib import Path

THIS_FILES_PATH = Path(__file__)
ROOT_DIR = THIS_FILES_PATH.parents[1]
DATA_PATH = ROOT_DIR / C.DATA_FOLDER_NAME


def make_data_dir():

    if not DATA_PATH.exists():
        DATA_PATH.mkdir(exist_ok=True)


def download_data():
    rename_dict = {"Sentence": "text", "Emoji": "emoji"}
    df_1 = pd.read_csv(
        hf_hub_download(
            repo_id=C.TEXT_2_EMOJI_ID,
            filename=C.TEXT_2_EMOJI_FILE,
            repo_type="dataset",
            local_dir=DATA_PATH,
        )
    )
    df_2 = pd.read_csv(
        hf_hub_download(
            repo_id=C.SENTIMENT_2_EMOJI_ID,
            filename=C.SENTIMENT_2_EMOJI_TRAIN,
            repo_type="dataset",
            local_dir=DATA_PATH,
        )
    )
    df_3 = pd.read_csv(
        hf_hub_download(
            repo_id=C.SENTIMENT_2_EMOJI_ID,
            filename=C.SENTIMENT_2_EMOJI_TEST,
            repo_type="dataset",
            local_dir=DATA_PATH,
        )
    )
    df_2.columns = df_2.columns.str.strip()
    df_3.columns = df_3.columns.str.strip()
    df_2 = df_2.rename(columns=rename_dict)
    df_3 = df_3.rename(columns=rename_dict)
    df_4 = pd.concat([df_1, df_2, df_3], ignore_index=False)
    print(df_4.tail(20))
    df_4.to_csv(DATA_PATH / "merged.csv")


def main():
    make_data_dir()
    download_data()


main()
