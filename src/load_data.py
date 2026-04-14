from huggingface_hub import hf_hub_download  # type: ignore
import pandas as pd
import a_CONSTANTS as C
from pathlib import Path

THIS_FILES_PATH = Path(__file__)
ROOT_DIR = THIS_FILES_PATH.parents[1]
DATA_PATH = ROOT_DIR / C.DATA_FOLDER_NAME
RENAME_DICT = {"Sentence": "text", "Emoji": "emoji"}
MERGE_NAME = "merged.csv"


def make_data_dir():

    if not DATA_PATH.exists():
        DATA_PATH.mkdir(exist_ok=True)
    return True


def get_dataset_contents() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    csv_1: Path | str = DATA_PATH / C.TEXT_2_EMOJI_FILE
    csv_2: Path | str = DATA_PATH / C.SENTIMENT_2_EMOJI_TRAIN
    csv_3: Path | str = DATA_PATH / C.SENTIMENT_2_EMOJI_TEST

    assert not isinstance(csv_1, str)
    assert not isinstance(csv_2, str)
    assert not isinstance(csv_3, str)

    if not csv_1.exists():
        csv_1 = hf_hub_download(
            repo_id=C.TEXT_2_EMOJI_ID,
            filename=C.TEXT_2_EMOJI_FILE,
            repo_type="dataset",
            local_dir=DATA_PATH,
        )

    if not csv_2.exists():
        csv_2 = hf_hub_download(
            repo_id=C.SENTIMENT_2_EMOJI_ID,
            filename=C.SENTIMENT_2_EMOJI_TRAIN,
            repo_type="dataset",
            local_dir=DATA_PATH,
        )

    if not csv_3.exists():
        csv_3 = hf_hub_download(
            repo_id=C.SENTIMENT_2_EMOJI_ID,
            filename=C.SENTIMENT_2_EMOJI_TEST,
            repo_type="dataset",
            local_dir=DATA_PATH,
        )

    return pd.read_csv(csv_1), pd.read_csv(csv_2), pd.read_csv(csv_3)


def process_dataframes(
    df_1: pd.DataFrame, df_2: pd.DataFrame, df_3: pd.DataFrame
) -> pd.DataFrame:
    df_2.columns = df_2.columns.str.strip()
    df_3.columns = df_3.columns.str.strip()
    df_2 = df_2.rename(columns=RENAME_DICT)
    df_3 = df_3.rename(columns=RENAME_DICT)
    df_4 = pd.concat([df_1, df_2, df_3], ignore_index=False)
    df_4.to_csv(DATA_PATH / MERGE_NAME)

    return df_4
