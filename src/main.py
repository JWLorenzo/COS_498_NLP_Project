import a_CONSTANTS as C
import tokenizer as T
import load_data as D
import pandas as pd
from sentence_transformers import SentenceTransformer
from numpy.typing import NDArray
import numpy as np
from torch import Tensor

SAMPLE = 20000
BATCH_SIZE = 128

USE_SAMPLE = False
USE_GPU = True
RERUN = False


def generate_indexes(
    vectors: NDArray[np.float32], frame: pd.DataFrame
) -> tuple[NDArray[np.float32], list[str]]:
    vec_list: list[Tensor] = [
        vector for vector, emojis in zip(vectors, frame["emoji_list"]) for _ in emojis
    ]

    emojis: list[str] = [emoji for emojis in frame["emoji_list"] for emoji in emojis]

    vec_array: NDArray[np.float32] = np.array(vec_list)

    return (vec_array, emojis)


def encoder(
    model: SentenceTransformer, t_input: list[str] | str | pd.Series
) -> NDArray[np.float32]:
    return np.array(
        model.encode(  # type: ignore
            t_input,
            batch_size=BATCH_SIZE,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
    )


def initialize_data(
    model: SentenceTransformer,
) -> tuple[NDArray[np.float32], list[str]]:
    merged: pd.DataFrame = D.process_dataframes(*D.get_dataset_contents())
    merged = merged.dropna(subset=["text"])

    if USE_SAMPLE:
        merged = merged[:SAMPLE]

    merged["emoji_list"] = merged["emoji"].apply(T.process_emojis)

    vectors: NDArray[np.float32] = encoder(model, merged["text"].tolist())
    # Normalized the vectors to make the cosine similarity easier
    return generate_indexes(vectors, merged)


def get_similarities(
    vector: NDArray[np.float32], vectors: NDArray[np.float32]
) -> NDArray[np.float32]:

    return vectors @ vector


def get_top_k(similarities: NDArray[np.float32], k: int) -> NDArray[np.float32]:
    indices: NDArray[np.intp] | NDArray[np.float32] = np.argpartition(similarities, -k)[
        -k:
    ]
    return indices[np.argsort(similarities[indices])[::-1]]


def main(model: SentenceTransformer) -> None:
    if not D.make_data_dir() or RERUN:
        vec_array, emojis = initialize_data(model)
        D.save_data(vec_array, emojis)
    else:
        vec_array, emojis = D.load_data()

    test: str = (
        "Being a nurse is a rollercoaster of emotions, from comforting patients to dealing with medical emergencies."
    )

    print(get_similarities(encoder(model, test), vec_array))


if __name__ == "__main__":
    # https://sbert.net/ for model
    if USE_GPU:
        model = SentenceTransformer(C.MODEL, device="cuda")
    else:
        model = SentenceTransformer(C.MODEL)
    main(model)
