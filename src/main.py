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


def create_emoji_mapping(
    merged: pd.DataFrame, model: SentenceTransformer
) -> list[tuple[str, NDArray[np.float32]]]:
    sents: list[str] = merged["text"].tolist()
    emoji_set: set[str] = set(
        emoji for emojis in merged["emoji_list"].tolist() for emoji in emojis
    )
    vectors: NDArray[np.float32] = encoder(model, sents)
    emoji_mapping: list[tuple[str, list[NDArray[np.float32]]]] = [
        (
            emoji,
            [
                vector
                for vector, emojis in zip(vectors, merged["emoji_list"].to_list())
                if emoji in emojis
            ],
        )
        for emoji in emoji_set
    ]

    return [(emoji, np.mean(vec, 0)) for emoji, vec in emoji_mapping]


def encoder(
    model: SentenceTransformer, t_input: list[str] | str | pd.Series
) -> NDArray[np.float32]:
    # Normalized the vectors to make the cosine similarity easier
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
) -> tuple[list[str], NDArray[np.float32]]:
    merged: pd.DataFrame = D.process_dataframes(*D.get_dataset_contents())
    merged = merged.dropna(subset=["text"])

    if USE_SAMPLE:
        merged = merged[:SAMPLE]

    merged["emoji_list"] = merged["emoji"].apply(T.process_emojis)
    mapping: list[tuple[str, NDArray[np.float32]]] = create_emoji_mapping(merged, model)
    vectors: NDArray[np.float32] = np.array([vector for _, vector in mapping])
    emojis: list[str] = [emoji for emoji, _ in mapping]
    return (emojis, vectors)


def get_similarities(
    vector: NDArray[np.float32], vectors: NDArray[np.float32]
) -> NDArray[np.float32]:

    return vectors @ vector


def get_top_k(similarities: NDArray[np.float32], k: int) -> NDArray[np.intp]:
    indices: NDArray[np.intp] = np.argpartition(similarities, -k)[-k:]
    return indices[np.argsort(similarities[indices])[::-1]]


def main(model: SentenceTransformer) -> None:
    if not D.make_data_dir() or RERUN:
        emojis, vec_array = initialize_data(model)
        D.save_data(vec_array, emojis)
    else:
        vec_array, emojis = D.load_data()

    # test: str = (
    #     "Being a nurse is a rollercoaster of emotions, from comforting patients to dealing with medical emergencies."
    # )

    # print(get_similarities(encoder(model, test), vec_array))


if __name__ == "__main__":
    # https://sbert.net/ for model
    if USE_GPU:
        model = SentenceTransformer(C.MODEL, device="cuda")
    else:
        model = SentenceTransformer(C.MODEL)
    main(model)
