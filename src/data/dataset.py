# This code is a Python file that defines a function for preprocessing and loading datasets.
# The function reads in the dataset, performs necessary preprocessing tasks,
# and transforms it into a format that can be used for model training.
from __future__ import annotations

from pathlib import Path

import pandas as pd
from omegaconf import DictConfig, OmegaConf

from generator import create_genre_flags


def load_dataset(cfg: DictConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    path = Path(cfg.data.path)
    anime_info_df = pd.read_csv(path / "anime_info.csv")
    anime_info_df["is_dirty"] = anime_info_df["Genres"].map(lambda x: 1 if "Hentai" in str(x) else 0)
    anime_info_df = anime_info_df[anime_info_df["is_dirty"] == 0]
    anime_info_df = anime_info_df.drop(columns=["is_dirty"])

    relavence_scores = pd.read_csv(path / "relavence_scores.csv")
    user_info = pd.read_csv(path / "user_info.csv")
    relavence_scores = relavence_scores[~(relavence_scores["user_id"] == 11100)]
    popular_genres = OmegaConf.to_container(cfg.data.popular_genres)
    anime_genre_info_df = create_genre_flags(anime_info_df, popular_genres)

    anime_info_df_final = anime_info_df.merge(anime_genre_info_df, on="anime_id")

    anime_info_df_final = anime_info_df_final.drop(columns=["Genres"])
    anime_info_df_final.columns = [
        col if col == "anime_id" else f"ANIME_FEATURE {col}".upper() for col in anime_info_df_final.columns
    ]
    user_info.columns = [col if col == "user_id" else f"USER_FEATURE {col}".upper() for col in user_info.columns]

    return anime_info_df_final, relavence_scores, user_info


def load_train_dataset(cfg: DictConfig) -> pd.DataFrame:
    anime_info_df_final, relavence_scores, user_info = load_dataset(cfg)
    features = OmegaConf.to_container(cfg.data.features)

    train_interim = relavence_scores.merge(anime_info_df_final, on="anime_id")
    train = train_interim.merge(user_info, how="inner")

    train = train.sort_values(by="user_id")
    train = train.set_index("user_id")

    test_size = int(1e5)
    X, y = train[features], train[cfg.data.target].apply(lambda x: int(x * 10))
    test_idx_start = len(X) - test_size
    X_train, X_test, y_train, y_test = (
        X.iloc[0:test_idx_start],
        X.iloc[test_idx_start:],
        y.iloc[0:test_idx_start],
        y.iloc[test_idx_start:],
    )

    return X_train, X_test, y_train, y_test


def load_test_dataset(cfg: DictConfig) -> tuple[dict[int, list[str]], list[str], dict[int, list[str]]]:
    anime_info_df_final, relavence_scores, _ = load_dataset(cfg)
    user_2_anime_df = relavence_scores.groupby("user_id").agg({"anime_id": lambda x: list(set(x))})
    user_2_anime_map = dict(zip(user_2_anime_df.index, user_2_anime_df["anime_id"]))
    candidate_pool = anime_info_df_final["anime_id"].unique().tolist()
    anime_id_2_name = relavence_scores.drop_duplicates(subset=["anime_id", "Name"])[["anime_id", "Name"]]
    anime_id_2_name_map = dict(zip(anime_id_2_name["anime_id"], anime_id_2_name["Name"]))

    return user_2_anime_map, candidate_pool, anime_id_2_name_map
