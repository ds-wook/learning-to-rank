# This code is a Python file that defines a function for preprocessing and loading datasets.
# The function reads in the dataset, performs necessary preprocessing tasks,
# and transforms it into a format that can be used for model training.
from __future__ import annotations

from pathlib import Path

import pandas as pd
from omegaconf import DictConfig, OmegaConf

from tools import create_genre_flags


def load_dataset(cfg: DictConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    path = Path(cfg.data.path)
    anime_info_df = pd.read_csv(path / "anime_info.csv")
    relavence_scores = pd.read_csv(path / "relavence_scores.csv")
    user_info = pd.read_csv(path / "user_info.csv")
    relavence_scores = relavence_scores[~(relavence_scores["user_id"] == 11100)]
    popular_genres = OmegaConf.to_container(cfg.data.popular_genres)
    features = OmegaConf.to_container(cfg.data.features)

    anime_genre_info_df = create_genre_flags(anime_info_df, popular_genres)

    anime_info_df_final = anime_info_df.merge(anime_genre_info_df, on="anime_id")
    anime_info_df_final = anime_info_df_final.drop(columns=["Genres"])
    anime_info_df_final.columns = [
        col if col == "anime_id" else f"ANIME_FEATURE {col}".upper() for col in anime_info_df_final.columns
    ]
    user_info.columns = [col if col == "user_id" else f"USER_FEATURE {col}".upper() for col in user_info.columns]

    train_interim = relavence_scores.merge(anime_info_df_final, on="anime_id")
    train = train_interim.merge(user_info, how="inner")
    na_counts = train.isna().sum() * 100 / len(train)

    train_processed = train.drop(na_counts[na_counts > 50].index, axis=1)

    train_processed = train_processed.sort_values(by="user_id")
    train_processed = train_processed.set_index("user_id")

    test_size = int(1e5)
    X, y = train_processed[features], train_processed[cfg.data.target].apply(lambda x: int(x * 10))
    test_idx_start = len(X) - test_size
    X_train, X_test, y_train, y_test = (
        X.iloc[0:test_idx_start],
        X.iloc[test_idx_start:],
        y.iloc[0:test_idx_start],
        y.iloc[test_idx_start:],
    )

    return X_train, X_test, y_train, y_test
