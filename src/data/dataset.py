from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from features import create_genre_flags


def create_target_features(X: pd.DataFrame) -> pd.DataFrame:
    X["target"] = (X["relavence_score"] == 1).astype(int)
    X = X.drop(columns=["relavence_score"])
    return X


def negative_sampling(cfg: DictConfig, df: pd.DataFrame) -> pd.DataFrame:
    # set random seed
    np.random.seed(cfg.data.seed)

    # Get list of anime watched by each user
    user_2_anime_df = df.groupby("user_id").agg({"anime_id": list})
    user_2_anime_map = dict(zip(user_2_anime_df.index, user_2_anime_df["anime_id"]))

    # Get all unique anime and users
    candidate_pool = df["anime_id"].unique().tolist()
    all_users = list(user_2_anime_map.keys())

    # Calculate anime popularity
    anime_popularity = df["anime_id"].value_counts()

    neg_samples = []
    for user_id in tqdm(all_users, desc="sampling negative samples"):
        # Get anime that user hasn't watched
        user_anime = set(user_2_anime_map[user_id])
        available_anime = list(set(candidate_pool) - user_anime)

        if not available_anime:
            continue

        # Calculate sampling size (n% of available anime)
        n_samples = max(1, int(len(available_anime) * cfg.data.negative_sampling_ratio))
        n_samples = min(n_samples, 9000)

        # Calculate sampling probabilities based on popularity
        available_probs = anime_popularity[available_anime]
        available_probs = available_probs / available_probs.sum()

        # Sample negative anime
        sampled_anime = np.random.choice(available_anime, size=n_samples, p=available_probs, replace=False)

        # Create negative samples
        neg_samples.extend([{"user_id": user_id, "anime_id": anime_id, "target": 0} for anime_id in sampled_anime])

    # Convert to DataFrame and combine with original data
    neg_df = pd.DataFrame(neg_samples)
    all_data = pd.concat([df, neg_df], ignore_index=True)

    return all_data


def limit_samples_per_user(cfg: DictConfig, df: pd.DataFrame, limit: int = 10000) -> pd.DataFrame:
    """
    Limit total samples per user to 10000 (including both positive and negative)
    Prioritize positive samples
    Args:
        df: pd.DataFrame
        limit: int
    Returns:
        pd.DataFrame
    """
    user_counts = df.groupby("user_id").size()
    users_to_limit = user_counts[user_counts > limit].index

    if len(users_to_limit) > 0:
        print(f"\nLimiting total samples for {len(users_to_limit)} users to {limit}")
        limited_data = []
        for user_id in users_to_limit:
            user_data = df[df["user_id"] == user_id]
            # Prioritize positive samples
            pos_data = user_data[user_data["target"] == 1]
            neg_data = user_data[user_data["target"] == 0]

            if len(pos_data) >= limit:
                # If positive samples exceed limit, sample from them
                sampled_data = pos_data.sample(n=limit, random_state=cfg.data.seed)
            else:
                # Otherwise, keep all positive samples and sample from negative
                n_neg_samples = limit - len(pos_data)
                sampled_neg = neg_data.sample(n=n_neg_samples, random_state=cfg.data.seed)
                sampled_data = pd.concat([pos_data, sampled_neg], ignore_index=True)

            limited_data.append(sampled_data)

        other_data = df[~df["user_id"].isin(users_to_limit)]

        return pd.concat([*limited_data, other_data], ignore_index=True)

    return df


def load_dataset(cfg: DictConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    path = Path(cfg.data.path)
    anime_info_df = pd.read_csv(path / "anime_info.csv")
    anime_info_df["is_dirty"] = anime_info_df["Genres"].map(lambda x: 1 if "Hentai" in str(x) else 0)
    anime_info_df = anime_info_df[anime_info_df["is_dirty"] == 0]
    anime_info_df = anime_info_df.drop(columns=["is_dirty"])

    relavence_scores = pd.read_csv(path / "relavence_scores.csv")
    user_info = pd.read_csv(path / "user_info.csv")

    # Filter out problematic users
    problematic_users = [10255]  # Add more user_ids if needed
    relavence_scores = relavence_scores[~relavence_scores["user_id"].isin(problematic_users)]
    user_info = user_info[~user_info["user_id"].isin(problematic_users)]

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

    interim = relavence_scores.merge(anime_info_df_final, on="anime_id")
    train, valid = train_test_split(interim, test_size=0.2, random_state=cfg.data.seed)

    train = train.merge(user_info, how="inner")
    valid = valid.merge(user_info, how="inner")

    train = create_target_features(train)
    valid = create_target_features(valid)

    pos_train = train[train["target"] == 1]
    neg_train = negative_sampling(cfg, train)
    pos_valid = valid[valid["target"] == 1]
    neg_valid = negative_sampling(cfg, valid)

    train = pd.concat([pos_train, neg_train], ignore_index=True)
    valid = pd.concat([pos_valid, neg_valid], ignore_index=True)

    train = limit_samples_per_user(cfg, train)
    valid = limit_samples_per_user(cfg, valid)

    train = train.sort_values(by="user_id")
    train = train.set_index("user_id")
    valid = valid.sort_values(by="user_id")
    valid = valid.set_index("user_id")

    X_train, X_valid, y_train, y_valid = (
        train[features],
        valid[features],
        train[cfg.data.target],
        valid[cfg.data.target],
    )

    return X_train, X_valid, y_train, y_valid


def load_test_dataset(cfg: DictConfig) -> tuple[dict[int, list[str]], list[str], dict[int, list[str]]]:
    anime_info_df_final, relavence_scores, _ = load_dataset(cfg)
    user_2_anime_df = relavence_scores.groupby("user_id").agg({"anime_id": lambda x: list(set(x))})
    user_2_anime_map = dict(zip(user_2_anime_df.index, user_2_anime_df["anime_id"]))
    candidate_pool = anime_info_df_final["anime_id"].unique().tolist()
    anime_id_2_name = relavence_scores.drop_duplicates(subset=["anime_id", "Name"])[["anime_id", "Name"]]
    anime_id_2_name_map = dict(zip(anime_id_2_name["anime_id"], anime_id_2_name["Name"]))

    return user_2_anime_map, candidate_pool, anime_id_2_name_map
