from __future__ import annotations

from pathlib import Path

import hydra
import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostRanker
from omegaconf import DictConfig
from prettytable import PrettyTable
from tqdm import tqdm

from data import candidate_generation, load_dataset, load_test_dataset
from evaluation import ndcg_at_k
from models import BulidModel


def generate_predictions(
    cfg: DictConfig,
    user_id: int,
    user_2_anime_map: dict[int, list[str]],
    candidate_pool: list[str],
    feature_columns: list[str],
    anime_id_2_name_map: dict[int, list[str]],
    ranker: BulidModel,
    top_k: int = 100,
) -> pd.DataFrame:
    anime_info_df_final, _, user_info = load_dataset(cfg)
    already_liked, candidates = candidate_generation(user_id, candidate_pool, user_2_anime_map, N=10000)
    candidates_df = pd.DataFrame(data=pd.Series(candidates, name="anime_id"))
    features = anime_info_df_final.merge(candidates_df)
    features["user_id"] = user_id
    features = features.merge(user_info)

    already_liked = list(already_liked)
    if len(already_liked) < len(candidates):
        append_list = np.full(fill_value=-1, shape=(len(candidates) - len(already_liked)))
        already_liked.extend(list(append_list))

    predictions = pd.DataFrame(index=candidates)
    predictions["name"] = np.array([anime_id_2_name_map.get(id_) for id_ in candidates])
    predictions["score"] = ranker.predict(features[feature_columns])
    predictions = predictions.sort_values(by="score", ascending=False).head(top_k)

    predictions["already_liked"] = [anime_id_2_name_map.get(_id) for _id in already_liked[:top_k]]
    return predictions


@hydra.main(config_path="../config/", config_name="inference", version_base="1.3.1")
def _main(cfg: DictConfig):
    user_2_anime_map, candidate_pool, anime_id_2_name_map = load_test_dataset(cfg)

    ranker = (
        lgb.Booster(model_file=Path(cfg.models.model_path) / f"{cfg.models.results}.model")
        if cfg.models.name == "lightgbm"
        else CatBoostRanker().load_model(Path(cfg.models.model_path) / f"{cfg.models.results}.model")
    )

    user_ids = list(user_2_anime_map.keys())
    already_likes = []
    names = []

    for user_id in tqdm(user_ids[: cfg.top_k]):
        predictions = generate_predictions(
            cfg=cfg,
            user_id=user_id,
            user_2_anime_map=user_2_anime_map,
            candidate_pool=candidate_pool,
            feature_columns=cfg.data.features,
            anime_id_2_name_map=anime_id_2_name_map,
            ranker=ranker,
        )
        already_likes.append(predictions["already_liked"].to_numpy())
        names.append(predictions["name"].tolist())

    output = pd.DataFrame({"user": user_ids[: cfg.top_k], "already_liked": already_likes, "name": names})

    # calculate NDCG@K
    table = PrettyTable()
    table.field_names = ["K", "NDCG@K"]
    top_k = [20, 50, 100, 200]
    already_liked = output["already_liked"].to_numpy()
    predicted_scores = output["name"].to_numpy()

    for k in top_k:
        table.add_row([k, ndcg_at_k(already_liked, predicted_scores, k)])

    print(table)


if __name__ == "__main__":
    _main()
