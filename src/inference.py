from __future__ import annotations

from pathlib import Path

import hydra
import lightgbm as lgb
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from prettytable import PrettyTable

from data import load_dataset, load_test_dataset
from models import BulidModel
from tools import candidate_generation


def generate_predictions(
    cfg: DictConfig,
    user_id: int,
    user_2_anime_map: dict[int, list[str]],
    candidate_pool: list[str],
    feature_columns: list[str],
    anime_id_2_name_map: dict[int, list[str]],
    ranker: BulidModel,
    N: int = 100,
):
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
    predictions = predictions.sort_values(by="score", ascending=False).head(N)

    predictions[f"already_liked - sample[{N}]"] = [
        anime_id_2_name_map.get(id_) for id_ in already_liked[0 : len(predictions)]
    ]
    return predictions


@hydra.main(config_path="../config/", config_name="inference", version_base="1.3.1")
def _main(cfg: DictConfig):
    user_2_anime_map, candidate_pool, anime_id_2_name_map = load_test_dataset(cfg)

    ranker = lgb.Booster(model_file=Path(cfg.model_path) / f"{cfg.results}.model")
    predictions = generate_predictions(
        cfg=cfg,
        user_id=123,
        user_2_anime_map=user_2_anime_map,
        candidate_pool=candidate_pool,
        feature_columns=cfg.data.features,
        anime_id_2_name_map=anime_id_2_name_map,
        ranker=ranker,
        N=cfg.N,
    )

    table = PrettyTable()
    table.field_names = ["Anime Name", "Already Liked", "Predicted Score"]

    for _, row in predictions.iterrows():
        table.add_row([row["name"], row[f"already_liked - sample[{cfg.N}]"], f"{row['score']:.3f}"])

    print(table)


if __name__ == "__main__":
    _main()
