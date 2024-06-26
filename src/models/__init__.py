from omegaconf import DictConfig

from .base import BaseModel
from .boosting import CatBoostTrainer, LightGBMTrainer, XGBoostTrainer

BulidModel = CatBoostTrainer | LightGBMTrainer | XGBoostTrainer


def bulid_model(cfg: DictConfig) -> BulidModel:
    model_type = {
        "lightgbm": LightGBMTrainer(cfg),
        "xgboost": XGBoostTrainer(cfg),
        "catboost": CatBoostTrainer(cfg),
    }

    if trainer := model_type.get(cfg.models.name):
        return trainer

    else:
        raise NotImplementedError
