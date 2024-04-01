from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostRanker, Pool
from omegaconf import DictConfig, OmegaConf

from models import BaseModel


class XGBoostTrainer(BaseModel):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)

    def _fit(
        self,
        X_train: pd.DataFrame | np.ndarray,
        y_train: pd.Series | np.ndarray,
        X_valid: pd.DataFrame | np.ndarray | None = None,
        y_valid: pd.Series | np.ndarray | None = None,
    ) -> xgb.Booster:
        dtrain = xgb.DMatrix(X_train, y_train)
        dvalid = xgb.DMatrix(X_valid, y_valid)

        params = OmegaConf.to_container(self.cfg.models.params)
        params["seed"] = self.cfg.models.seed

        model = xgb.train(
            params=params,
            dtrain=dtrain,
            evals=[(dtrain, "train"), (dvalid, "eval")],
            num_boost_round=self.cfg.models.num_boost_round,
            early_stopping_rounds=self.cfg.models.early_stopping_rounds,
            verbose_eval=self.cfg.models.verbose_eval,
        )

        return model


class CatBoostTrainer(BaseModel):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)

    def _fit(
        self,
        X_train: pd.DataFrame | np.ndarray,
        y_train: pd.Series | np.ndarray,
        X_valid: pd.DataFrame | np.ndarray | None = None,
        y_valid: pd.Series | np.ndarray | None = None,
    ) -> CatBoostRanker:
        train_groups = X_train.index.to_numpy()  # user_id query
        valid_groups = X_valid.index.to_numpy()  # user_id query
        train_set = Pool(X_train, y_train, cat_features=self.cfg.tools.categorical_features, group_id=train_groups)
        valid_set = Pool(X_valid, y_valid, cat_features=self.cfg.tools.categorical_features, group_id=valid_groups)

        params = OmegaConf.to_container(self.cfg.models.params)
        model = CatBoostRanker(random_state=self.cfg.models.seed, **params)

        model.fit(
            train_set,
            eval_set=valid_set,
            verbose_eval=self.cfg.models.verbose_eval,
            early_stopping_rounds=self.cfg.models.early_stopping_rounds,
        )

        return model


class LightGBMTrainer(BaseModel):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)

    def _fit(
        self,
        X_train: pd.DataFrame | np.ndarray,
        y_train: pd.Series | np.ndarray,
        X_valid: pd.DataFrame | np.ndarray | None = None,
        y_valid: pd.Series | np.ndarray | None = None,
    ) -> lgb.Booster:
        params = OmegaConf.to_container(self.cfg.models.params)
        params["seed"] = self.cfg.models.seed
        train_groups = X_train.groupby("user_id").size().to_numpy()
        valid_groups = X_valid.groupby("user_id").size().to_numpy()

        train_set = lgb.Dataset(
            X_train, y_train, params=params, group=train_groups, feature_name=self.cfg.data.features
        )
        valid_set = lgb.Dataset(
            X_valid, y_valid, params=params, group=valid_groups, feature_name=self.cfg.data.features
        )

        model = lgb.train(
            params=params,
            train_set=train_set,
            valid_sets=[train_set, valid_set],
            num_boost_round=self.cfg.models.num_boost_round,
            callbacks=[
                lgb.log_evaluation(self.cfg.models.verbose_eval),
                lgb.early_stopping(self.cfg.models.early_stopping_rounds),
            ],
        )

        return model
