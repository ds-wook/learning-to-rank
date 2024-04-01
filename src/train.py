from __future__ import annotations

import warnings
from pathlib import Path

import hydra
from omegaconf import DictConfig

from data import load_train_dataset
from models import bulid_model


@hydra.main(config_path="../config/", config_name="train", version_base="1.2.0")
def _main(cfg: DictConfig):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)

        # load dataset
        X_train, X_test, y_train, y_test = load_train_dataset(cfg)

        # choose trainer
        trainer = bulid_model(cfg)

        # train model
        ranker = trainer.fit(X_train, y_train, X_test, y_test)

        # save model
        ranker.save_model(Path(cfg.models.model_path) / f"{cfg.models.results}.model")


if __name__ == "__main__":
    _main()
