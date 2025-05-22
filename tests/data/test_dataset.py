try:
    import os
    import sys

    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../src"))

except ModuleNotFoundError:
    raise Exception("Module not found")

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from data.dataset import (
    create_target_features,
    limit_samples_per_user,
    load_dataset,
    load_test_dataset,
    load_train_dataset,
    negative_sampling,
)


@pytest.fixture
def sample_data():
    return pd.DataFrame(
        {"user_id": [1, 1, 2, 2, 3], "anime_id": [101, 102, 101, 103, 102], "relavence_score": [1, 0, 1, 1, 0]}
    )


@pytest.fixture
def sample_config():
    config = {
        "data": {
            "seed": 42,
            "negative_sampling_ratio": 0.1,
            "path": "test/data/fixtures",
            "popular_genres": ["Action", "Comedy", "Drama"],
            "features": ["user_id", "anime_id", "target"],
            "target": "target",
        }
    }
    return OmegaConf.create(config)


def test_create_target_features(sample_data):
    result = create_target_features(sample_data)

    assert "target" in result.columns
    assert "relavence_score" not in result.columns
    assert result["target"].dtype == int
    assert result["target"].isin([0, 1]).all()
    assert len(result) == len(sample_data)
