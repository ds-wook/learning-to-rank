import os
import random

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


def seed_everything(seed: int = 42) -> None:
    os.environ["PYTHONASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def reduce_mem_usage(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Iterate through all the columns of a dataframe and modify the data type to reduce memory usage.
    """
    numerics = ["int16", "int32", "int64", "float16", "float32", "float64"]
    start_mem = df.memory_usage().sum() / 1024**2

    for col in tqdm(df.columns, leave=False):
        df[col] = _reduce_column_memory(df[col], numerics)

    end_mem = df.memory_usage().sum() / 1024**2

    if verbose:
        print(
            f"Mem. usage decreased to {end_mem:5.2f} Mb ({100 * (start_mem - end_mem) / start_mem:.1f}% reduction)"
        )

    return df


def _reduce_column_memory(col: pd.Series, numerics: list) -> pd.Series:
    col_type = col.dtypes
    if col_type in numerics:
        c_min = col.min()
        c_max = col.max()
        if str(col_type)[:3] == "int":
            col = _reduce_int_memory(col, c_min, c_max)
        else:
            col = _reduce_float_memory(col, c_min, c_max)

    return col


def _reduce_int_memory(col: pd.Series, c_min: int, c_max: int) -> pd.Series:
    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
        return col.astype(np.int8)

    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
        return col.astype(np.int16)

    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
        return col.astype(np.int32)

    elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
        return col.astype(np.int64)

    return col


def _reduce_float_memory(col: pd.Series, c_min: float, c_max: float) -> pd.Series:
    if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
        return col.astype(np.float16)

    elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
        return col.astype(np.float32)

    return col.astype(np.float64)
