from typing import Iterable

import numpy as np


def _dcg_at_k(actual: list[int], predicted: list[int], k: int = 10) -> float:
    """Compute DCG@K for a single user."""
    if len(predicted) > k:
        predicted = predicted[:k]

    dcg = 0.0
    for i, p in enumerate(predicted):
        if p in actual:
            # DCG 계산: relevance / log2(position + 1)
            dcg += 1.0 / np.log2(i + 2)  # log2(i+2) to handle 1-based indexing

    return dcg


def _idcg_at_k(actual: list[int], k: int = 10) -> float:
    """Compute ideal DCG@K for a single user."""
    actual = actual[:k]
    idcg = sum(1.0 / np.log2(i + 2) for i in range(len(actual)))  # Ideal ranking

    return idcg


def _ndcg_at_k(actual: list[int], predicted: list[int], k: int = 10) -> float:
    """Compute NDCG@K for a single user."""
    dcg = _dcg_at_k(actual, predicted, k)
    idcg = _idcg_at_k(actual, k)

    if idcg == 0:
        return 0.0

    return dcg / idcg


def ndcg_at_k(actual: Iterable, predicted: Iterable, k: int = 12) -> float:
    """Compute mean NDCG@K across all users.

    Parameters
    ----------
    actual : Iterable
        Label (ground truth).
    predicted : Iterable
        Predictions.
    k : int, optional
        k, by default ``12``.

    Returns
    -------
    float
        Mean NDCG@K.
    """
    return np.mean([_ndcg_at_k(a, p, k) for a, p in zip(actual, predicted) if a is not None])


def _hit_at_k(actual: list[int], predicted: list[int], k: int = 10) -> float:
    """Compute Hit@K for a single user."""
    if len(predicted) > k:
        predicted = predicted[:k]

    # 적어도 하나의 관련 항목이 추천에 포함되면 Hit
    return 1.0 if any(p in actual for p in predicted) else 0.0


def hit_at_k(actual: Iterable, predicted: Iterable, k: int = 10) -> float:
    """Compute mean Hit@K across all users.

    Parameters
    ----------
    actual : Iterable
        Label (ground truth).
    predicted : Iterable
        Predictions.
    k : int, optional
        k, by default ``10``.

    Returns
    -------
    float
        Mean Hit@K.
    """
    return np.mean([_hit_at_k(a, p, k) for a, p in zip(actual, predicted) if a is not None])


def _recall_at_k(actual: list[int], predicted: list[int], k: int = 10) -> float:
    """Compute Recall@K for a single user."""
    if len(predicted) > k:
        predicted = predicted[:k]

    relevant_items = [p for p in predicted if p in actual]
    if len(actual) == 0:
        return 0.0

    # Recall: 추천된 항목 중 실제 관련 항목의 비율
    return len(relevant_items) / len(actual)


def recall_at_k(actual: Iterable, predicted: Iterable, k: int = 10) -> float:
    """Compute mean Recall@K across all users.

    Parameters
    ----------
    actual : Iterable
        Label (ground truth).
    predicted : Iterable
        Predictions.
    k : int, optional
        k, by default ``10``.

    Returns
    -------
    float
        Mean Recall@K.
    """
    return np.mean([_recall_at_k(a, p, k) for a, p in zip(actual, predicted) if a is not None])
