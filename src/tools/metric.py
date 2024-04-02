import numpy as np


def _ap_at_k(actual: list[float], predicted: list[float], k: int = 10) -> float:
    """
    Computes the average precision at k.
    This function computes the average prescision at k between two lists of
    items.
    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The average precision at k over the input lists
    """
    if not actual:
        return 0.0

    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        # first condition checks whether it is valid prediction
        # second condition checks if prediction is not repeated
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    return score / min(len(actual), k)


def map_at_k(actual: list[float], predicted: list[float], k: int = 10) -> float:
    """
    Computes the mean average precision at k.
    This function computes the mean average prescision at k between two lists
    of lists of items.
    Parameters
    ----------
    actual : list
             A list of lists of elements that are to be predicted
             (order doesn't matter in the lists)
    predicted : list
                A list of lists of predicted elements
                (order matters in the lists)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The mean average precision at k over the input lists
    """
    return np.mean([_ap_at_k(a, p, k) for a, p in zip(actual, predicted)])


def ndcg_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int = 10) -> float:
    """
    Normalized discounted cumulative gain (NDCG) at rank k

    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Ground truth (true relevance labels)

    y_score : array-like, shape = [n_samples]
        Predicted scores

    k : int
        Rank

    Returns
    -------
    ndcg : float, 0.0 ~ 1.0
    """
    actual = dcg_at_k(y_true, y_score, k)
    best = dcg_at_k(y_true, y_true, k)
    ndcg = actual / best
    return ndcg


def dcg_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int = 10) -> float:
    """
    Discounted cumulative gain (DCG) at rank k

    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Ground truth (true relevance labels)

    y_score : array-like, shape = [n_samples]
        Predicted scores

    k : int
        Rank

    Returns
    -------
    dcg : float
    """
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    gains = 2**y_true - 1
    discounts = np.log2(np.arange(2, gains.size + 2))
    dcg = np.sum(gains / discounts)
    return dcg
