import numpy as np


def candidate_generation(
    user_id: int, candidate_pool: list[str], user_to_anime_map: dict[int, list[str]], N: int
) -> tuple[list[str], list[str]]:
    already_interacted = user_to_anime_map[user_id]
    candidates = list(set(candidate_pool) - set(already_interacted))
    return already_interacted, np.random.choice(candidates, size=N)
