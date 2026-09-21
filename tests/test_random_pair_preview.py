"""Random previews rotate whole paired samples across consecutive runs."""

import numpy as np

from tools.random_pair_preview import choose_indices


def test_next_preview_avoids_previous_pair_ids_when_pool_allows():
    ids = [f"frame-{index}" for index in range(12)]
    eligible = list(range(12))
    rng = np.random.default_rng(7)
    first = choose_indices(ids, eligible, 4, set(), rng)
    second = choose_indices(ids, eligible, 4, {ids[index] for index in first}, rng)

    assert len(first) == len(second) == 4
    assert len(set(first)) == len(set(second)) == 4
    assert set(first).isdisjoint(second)


def test_preview_seed_replays_same_selection():
    ids = [f"frame-{index}" for index in range(10)]
    draw = lambda: choose_indices(ids, list(range(10)), 4, set(), np.random.default_rng(91))
    assert draw() == draw()
