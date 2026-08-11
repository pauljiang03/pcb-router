"""Tests for scripts/wm_guided_search.py. The load-bearing one is obs-builder
parity: manually-built episode rows must exactly equal what the real wrapped
env emits for the same actions."""
import numpy as np

from envs.board import check_tp_spacing


def _make_env():
    from train import make_env
    return make_env("test", 0, seed=11, num_traces=4,
                    reward_mode="single_layer", boards="central",
                    shaping="potential")


def _greedy_valid_sequence(env):
    """First still-valid candidate at each step; restores env state after."""
    inner = env._inner
    saved = (list(inner.placed_tps), inner.current_trace,
             inner.candidate_mask.copy())
    seq = []
    for _ in range(inner.num_traces):
        idx = int(np.argmax(inner.candidate_mask))
        seq.append(idx)
        inner.placed_tps.append(tuple(inner.candidates[idx]))
        inner.current_trace += 1
        inner._update_candidate_mask()
    inner.placed_tps, inner.current_trace, inner.candidate_mask = saved
    return seq


def test_obs_builder_matches_real_env_rows():
    from scripts.wm_guided_search import ObsBuilder

    env = _make_env()
    builder = ObsBuilder(env)  # consumes the one reset; board is now fixed
    inner = env._inner
    n = inner.num_traces
    seq = _greedy_valid_sequence(env)

    rows = builder.build(seq)
    rows2 = builder.build(seq)  # state restore makes builds reproducible
    for k in rows:
        assert np.array_equal(rows[k], rows2[k]), k

    # Cache conventions: zero action row 0, action[t] produced obs row t.
    assert rows["image"].shape == (n + 1, 64, 64, 3)
    assert not rows["action"][0].any()
    for t, idx in enumerate(seq):
        assert rows["action"][t + 1, idx] == 1.0
        assert rows["action"][t + 1].sum() == 1.0
    assert rows["is_first"][0] and not rows["is_first"][1:].any()
    assert rows["is_terminal"][-1] and not rows["is_terminal"][:-1].any()

    # Replay through the real env. No new reset: builder restored the
    # post-reset state, and a reset would resample the central-family board.
    real = [dict(image=builder._row0["image"], mask=builder._row0["mask"],
                 vector=builder._row0["vector"])]
    done = False
    for idx in seq:
        a = np.zeros(env.action_space.shape[0], np.float32)
        a[idx] = 1.0
        obs, _r, done, _info = env.step({"action": a})
        real.append(obs)
    assert done

    for t in range(n + 1):
        assert np.array_equal(rows["image"][t], real[t]["image"]), f"image row {t}"
        assert np.array_equal(rows["mask"][t], real[t]["mask"]), f"mask row {t}"
        assert np.array_equal(rows["vector"][t], real[t]["vector"]), f"vector row {t}"
    assert real[1]["is_first"] is False and real[-1]["is_terminal"] is True


def test_snap_and_mutations_stay_valid():
    from envs.board import (generate_candidate_grid, load_te_example,
                            smart_placement)
    from scripts.wm_guided_search import mutate_state, snap_to_indices

    board = load_te_example(num_traces=4, seed=11)
    cand, real = generate_candidate_grid(board, 6.5)
    n = 4

    state = snap_to_indices(smart_placement(board, n), cand, real)
    assert len(state) == n and len(set(state)) == n
    assert all(0 <= i < real for i in state)

    def pairwise_valid(st):
        coords = [tuple(cand[i]) for i in st]
        return all(check_tp_spacing(coords[:k], *coords[k])
                   for k in range(len(coords)))

    assert pairwise_valid(state)

    rng = np.random.RandomState(0)
    produced = 0
    for _ in range(300):
        trial = mutate_state(state, rng, cand, real)
        if trial is None:
            continue
        produced += 1
        assert len(trial) == n and len(set(trial)) == n
        assert all(0 <= i < real for i in trial)
        assert pairwise_valid(trial)  # env will never snap/penalize on replay
    assert produced > 100


def test_spearman_manual_implementation():
    from scripts.wm_guided_search import spearman

    assert spearman([1, 2, 3, 4, 5], [2, 4, 6, 8, 10]) == 1.0
    assert spearman([1, 2, 3, 4, 5], [10, 8, 6, 4, 2]) == -1.0
    # Ties get average ranks: [1.5,1.5,3.5,3.5] vs [1,2,3,4].
    assert np.isclose(spearman([1, 1, 2, 2], [1, 2, 3, 4]),
                      4 / np.sqrt(20.0))
