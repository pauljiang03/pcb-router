"""Tests for the cold-start pipeline: potential reward shaping, expert demo
collection, fixed-fraction demo mixing, and the behavior-cloning actor loss."""
import argparse
import pathlib
import sys

import numpy as np
import pytest

from envs.pcb_env import TPPlacementEnv


def _first_valid_actions(env):
    """Deterministic rollout: always pick the first still-valid candidate."""
    actions = []
    obs, _ = env.reset(seed=None)
    done = False
    total = 0.0
    rewards = []
    while not done:
        action = int(np.argmax(env.candidate_mask))
        actions.append(action)
        obs, r, done, _, _ = env.step(action)
        rewards.append(r)
        total += r
    return actions, rewards, total


# ------------------------------------------------------------ reward shaping

def test_potential_shaping_preserves_episode_total():
    kw = dict(num_traces=4, seed=7, reward_mode="layer_aware",
              route_n_starts=1, route_max_iters=4, route_repair_passes=0)
    plain = TPPlacementEnv(shaping="none", **kw)
    shaped = TPPlacementEnv(shaping="potential", **kw)

    actions, plain_rewards, plain_total = _first_valid_actions(plain)

    shaped.reset(seed=None)
    shaped_rewards, done = [], False
    for a in actions:
        _, r, done, _, _ = shaped.step(a)
        shaped_rewards.append(r)
    assert done

    # Redistribution, not a new objective: totals match exactly...
    assert np.isclose(sum(shaped_rewards), plain_total, atol=1e-9)
    # ...but per-step rewards genuinely moved (credit was redistributed).
    assert any(abs(a - b) > 1e-6
               for a, b in zip(shaped_rewards[:-1], plain_rewards[:-1]))
    # Terminal components expose the potential for logging.
    assert "phi" in shaped._reward_components


def test_potential_tracks_placed_traces():
    from envs.board import wire_estimate
    from envs.pcb_env import PHI_W_LEN, PHI_W_MAXLEN, PHI_W_CROSS
    from envs.routing import count_crossings

    env = TPPlacementEnv(num_traces=3, seed=7, shaping="potential",
                         route_n_starts=1, route_max_iters=2,
                         route_repair_passes=0)
    env.reset(seed=None)
    for _ in range(2):  # stop before terminal so phi state survives
        env.step(int(np.argmax(env.candidate_mask)))

    # One estimate per placed TP, computed against the trace it serves
    # (catches off-by-one between placement order and trace index).
    assert len(env._est_lens) == len(env.placed_tps) == 2
    for k, tp in enumerate(env.placed_tps):
        trace = env.board.traces[k]
        expected = wire_estimate(env.board, trace, tp) + trace.breakout_length
        assert np.isclose(env._est_lens[k], expected)
        assert env._est_segs[k][0] == (trace.start_x, trace.start_y)

    diag = np.hypot(env.board.width, env.board.height)
    phi = (-PHI_W_LEN * sum(env._est_lens) / (env.num_traces * diag)
           - PHI_W_MAXLEN * max(env._est_lens) / diag
           - PHI_W_CROSS * count_crossings(env._est_segs))
    assert np.isclose(env._phi, phi)
    assert env._phi < 0.0


# ------------------------------------------------------------------- demos

def test_demo_collection_roundtrip(tmp_path):
    from demos import collect_demos
    from train import make_env

    def env_fn(offset):
        return make_env("demo", 0, seed=100 + offset, num_traces=4,
                        reward_mode="layer_aware", boards="central")

    made = collect_demos(tmp_path, env_fn, 1, verbose=False)
    assert made == 1
    files = list(pathlib.Path(tmp_path).glob("*.npz"))
    assert len(files) == 1

    from dreamerv3 import tools
    eps = tools.load_episodes(tmp_path)
    assert len(eps) == 1
    ep = next(iter(eps.values()))
    assert len(ep["reward"]) == 5  # num_traces + 1 rows

    # Same schema tools.simulate writes: policy and demo episodes must batch
    # together interchangeably.
    for key in ("image", "mask", "vector", "is_first", "is_last",
                "is_terminal", "action", "logprob", "reward", "discount"):
        assert key in ep, key
    from envs.board import MAX_CANDIDATES
    assert ep["action"].shape == (5, MAX_CANDIDATES)
    assert ep["vector"].shape == (5, 3 + 2 * 4 + 2 * MAX_CANDIDATES)
    assert ep["vector"].min() >= -1.0 and ep["vector"].max() <= 1.0
    assert not ep["action"][0].any()                    # zero reset filler
    assert (ep["action"][1:].sum(axis=1) == 1).all()    # one-hot decisions
    assert ep["is_first"][0] and not ep["is_first"][1:].any()
    assert ep["is_terminal"][-1]

    # Resumable: a second call with the same target generates nothing new.
    assert collect_demos(tmp_path, env_fn, 1, verbose=False) == 0


@pytest.mark.skipif(sys.platform != "linux",
                    reason="fork-based parallel demo collection is Linux-only")
def test_demo_collection_parallel(tmp_path):
    from demos import collect_demos
    from train import make_env

    def env_fn(offset):
        return make_env("demo", 0, seed=300 + offset, num_traces=4,
                        reward_mode="layer_aware", boards="central")

    made = collect_demos(tmp_path, env_fn, 4, verbose=False, workers=2)
    assert made == 4
    assert len(list(pathlib.Path(tmp_path).glob("*.npz"))) == 4
    # Workers cover disjoint seed slices; resume sees the full set.
    assert collect_demos(tmp_path, env_fn, 4, verbose=False, workers=2) == 0


def test_canonical_board_factory():
    from train import canonical_board_factory
    from envs.board import generate_candidate_grid, load_te_excel

    f = canonical_board_factory(20)
    a, b, a2 = f(1), f(2), f(1)
    pa = np.array([(t.start_x, t.start_y) for t in a.traces])
    pb = np.array([(t.start_x, t.start_y) for t in b.traces])
    pa2 = np.array([(t.start_x, t.start_y) for t in a2.traces])
    assert len(a.traces) == 20
    assert np.allclose(pa, pa2)              # deterministic per seed
    assert not np.allclose(pa, pb)           # jitter varies across seeds
    assert np.abs(pa - pb).max() <= 6.0 + 1e-6   # bounded shift
    # ~25% of episodes must be the EXACT unjittered board (scoreboard target).
    base_pins = np.array([(t.start_x, t.start_y)
                          for t in load_te_excel().traces])
    hits = sum(np.allclose(np.array([(t.start_x, t.start_y)
                                     for t in f(s).traces]), base_pins)
               for s in range(40))
    assert 3 <= hits <= 20, hits
    # The jittered board still yields a workable candidate grid.
    _cand, real = generate_candidate_grid(a, 6.5)
    assert real >= 60
    # Cluster shift moved obstacles with the pins (rigid translation).
    base = load_te_excel()
    d_pin = pa[0] - np.array([base.traces[0].start_x, base.traces[0].start_y])
    d_obs = np.array([a.rect_obstacles[0].cx - base.rect_obstacles[0].cx,
                      a.rect_obstacles[0].cy - base.rect_obstacles[0].cy])
    assert np.allclose(d_pin, d_obs, atol=1e-9)


def test_repair_placement_time_budget():
    import time
    from envs.board import (load_te_example, generate_candidate_grid,
                            repair_placement)
    board = load_te_example(num_traces=6, seed=0)
    cand, rc = generate_candidate_grid(board, 6.5)
    placed = [tuple(cand[i]) for i in range(6)]  # clustered, surely broken
    t0 = time.monotonic()
    out = repair_placement(board, placed, time_budget=0.0)
    assert out == placed                    # zero budget: no moves attempted
    assert time.monotonic() - t0 < 5.0      # and no routing performed


def test_expert_action_maps_to_valid_candidates():
    from demos import _expert_action
    env = TPPlacementEnv(num_traces=4, seed=3)
    env.reset(seed=None)
    # An exact candidate maps to itself.
    idx = 5
    target = tuple(env.candidates[idx])
    assert _expert_action(env, target) == idx
    # A masked-out candidate is replaced by a valid one.
    env.candidate_mask[idx] = False
    got = _expert_action(env, target)
    assert got != idx and env.candidate_mask[got]
    # No assignment still returns a valid candidate.
    got = _expert_action(env, None)
    assert env.candidate_mask[got]


# ------------------------------------------------------------------ mixing

def test_mixed_generator_fraction_and_tags():
    from train import tag_is_demo, mixed_generator

    def fake(source):
        while True:
            yield {"reward": np.zeros(4, np.float32), "src": source}

    demo = tag_is_demo(fake("demo"), 1.0)
    train = tag_is_demo(fake("train"), 0.0)

    seq = next(demo)
    assert seq["is_demo"].shape == (4,) and (seq["is_demo"] == 1.0).all()

    always = mixed_generator(train, demo, 1.0, seed=0)
    assert all(next(always)["src"] == "demo" for _ in range(5))
    never = mixed_generator(train, demo, 0.0, seed=0)
    assert all(next(never)["src"] == "train" for _ in range(5))
    sometimes = mixed_generator(train, demo, 0.5, seed=0)
    srcs = {next(sometimes)["src"] for _ in range(40)}
    assert srcs == {"demo", "train"}


# ---------------------------------------------------------------- BC loss

def _tiny_config(num_actions):
    import ruamel.yaml as yaml
    root = pathlib.Path(__file__).resolve().parents[1]
    cfg = yaml.YAML(typ="safe").load((root / "configs.yaml").read_text())["defaults"]
    cfg.update(dict(
        device="cpu", compile=False, units=16, dyn_hidden=16, dyn_deter=16,
        dyn_stoch=4, dyn_discrete=4, imag_horizon=5, batch_size=2,
        batch_length=8, num_actions=num_actions, video_pred_log=False,
        bc_scale=1.0, bc_decay=0, bc_floor=0.0,
    ))
    cfg["encoder"] = {**cfg["encoder"], "cnn_depth": 8, "mlp_units": 16}
    cfg["decoder"] = {**cfg["decoder"], "cnn_depth": 8, "mlp_units": 16}
    return argparse.Namespace(**cfg)


def _tiny_batch(num_actions, is_demo, B=2, T=8, size=32, seed=0):
    rng = np.random.RandomState(seed)
    action = np.zeros((B, T, num_actions), np.float32)
    for b in range(B):
        for t in range(1, T):
            action[b, t, rng.randint(num_actions)] = 1.0
    is_first = np.zeros((B, T), bool)
    is_first[:, 0] = True
    return {
        "image": rng.randint(0, 255, (B, T, size, size, 3)).astype(np.uint8),
        "mask": np.ones((B, T, num_actions), np.float32),
        "vector": rng.uniform(-1, 1, (B, T, 11)).astype(np.float32),
        "action": action,
        "logprob": np.zeros((B, T), np.float32),
        "reward": rng.randn(B, T).astype(np.float32),
        "discount": np.ones((B, T), np.float32),
        "is_first": is_first,
        "is_last": np.zeros((B, T), bool),
        "is_terminal": np.zeros((B, T), bool),
        "is_demo": np.full((B, T), float(is_demo), np.float32),
    }


def test_bc_loss_wiring(tmp_path):
    torch = pytest.importorskip("torch")
    torch.distributions.Distribution.set_default_validate_args(False)
    import gymnasium.spaces as spaces
    from dreamerv3 import tools
    from dreamerv3.dreamer import Dreamer

    A = 12
    config = _tiny_config(A)
    obs_space = spaces.Dict({
        "image": spaces.Box(0, 255, (32, 32, 3), dtype=np.uint8),
        "mask": spaces.Box(0, 1, (A,), dtype=np.float32),
        # Exercises the encoder/decoder MLP branch (mlp_keys: 'vector').
        "vector": spaces.Box(-1, 1, (11,), dtype=np.float32),
        "is_first": spaces.Box(0, 1, (), dtype=np.uint8),
        "is_last": spaces.Box(0, 1, (), dtype=np.uint8),
        "is_terminal": spaces.Box(0, 1, (), dtype=np.uint8),
    })
    act_space = spaces.Box(0, 1, (A,), dtype=np.float32)
    act_space.discrete = True
    act_space.n = A

    logger = tools.Logger(pathlib.Path(tmp_path), 0)
    agent = Dreamer(obs_space, act_space, config, logger, dataset=None)
    agent.requires_grad_(requires_grad=False)

    # Demo batch -> BC term active and finite.
    agent._train(_tiny_batch(A, is_demo=1.0))
    assert "bc_loss" in agent._metrics
    assert np.isfinite(agent._metrics["bc_loss"][0])
    assert agent._metrics["bc_scale"][0] == pytest.approx(1.0)

    # No demo steps in the batch -> no BC term.
    agent._metrics.clear()
    agent._train(_tiny_batch(A, is_demo=0.0))
    assert "bc_loss" not in agent._metrics

    # Fully decayed -> no BC term even on demo batches.
    agent._metrics.clear()
    agent._config.bc_decay = 100
    agent._step = 1000
    agent._train(_tiny_batch(A, is_demo=1.0))
    assert "bc_loss" not in agent._metrics

    # bc_floor keeps a permanent decayed anchor.
    agent._metrics.clear()
    agent._config.bc_floor = 0.5
    agent._train(_tiny_batch(A, is_demo=1.0))
    assert agent._metrics["bc_scale"][0] == pytest.approx(0.5)


def test_log_metrics_handles_scalars_and_empties(tmp_path):
    pytest.importorskip("torch")
    from types import SimpleNamespace
    from dreamerv3 import tools
    from dreamerv3.dreamer import Dreamer

    # __call__ stores lists for training metrics but a plain int for
    # update_count, and intermittent metrics leave empty lists behind.
    fake = SimpleNamespace(
        _metrics={"loss": [1.0, 3.0], "update_count": 7, "stale": []},
        _logger=tools.Logger(pathlib.Path(tmp_path), 0))
    Dreamer._log_metrics(fake)
    assert fake._metrics["loss"] == []
    assert fake._metrics["update_count"] == []
    assert fake._metrics["stale"] == []  # skipped, never logged as NaN
