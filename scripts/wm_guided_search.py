"""World-model-guided placement search on the static canonical board.

Observations are pure geometry (routing only enters the terminal reward), so
a full episode for any placement can be built without routing and scored by
the reward head in one batched forward. Phases: (1) fidelity, rank
correlation between surrogate and true returns; (2) guided arm, route only
the surrogate's top-k per iteration, counting router calls C; (3) blind arm,
same generator/acceptance/seed, routing every mutation up to blind_mult x C.
Output JSON is optimize_placement.py-compatible (train.py --demo_placement).

Without --checkpoint the model has RANDOM weights: plumbing smoke only.
"""
import argparse
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np

from envs.board import check_tp_spacing, smart_placement
from envs.routing import route_all_traces, equalize_lengths

# Training router budget, same as the env terminal step and the blind optimizer.
FAST = dict(n_starts=1, max_iters=12, repair_passes=1)


def _num_actions(env):
    # OneHotAction exposes a Box without .n (train.py uses the same fallback).
    acts = env.action_space
    return acts.n if hasattr(acts, "n") else acts.shape[0]


class ObsBuilder:
    """Builds the exact episode observation rows the training cache holds for
    an action-index sequence, without terminal routing. State is snapshotted
    and restored around every build; parity is pinned by tests/test_wm_search.py."""

    def __init__(self, env):
        self._env = env
        obs0 = env.reset()
        inner = env._inner
        self._n = inner.num_traces
        self._num_actions = _num_actions(env)
        self._row0 = {k: np.asarray(obs0[k]) for k in ("image", "mask", "vector")}
        self._reset_state = (list(inner.placed_tps), inner.current_trace,
                             inner.candidate_mask.copy())

    def build(self, seq):
        assert len(seq) == self._n, (len(seq), self._n)
        env, inner = self._env, self._env._inner
        saved = (list(inner.placed_tps), inner.current_trace,
                 inner.candidate_mask.copy())
        placed0, trace0, mask0 = self._reset_state
        inner.placed_tps = list(placed0)
        inner.current_trace = trace0
        inner.candidate_mask = mask0.copy()
        images = [self._row0["image"]]
        masks = [self._row0["mask"]]
        vectors = [self._row0["vector"]]
        for idx in seq:
            # TPPlacementEnv.step minus reward/terminal routing.
            inner.placed_tps.append(tuple(inner.candidates[idx]))
            inner.current_trace += 1
            inner._update_candidate_mask()
            images.append(inner._render_obs())
            masks.append(env._mask())
            vectors.append(env._vector())
        inner.placed_tps, inner.current_trace, inner.candidate_mask = saved
        rows = self._n + 1
        # Cache convention: action[t] is the one-hot that produced obs row t.
        action = np.zeros((rows, self._num_actions), np.float32)
        for t, idx in enumerate(seq):
            action[t + 1, idx] = 1.0
        is_first = np.zeros(rows, bool)
        is_first[0] = True
        is_terminal = np.zeros(rows, bool)
        is_terminal[-1] = True
        return {
            "image": np.stack(images).astype(np.uint8),
            "mask": np.stack(masks).astype(np.float32),
            "vector": np.stack(vectors).astype(np.float32),
            "action": action,
            "is_first": is_first,
            "is_terminal": is_terminal,
        }

    def batch(self, seqs):
        rows = [self.build(s) for s in seqs]
        return {k: np.stack([r[k] for r in rows]) for k in rows[0]}


def surrogate_scores(wm, builder, states, device, chunk=64):
    """Predicted episode returns: posterior on built obs, reward head summed."""
    import torch
    out = []
    for start in range(0, len(states), chunk):
        part = states[start:start + chunk]
        data = builder.batch(part)
        with torch.no_grad():
            obs = wm.preprocess(data)
            embed = wm.encoder(obs)
            post, _ = wm.dynamics.observe(embed, obs["action"], obs["is_first"])
            feat = wm.dynamics.get_feat(post)
            pred = wm.heads["reward"](feat).mode()
            # .mode() carries a trailing singleton dim.
            out.append(pred.reshape(len(part), -1).sum(-1).cpu().numpy())
    return np.concatenate(out)


def load_world_model(env, checkpoint, configs, device, log_dir):
    """Same loading recipe as eval.py:run_dreamer_policy."""
    import torch
    import ruamel.yaml as yaml
    from dreamerv3 import tools as dv3_tools
    from dreamerv3.dreamer import Dreamer

    torch.distributions.Distribution.set_default_validate_args(False)
    cfg_all = yaml.YAML(typ="safe").load(
        (pathlib.Path(__file__).resolve().parents[1] / "configs.yaml").read_text())
    cfg = {}
    for name in configs:
        assert name in cfg_all, f"Config '{name}' not in {list(cfg_all.keys())}"
        cfg.update(cfg_all[name])
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = "cpu"
    cfg["device"] = device
    cfg["num_actions"] = _num_actions(env)
    config = argparse.Namespace(**cfg)
    logger = dv3_tools.Logger(pathlib.Path(log_dir), 0)
    agent = Dreamer(env.observation_space, env.action_space, config, logger,
                    dataset=None).to(device)
    agent.requires_grad_(requires_grad=False)
    if checkpoint:
        ckpt = torch.load(checkpoint, map_location=device)
        agent.load_state_dict(ckpt["agent_state_dict"])
        print(f"loaded checkpoint: {checkpoint}")
    else:
        print("=" * 68)
        print("WARNING: NO CHECKPOINT -- the world model has RANDOM WEIGHTS.")
        print("All scores are meaningless; this run only smoke-tests plumbing.")
        print("=" * 68)
    agent.eval()
    return agent._wm, device


def snap_to_indices(coords, cand, real_count):
    """Nearest-candidate snap keeping indices distinct and spacing valid
    (smart coords are off-grid; the model only saw grid placements)."""
    idxs = []
    for (x, y) in coords:
        d = np.hypot(cand[:real_count, 0] - x, cand[:real_count, 1] - y)
        for j in np.argsort(d):
            j = int(j)
            if j in idxs:
                continue
            if check_tp_spacing([tuple(cand[k]) for k in idxs], *cand[j]):
                idxs.append(j)
                break
        else:
            raise RuntimeError(f"cannot snap TP {(x, y)} to a valid candidate")
    return idxs


def mutate_state(state, rng, cand, real_count):
    """One mutation in index space, mirroring optimize_placement.py:
    swap two / rotate three / move one to a candidate 0.1-30mm away."""
    n = len(state)
    trial = list(state)
    r = rng.rand()
    if r < 0.35 and n >= 2:
        i, j = rng.choice(n, 2, replace=False)
        trial[i], trial[j] = trial[j], trial[i]
    elif r < 0.5 and n >= 3:
        i, j, k = rng.choice(n, 3, replace=False)
        trial[i], trial[j], trial[k] = trial[j], trial[k], trial[i]
    else:
        i = rng.randint(n)
        bx, by = cand[trial[i]]
        d = np.hypot(cand[:real_count, 0] - bx, cand[:real_count, 1] - by)
        near = np.where((d > 0.1) & (d < 30.0))[0]
        if not len(near):
            return None
        j = int(rng.choice(near))
        others = [tuple(cand[t]) for m, t in enumerate(trial) if m != i]
        if not check_tp_spacing(others, *cand[j]):
            return None
        trial[i] = j
    return trial


def draw_mutation(state, rng, cand, real_count, tries=50):
    for _ in range(tries):
        m = mutate_state(state, rng, cand, real_count)
        if m is not None:
            return m
    return None


def coords_of(state, cand):
    return [tuple(cand[i]) for i in state]


def route_score(board, coords):
    """(failures, max_mm, total_mm) at the training budget."""
    _p, lengths, fails = route_all_traces(board, coords, **FAST)
    fin = [x for x in lengths if x < 1e9]
    return int(fails), (max(fin) if fin else 1e9), (sum(fin) if fin else 1e9)


def fully_matched(board, coords, n):
    """True iff routed clean and every trace equalizes to the padding target."""
    paths, _L, fails = route_all_traces(board, coords, **FAST)
    if fails:
        return False
    _eqp, _eqL, _t, m = equalize_lengths(board, paths, test_points=coords)
    return m == n


def _accept(trial_sc, best_sc):
    f, m, t = trial_sc
    _bf, bm, bt = best_sc
    return f == 0 and (m < bm - 1e-6 or (m < bm + 1e-6 and t < bt - 0.5))


def _ranks(v):
    v = np.asarray(v, dtype=float)
    order = np.argsort(v, kind="mergesort")
    ranks = np.empty(len(v))
    sv = v[order]
    i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and sv[j + 1] == sv[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def pearson(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a - a.mean()
    b = b - b.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 0 else float("nan")


def spearman(a, b):
    # scipy is deliberately not a dependency (see .github/workflows/tests.yml).
    return pearson(_ranks(a), _ranks(b))


def replay_true_return(env, seq):
    """True shaped return + routing outcome by stepping the real wrapped env."""
    num_actions = _num_actions(env)
    env.reset()
    total, done, info = 0.0, False, {}
    for idx in seq:
        a = np.zeros(num_actions, np.float32)
        a[idx] = 1.0
        _obs, r, done, info = env.step({"action": a})
        total += float(r)
    assert done, "sequence did not terminate the episode"
    lengths = info.get("trace_lengths") or []
    fin = [l for l in lengths if l < 1e9]
    return {"return": total,
            "fails": int(info.get("failures", len(seq))),
            "max_mm": (float(max(fin)) if fin else float("inf")),
            "total_mm": (float(sum(fin)) if fin else float("inf"))}


def fidelity_phase(args, env, wm, builder, device, seed_state, cand, real_count):
    """Rank correlation between surrogate and true returns at depths 1/2/4/8."""
    rng = np.random.RandomState(args.seed)
    depths = (1, 2, 4, 8)
    samples = []
    print(f"\n[fidelity] {args.fidelity} samples, depths {depths} ...", flush=True)
    t0 = time.monotonic()
    # Calibration anchor: the seed is the placement training saw most often.
    seed_pred = float(surrogate_scores(wm, builder, [list(seed_state)], device)[0])
    seed_true = replay_true_return(env, seed_state)
    print(f"[fidelity] seed: pred={seed_pred:.2f} true={seed_true['return']:.2f}",
          flush=True)
    states = []
    for k in range(args.fidelity):
        depth = depths[k % len(depths)]
        state = list(seed_state)
        for _ in range(depth):
            state = draw_mutation(state, rng, cand, real_count) or state
        states.append((depth, state))
    preds = surrogate_scores(wm, builder, [s for _d, s in states], device,
                             chunk=args.batch)
    for (depth, state), pred in zip(states, preds):
        true = replay_true_return(env, state)
        samples.append({"depth": depth, "pred": float(pred), **true})
    pred_v = [s["pred"] for s in samples]
    true_v = [s["return"] for s in samples]
    res = {
        "n": len(samples),
        "spearman_return": spearman(pred_v, true_v),
        "pearson_return": pearson(pred_v, true_v),
        "seed_pred": seed_pred,
        "seed_true_return": seed_true["return"],
        "router_calls": len(samples) + 1,
        "samples": samples,
    }
    # Also rank against the search objective (-max) on the routable subset.
    ok = [s for s in samples if s["fails"] == 0]
    res["n_zero_fail"] = len(ok)
    if len(ok) >= 3:
        res["spearman_neg_max_zero_fail"] = spearman(
            [s["pred"] for s in ok], [-s["max_mm"] for s in ok])
    else:
        res["spearman_neg_max_zero_fail"] = None
    print(f"[fidelity] rho(return)={res['spearman_return']:.3f} "
          f"pearson={res['pearson_return']:.3f} "
          f"rho(-max | 0 fails, n={len(ok)})="
          f"{res['spearman_neg_max_zero_fail']} "
          f"[{time.monotonic() - t0:.0f}s]", flush=True)
    return res


def guided_arm(args, wm, builder, device, board, cand, real_count,
               seed_state, seed_sc, seed_ok):
    """Hill climb routing only the surrogate's top-k per iteration."""
    n = len(seed_state)
    rng = np.random.RandomState(args.seed + 1)
    best, bsc = list(seed_state), seed_sc
    best_matched = (list(best), bsc[1], bsc[2]) if seed_ok else None
    calls = matched_checks = iters = hits = scored = 0
    curve = [[0, bsc[1], bsc[2]]]
    t_start = time.monotonic()
    t_end = t_start + args.minutes * 60
    print(f"\n[guided] budget {args.minutes} min, batch {args.batch}, "
          f"topk {args.topk}", flush=True)
    while time.monotonic() < t_end:
        iters += 1
        trials = []
        for _ in range(args.batch * 20):
            m = mutate_state(best, rng, cand, real_count)
            if m is not None:
                trials.append(m)
                if len(trials) == args.batch:
                    break
        if not trials:
            break
        scores = surrogate_scores(wm, builder, trials, device, chunk=args.batch)
        scored += len(trials)
        for oi in np.argsort(-scores)[:args.topk]:
            trial = trials[int(oi)]
            sc = route_score(board, coords_of(trial, cand))
            calls += 1
            if _accept(sc, bsc):
                improved_max = sc[1] < bsc[1] - 1e-6
                best, bsc = list(trial), sc
                hits += 1
                note = ""
                if improved_max:  # gate max milestones on equalization fitting
                    matched_checks += 1
                    if fully_matched(board, coords_of(best, cand), n):
                        best_matched = (list(best), bsc[1], bsc[2])
                    else:
                        note = "  [not equalizable -- kept climbing, not saved]"
                curve.append([calls, bsc[1], bsc[2]])
                print(f"  guided: max={sc[1]:.1f} total={sc[2]:.0f} "
                      f"(call {calls}, iter {iters}){note}", flush=True)
    stats = {"router_calls": calls, "iterations": iters,
             "surrogate_scored": scored,
             "accepted": hits,
             "hit_rate": (hits / max(calls, 1)),
             "matched_checks": matched_checks,
             "seconds": round(time.monotonic() - t_start, 1),
             "best_max_mm": bsc[1], "best_total_mm": bsc[2], "curve": curve}
    return best, bsc, best_matched, stats


def blind_arm(args, board, cand, real_count, seed_state, seed_sc, seed_ok,
              call_cap, minutes_cap):
    """Same generator/acceptance/seed; every mutation routed, up to call_cap."""
    n = len(seed_state)
    rng = np.random.RandomState(args.seed + 2)
    best, bsc = list(seed_state), seed_sc
    best_matched = (list(best), bsc[1], bsc[2]) if seed_ok else None
    calls = matched_checks = hits = 0
    curve = [[0, bsc[1], bsc[2]]]
    t_start = time.monotonic()
    t_end = t_start + minutes_cap * 60
    print(f"\n[blind] cap {call_cap} router calls or {minutes_cap:.1f} min",
          flush=True)
    while calls < call_cap and time.monotonic() < t_end:
        trial = draw_mutation(best, rng, cand, real_count)
        if trial is None:
            break
        sc = route_score(board, coords_of(trial, cand))
        calls += 1
        if _accept(sc, bsc):
            improved_max = sc[1] < bsc[1] - 1e-6
            best, bsc = list(trial), sc
            hits += 1
            note = ""
            if improved_max:
                matched_checks += 1
                if fully_matched(board, coords_of(best, cand), n):
                    best_matched = (list(best), bsc[1], bsc[2])
                else:
                    note = "  [not equalizable -- kept climbing, not saved]"
            curve.append([calls, bsc[1], bsc[2]])
            print(f"  blind: max={sc[1]:.1f} total={sc[2]:.0f} "
                  f"(call {calls}){note}", flush=True)
    stats = {"router_calls": calls, "call_cap": call_cap, "accepted": hits,
             "matched_checks": matched_checks,
             "seconds": round(time.monotonic() - t_start, 1),
             "best_max_mm": bsc[1], "best_total_mm": bsc[2], "curve": curve}
    return best, bsc, best_matched, stats


def build_env(num_traces, seed):
    """reward_mode/shaping must match the checkpoint's run; make_env's default
    reward_mode is layer_aware, NOT what the canonical run trained with."""
    from train import make_env
    return make_env("wm_search", 0, seed=seed, num_traces=num_traces,
                    reward_mode="single_layer", boards="canonical",
                    shaping="potential")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default=None,
                    help="latest.pt from the static canonical run; omit for a "
                         "random-weights plumbing smoke test")
    ap.add_argument("--configs", nargs="+", default=["defaults", "colab_a100"],
                    help="configs.yaml sections (must match the checkpoint)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--minutes", type=float, default=10.0,
                    help="guided-arm search budget")
    ap.add_argument("--batch", type=int, default=48,
                    help="mutations scored by the surrogate per iteration")
    ap.add_argument("--topk", type=int, default=2,
                    help="real router calls per iteration")
    ap.add_argument("--fidelity", type=int, default=32,
                    help="fidelity samples (0 skips the phase)")
    ap.add_argument("--blind_mult", type=float, default=3.0,
                    help="blind arm runs to blind_mult x guided router calls")
    ap.add_argument("--blind_minutes", type=float, default=None,
                    help="blind-arm wall-clock cap (default 3 x --minutes)")
    ap.add_argument("--num_traces", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="eval_results/wm_search.json")
    args = ap.parse_args()

    if args.checkpoint and not pathlib.Path(args.checkpoint).exists():
        sys.exit(f"checkpoint not found: {args.checkpoint}")
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    env = build_env(args.num_traces, args.seed)
    builder = ObsBuilder(env)
    inner = env._inner
    board, cand, real_count = inner.board, inner.candidates, inner._real_count
    n = inner.num_traces
    print(f"board {board.width:.0f}x{board.height:.0f}mm, {n} traces, "
          f"{real_count} real candidates")

    wm, device = load_world_model(env, args.checkpoint, args.configs,
                                  args.device, out.parent / "wm_search_logs")

    seed_state = snap_to_indices(smart_placement(board, n), cand, real_count)
    seed_sc = route_score(board, coords_of(seed_state, cand))
    seed_ok = fully_matched(board, coords_of(seed_state, cand), n)
    print(f"seed (smart, grid-snapped): fails={seed_sc[0]} max={seed_sc[1]:.1f} "
          f"total={seed_sc[2]:.0f} matched={seed_ok}")

    fid = None
    if args.fidelity > 0:
        fid = fidelity_phase(args, env, wm, builder, device, seed_state,
                             cand, real_count)

    g_best, g_sc, g_matched, g_stats = guided_arm(
        args, wm, builder, device, board, cand, real_count,
        seed_state, seed_sc, seed_ok)

    call_cap = max(1, int(np.ceil(args.blind_mult * max(g_stats["router_calls"], 1))))
    blind_minutes = (args.blind_minutes if args.blind_minutes is not None
                     else 3.0 * args.minutes)
    b_best, b_sc, b_matched, b_stats = blind_arm(
        args, board, cand, real_count, seed_state, seed_sc, seed_ok,
        call_cap, blind_minutes)

    C = g_stats["router_calls"]
    blind_at_C = next((row for row in reversed(b_stats["curve"])
                       if row[0] <= C), b_stats["curve"][0])
    # 0 = the shared seed already matched guided's final max.
    match_calls = next((row[0] for row in b_stats["curve"]
                        if row[1] < g_sc[1] + 1e-6), None)
    b_stats["best_at_equal_calls_max_mm"] = blind_at_C[1]
    b_stats["best_at_equal_calls_total_mm"] = blind_at_C[2]
    b_stats["calls_to_match_guided"] = match_calls

    # Saved answer must equalize (same fallback as optimize_placement.py).
    final, final_sc = g_best, g_sc
    if not fully_matched(board, coords_of(final, cand), n):
        if g_matched is not None:
            print(f"guided final not equalizable; reverting to last matched "
                  f"milestone (max={g_matched[1]:.1f})")
            final = list(g_matched[0])
        else:
            print("WARNING: no fully-matched guided placement; saving the "
                  "best routed one anyway")

    # Quality-budget verify, same as eval.py scoring.
    coords = coords_of(final, cand)
    paths, lengths, fails = route_all_traces(board, coords)
    fin = [x for x in lengths if x < 1e9]
    qmax, qtot = (max(fin), sum(fin)) if fin else (1e9, 1e9)
    matched = 0
    if fails == 0:
        _eqp, _eqL, _t, matched = equalize_lengths(board, paths,
                                                   test_points=coords)

    print("\n================ results ================")
    if not args.checkpoint:
        print("(RANDOM WEIGHTS -- numbers below are plumbing only)")
    if fid:
        print(f"fidelity: rho(return)={fid['spearman_return']:.3f}  "
              f"pearson={fid['pearson_return']:.3f}  "
              f"rho(-max|0fail)={fid['spearman_neg_max_zero_fail']}  "
              f"seed pred={fid['seed_pred']:.2f}/true={fid['seed_true_return']:.2f}")
    print(f"guided : C={C} calls in {g_stats['seconds']:.0f}s, "
          f"best max={g_sc[1]:.1f} total={g_sc[2]:.0f}, "
          f"hit_rate={g_stats['hit_rate']:.2f}")
    print(f"blind  : at C calls max={blind_at_C[1]:.1f}; "
          f"ran to {b_stats['router_calls']} calls in {b_stats['seconds']:.0f}s "
          f"(cap {call_cap}), final max={b_sc[1]:.1f}")
    print(f"blind calls to match guided's max: "
          f"{match_calls if match_calls is not None else 'not within cap'}")
    print(f"final (quality verify): fails={fails} max={qmax:.1f} "
          f"total={qtot:.0f} matched={matched}/{n}")

    payload = {
        "board": "canonical", "num_traces": n,
        "placement": [[float(x), float(y)] for x, y in coords],
        "failures": int(fails), "max_mm": float(qmax),
        "total_mm": float(qtot), "matched": int(matched),
        "wm_guided_search": {
            "checkpoint": args.checkpoint,
            "random_weights": not bool(args.checkpoint),
            "settings": {k: getattr(args, k) for k in
                         ("configs", "minutes", "batch", "topk", "fidelity",
                          "blind_mult", "seed", "num_traces")},
            "seed_placement": {
                "state": [int(i) for i in seed_state],
                "fails": seed_sc[0], "max_mm": seed_sc[1],
                "total_mm": seed_sc[2], "matched": bool(seed_ok)},
            "fidelity": fid,
            "guided": g_stats,
            "blind": b_stats,
        },
    }
    json.dump(payload, out.open("w"), indent=1)
    print("saved ->", out)


if __name__ == "__main__":
    main()
