"""Evaluate baselines (and optionally a trained Dreamer policy) on PCB test
point placement; all methods are scored on the same boards.
Run: python eval.py --episodes 5 --num_traces 10 [--freerouting] [--checkpoint ...]
"""

import argparse
import pathlib
import os
import time

os.environ["MUJOCO_GL"] = "osmesa"

import numpy as np

from envs.board import load_te_example, generate_candidate_grid, check_tp_spacing
from envs.routing import route_all_traces, validate_routing_constraints
# matplotlib / torch / dreamerv3 are imported lazily so the numeric A* eval runs without them.


def _board_key(board):
    """Cheap board fingerprint for result caching (id() alone could be reused
    after garbage collection)."""
    return (round(board.x_min, 3), round(board.x_max, 3),
            round(board.y_min, 3), round(board.y_max, 3),
            round(board.connector_x, 3), round(board.connector_y, 3),
            tuple((round(t.start_x, 3), round(t.start_y, 3))
                  for t in board.traces))


# (board, placement, router settings) -> result. Routing is deterministic, and
# the default fixed-board mode scores IDENTICAL placements every episode for
# the deterministic baselines -- without this, that work is redone per episode
# (30s+ per unroutable 20-trace placement at the quality budget).
_route_cache = {}


def evaluate_placement(board, placed_tps, use_freerouting=False, route_kw=None):
    """Route and score a placement (memoized; route_kw overrides the quality
    router budget, e.g. the --fast preset)."""
    key = (_board_key(board),
           tuple(None if p is None else (round(p[0], 6), round(p[1], 6))
                 for p in placed_tps),
           use_freerouting,
           tuple(sorted((route_kw or {}).items())))
    if key in _route_cache:
        return _route_cache[key]
    if use_freerouting:
        from envs.freerouting import route_with_freerouting
        paths, lengths, failures = route_with_freerouting(board, placed_tps)
    else:
        paths, lengths, failures = route_all_traces(
            board, placed_tps, **(route_kw or {}))

    validation = validate_routing_constraints(board, paths)
    finite = [l for l in lengths if l < float('inf')]
    spread = ((max(finite) - min(finite)) / np.mean(finite)
              if len(finite) > 1 else 0)
    result = {
        "placed": placed_tps, "paths": paths, "lengths": lengths,
        "failures": failures,
        "total_length": sum(finite) if finite else 0,
        # Equalization pads all traces to the max, so final length = n * max.
        "max_length": max(finite) if finite else 0,
        "spread": spread,
        "validation": validation,
    }
    _route_cache[key] = result
    return result


def run_random_baseline(boards, num_traces, use_freerouting=False, route_kw=None):
    """Random TP placement, one per trace in order."""
    results = []
    rng = np.random.RandomState(0)
    for board, candidates in boards:
        placed = []
        for i in range(num_traces):
            for idx in rng.permutation(len(candidates)):
                cx, cy = candidates[idx]
                if check_tp_spacing(placed, cx, cy):
                    placed.append((cx, cy))
                    break
            else:
                placed.append(tuple(candidates[rng.randint(len(candidates))]))
        results.append(evaluate_placement(board, placed, use_freerouting, route_kw))
    return results


def run_greedy_baseline(boards, num_traces, use_freerouting=False, route_kw=None):
    """For each trace, pick the closest valid candidate."""
    results = []
    for board, candidates in boards:
        placed = []
        for i in range(num_traces):
            trace = board.traces[i]
            dists = np.hypot(candidates[:, 0] - trace.start_x,
                             candidates[:, 1] - trace.start_y)
            for idx in np.argsort(dists):
                cx, cy = candidates[idx]
                if check_tp_spacing(placed, cx, cy):
                    placed.append((cx, cy))
                    break
            else:
                placed.append(tuple(candidates[np.argsort(dists)[0]]))
        results.append(evaluate_placement(board, placed, use_freerouting, route_kw))
    return results


def run_spread_baseline(boards, num_traces, use_freerouting=False, route_kw=None):
    """Place TPs far from connector, well-spread, then assign in order."""
    results = []
    for board, candidates in boards:
        cx_conn = board.connector_x + board.connector_w / 2
        cy_conn = board.connector_y + board.connector_h / 2
        dists = np.hypot(candidates[:, 0] - cx_conn, candidates[:, 1] - cy_conn)
        order = np.argsort(-dists)
        placed = []
        for idx in order:
            if len(placed) >= num_traces:
                break
            cx, cy = candidates[idx]
            if check_tp_spacing(placed, cx, cy):
                placed.append((cx, cy))
        while len(placed) < num_traces:
            before = len(placed)
            for idx in range(len(candidates)):
                if len(placed) >= num_traces:
                    break
                cx, cy = candidates[idx]
                if check_tp_spacing(placed, cx, cy):
                    placed.append((cx, cy))
            if len(placed) == before:      # board can't fit num_traces TPs
                break
        results.append(evaluate_placement(board, placed, use_freerouting, route_kw))
    return results


def run_planar_baseline(boards, num_traces, use_freerouting=False, route_kw=None):
    """Non-crossing by construction: match pins<->TPs in angular order around the connector."""
    results = []
    for board, candidates in boards:
        ccx = board.connector_x + board.connector_w / 2
        ccy = board.connector_y + board.connector_h / 2
        chosen = []
        for idx in np.argsort(-np.hypot(candidates[:, 0] - ccx,
                                        candidates[:, 1] - ccy)):
            if len(chosen) >= num_traces:
                break
            if check_tp_spacing(chosen, *candidates[idx]):
                chosen.append(tuple(candidates[idx]))
        tps = sorted(chosen, key=lambda p: np.arctan2(p[1] - ccy, p[0] - ccx))
        pins = sorted(range(num_traces),
                      key=lambda i: np.arctan2(board.traces[i].start_y - ccy,
                                               board.traces[i].start_x - ccx))
        placed = [None] * num_traces
        for k, i in enumerate(pins):
            if k < len(tps):
                placed[i] = tps[k]
        for i in range(num_traces):
            if placed[i] is None:
                for cx, cy in candidates:
                    if check_tp_spacing([q for q in placed if q], cx, cy):
                        placed[i] = (cx, cy)
                        break
        results.append(evaluate_placement(board, placed, use_freerouting, route_kw))
    return results


def run_smart_baseline(boards, num_traces, use_freerouting=False, route_kw=None):
    """No-training smart_placement: heuristics elected by trial routing (strongest classical baseline)."""
    from envs.board import smart_placement
    results = []
    placements = {}  # fixed-board mode: don't rerun the election per episode
    for board, _candidates in boards:
        bkey = _board_key(board)
        if bkey not in placements:
            placements[bkey] = smart_placement(board, num_traces)
        results.append(evaluate_placement(board, placements[bkey],
                                          use_freerouting, route_kw))
    return results


def run_dreamer_policy(checkpoint, boards, num_traces, board_seed=None,
                       configs=("defaults",), device="cpu",
                       use_freerouting=False, log_dir=None, route_kw=None,
                       board="te"):
    """Roll a trained Dreamer policy on the same boards and score with the same router as the baselines."""
    import torch
    import ruamel.yaml as yaml
    from dreamerv3 import tools as dv3_tools
    from dreamerv3.dreamer import Dreamer
    from envs.dreamer_wrapper import PCBDreamerEnv
    from envs import wrappers

    torch.distributions.Distribution.set_default_validate_args(False)

    # Same config recipe as train.py so the network matches the checkpoint.
    cfg_all = yaml.YAML(typ="safe").load(
        (pathlib.Path(__file__).parent / "configs.yaml").read_text())
    cfg = {}
    for name in configs:
        assert name in cfg_all, f"Config '{name}' not in {list(cfg_all.keys())}"
        cfg.update(cfg_all[name])
    if device == "cuda:0" and not torch.cuda.is_available():
        device = "cpu"
    cfg["device"] = device

    # Identical wrapper stack to train.py; board_seed=None pins the fixed board.
    if board == "canonical":
        # STATIC target: always the exact xlsx board (board_seed irrelevant).
        from envs.board import load_te_excel
        factory = lambda s: load_te_excel()
    else:
        factory = None if board_seed is not None else (
            lambda s: load_te_example(num_traces=num_traces))
    denv = PCBDreamerEnv(num_traces=num_traces, seed=board_seed or 0,
                         board_factory=factory)
    env = wrappers.OneHotAction(denv)
    env = wrappers.TimeLimit(env, num_traces)
    env = wrappers.SelectAction(env, key="action")
    env = wrappers.UUID(env)

    acts = env.action_space
    cfg["num_actions"] = acts.n if hasattr(acts, "n") else acts.shape[0]
    config = argparse.Namespace(**cfg)

    logger = dv3_tools.Logger(
        pathlib.Path(log_dir or "eval_results/dreamer_eval_logs"), 0)
    agent = Dreamer(env.observation_space, env.action_space,
                    config, logger, dataset=None).to(device)
    agent.requires_grad_(requires_grad=False)
    ckpt = torch.load(checkpoint, map_location=device)
    agent.load_state_dict(ckpt["agent_state_dict"])
    agent.eval()

    results = []
    for ep, (board, _cands) in enumerate(boards):
        denv._seed = (board_seed + ep) if board_seed is not None else 0
        obs = env.reset()
        agent_state, done = None, False
        while not done:
            batch = {k: np.stack([v]) for k, v in obs.items() if "log_" not in k}
            with torch.no_grad():
                out, agent_state = agent(batch, np.array([done]), agent_state,
                                         training=False)
            action = {"action": np.array(out["action"][0].detach().cpu())}
            obs, _reward, done, _info = env.step(action)
        results.append(evaluate_placement(
            denv._inner.board, list(denv._inner.placed_tps), use_freerouting,
            route_kw))
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--num_traces", type=int, default=10)
    parser.add_argument("--freerouting", action="store_true",
                        help="Use FreeRouting instead of A*")
    parser.add_argument("--no-plot", dest="plot", action="store_false",
                        help="Skip PNG plots (runs without matplotlib)")
    parser.add_argument("--board_seed", type=int, default=None,
                        help="Episode k evaluates the board from seed "
                             "board_seed+k (all methods). Default: the single "
                             "fixed TE board for every episode. Use a held-out "
                             "range (train.py holds out seed+1_000_000).")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to a Dreamer checkpoint (latest.pt); adds a "
                             "'Dreamer' row rolled on the same boards.")
    parser.add_argument("--configs", nargs="+", default=["defaults"],
                        help="configs.yaml sections for --checkpoint (must "
                             "match the trained model)")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device for --checkpoint inference")
    parser.add_argument("--fast", action="store_true",
                        help="Score with the training router budget "
                             "(n_starts=1, max_iters=12, repair_passes=1) "
                             "instead of the quality budget. 10-30x faster on "
                             "unroutable placements; failure counts can come "
                             "out slightly higher.")
    parser.add_argument("--board", type=str, default="te",
                        choices=["te", "canonical"],
                        help="'te' = synthetic TE-style board "
                             "(load_te_example); 'canonical' = the REAL "
                             "AutoLayout Example01 board from the xlsx, "
                             "always the exact static geometry.")
    args = parser.parse_args()

    router_name = "FreeRouting" if args.freerouting else "A*"
    route_kw = (dict(n_starts=1, max_iters=12, repair_passes=1)
                if args.fast else None)
    if args.fast:
        router_name += " fast"
    outdir = pathlib.Path("eval_results")
    outdir.mkdir(exist_ok=True)

    # Plotting is optional; metrics run without matplotlib.
    plot_board = None
    if args.plot:
        try:
            from envs.visualize import plot_board
        except ImportError:
            print("(matplotlib not installed; skipping plots, metrics only)")

    # One board (+ candidate grid) per episode, shared by every method.
    def make_board(ep):
        if args.board == "canonical":
            # STATIC target: always the exact xlsx board.
            from envs.board import load_te_excel
            board = load_te_excel()
        else:
            seed = None if args.board_seed is None else args.board_seed + ep
            board = load_te_example(num_traces=args.num_traces, seed=seed)
        candidates, real_count = generate_candidate_grid(board, resolution=6.5)
        return board, candidates[:real_count]

    boards = [make_board(ep) for ep in range(args.episodes)]
    board0 = boards[0][0]
    board_name = ("canonical AutoLayout board" if args.board == "canonical"
                  else "TE-style board")
    seed_note = (f"fixed {board_name}" if args.board_seed is None
                 else f"{board_name}, seeds {args.board_seed}.."
                      f"{args.board_seed + args.episodes - 1}")
    print(f"Board: {board0.width}x{board0.height}mm, {len(board0.traces)} traces, "
          f"{len(boards[0][1])} candidates, router={router_name}, {seed_note}")

    def print_results(name, results):
        for i, r in enumerate(results):
            v = r["validation"]
            t2t = (f"{v['trace_to_trace_min']:.2f}"
                   if v['trace_to_trace_min'] < float('inf') else "n/a")
            print(f"  Ep {i + 1}: failures={r['failures']}, "
                  f"crossings={v.get('crossings', 0)}, "
                  f"length={r['total_length']:.0f}mm, "
                  f"max={r['max_length']:.0f}mm, "
                  f"spread={r['spread']:.2f}, t2t={t2t}mm, "
                  f"pad_clr={v.get('tp_to_trace_min', float('inf')):.1f}mm")
            if plot_board is not None:
                plot_board(boards[i][0], test_points=r["placed"], paths=r["paths"],
                           candidates=boards[i][1],
                           title=f"{name} #{i + 1}: {r['failures']} fail, "
                                 f"{r['total_length']:.0f}mm",
                           filename=str(outdir / f"{name.lower()}_{i + 1}.png"))

    def timed(name, runner):
        """Run one method with a wall-clock readout (slow evals look hung
        otherwise -- an unroutable placement costs 30s+ at the quality budget)."""
        print(f"\n--- {name} Baseline ({router_name}) ---")
        t0 = time.perf_counter()
        results = runner(boards, args.num_traces, args.freerouting, route_kw)
        print(f"  [{time.perf_counter() - t0:.1f}s]")
        print_results(name, results)
        return results

    random_results = timed("Random", run_random_baseline)
    greedy_results = timed("Greedy", run_greedy_baseline)
    spread_results = timed("Spread", run_spread_baseline)
    planar_results = timed("Planar", run_planar_baseline)
    smart_results = timed("Smart", run_smart_baseline)

    tables = [("Random", random_results), ("Greedy", greedy_results),
              ("Spread", spread_results), ("Planar", planar_results),
              ("Smart", smart_results)]

    if args.checkpoint:
        print(f"\n--- Dreamer Policy ({router_name}) ---  ({args.checkpoint})")
        t0 = time.perf_counter()
        dreamer_results = run_dreamer_policy(
            args.checkpoint, boards, args.num_traces,
            board_seed=args.board_seed, configs=args.configs,
            device=args.device, use_freerouting=args.freerouting,
            log_dir=str(outdir / "dreamer_eval_logs"), route_kw=route_kw,
            board=args.board)
        print(f"  [{time.perf_counter() - t0:.1f}s]")
        print_results("Dreamer", dreamer_results)
        tables.append(("Dreamer", dreamer_results))

    print(f"\n--- Summary ({router_name}) ---")
    for name, results in tables:
        fails = [r["failures"] for r in results]
        lengths = [r["total_length"] for r in results]
        maxes = [r["max_length"] for r in results]
        spreads = [r["spread"] for r in results]
        crossings = [r.get("validation", {}).get("crossings", 0) for r in results]
        # Fully feasible = every trace routed AND no clearance violations / crossings.
        valid = sum(1 for r in results
                    if r["failures"] == 0
                    and r.get("validation", {}).get("all_valid", False))
        print(f"  {name:>10s}: failures={np.mean(fails):.1f}+/-{np.std(fails):.1f}, "
              f"crossings={np.mean(crossings):.1f}, "
              f"length={np.mean(lengths):.0f}mm, "
              f"max={np.mean(maxes):.0f}mm, "
              f"spread={np.mean(spreads):.2f}, "
              f"valid={valid}/{len(results)}")

    if plot_board is not None:
        print(f"\nPlots saved to {outdir}/")


if __name__ == "__main__":
    main()