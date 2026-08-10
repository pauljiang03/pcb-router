"""Expert demonstration collection for cold-start training.

Rolls the no-training smart_placement baseline (envs.board) through the SAME
wrapped environment stack train.py uses, so the saved episodes are
byte-compatible with the ones tools.simulate writes: identical obs keys,
one-hot actions, rewards from the same reward function and routing budget.
Episodes land in <logdir>/demo_eps and are sampled into training batches at a
fixed fraction (train.py --demo_fraction) and imitated by the decayed
behavior-cloning loss (configs.yaml bc_scale/bc_decay).

Collection is resumable: existing episodes in the directory are counted and
only the remainder is generated, continuing the board-seed stream.

Standalone use (defaults mirror train.py):
    python demos.py --logdir ./logdir/pcb --episodes 200 --num_traces 8
"""

import argparse
import multiprocessing as mp
import pathlib
import signal
import sys
import time

import numpy as np

from dreamerv3.tools import add_to_cache, convert, save_episodes
from envs.board import smart_placement


def _expert_plan(inner, budget_s=240):
    """smart_placement under a hard wall-clock cap. Repair inside
    smart_placement is already time-capped; this SIGALRM guard is the
    backstop for any other pathological board so one episode can never
    wedge a whole demo run. On expiry, fall back to the instant
    no-election heuristic."""
    if not hasattr(signal, "SIGALRM"):
        return smart_placement(inner.board, inner.num_traces)

    def _timeout(signum, frame):
        raise TimeoutError

    old = signal.signal(signal.SIGALRM, _timeout)
    signal.alarm(int(budget_s))
    try:
        return smart_placement(inner.board, inner.num_traces)
    except TimeoutError:
        print(f"  demo board exceeded the {budget_s}s planning budget; "
              f"using no-election placement", flush=True)
        return smart_placement(inner.board, inner.num_traces, elect=False)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def _expert_action(inner, target):
    """Nearest still-valid candidate index to the expert TP `target`.

    smart_placement picks from the same 6.5 mm candidate grid the action space
    is built on, so this is normally an exact hit; nearest-valid also absorbs
    repair_placement drift and (target=None) missing assignments.
    """
    valid = np.where(inner.candidate_mask[:inner._real_count])[0]
    pool = valid if len(valid) else np.arange(inner._real_count)
    cands = inner.candidates
    if target is None:
        # No expert assignment for this net: keep options open by taking the
        # valid candidate farthest from the already-placed TPs.
        if not inner.placed_tps:
            return int(pool[len(pool) // 2])
        placed = np.array(inner.placed_tps)
        d = np.min(
            (cands[pool, None, 0] - placed[None, :, 0]) ** 2
            + (cands[pool, None, 1] - placed[None, :, 1]) ** 2,
            axis=1,
        )
        return int(pool[int(np.argmax(d))])
    d = (cands[pool, 0] - target[0]) ** 2 + (cands[pool, 1] - target[1]) ** 2
    return int(pool[int(np.argmin(d))])


def _collect_range(directory, env_fn, start_offset, count, verbose=True):
    """Roll `count` expert episodes at seed offsets [start_offset, +count)."""
    directory = pathlib.Path(directory)
    env = env_fn(start_offset)  # env's board seed advances once per reset
    num_actions = env.action_space.shape[0]
    inner = env._inner  # underlying TPPlacementEnv (wrappers delegate reads)

    returns, failures = [], []
    for ep in range(count):
        obs = env.reset()
        cache = {}
        first = {k: convert(v) for k, v in obs.items()}
        first["reward"] = 0.0
        first["discount"] = 1.0
        add_to_cache(cache, env.id, first)

        plan = _expert_plan(inner)
        done, step_i, ep_return = False, 0, 0.0
        while not done:
            target = plan[step_i] if step_i < len(plan) else None
            idx = _expert_action(inner, target)
            action = np.zeros(num_actions, dtype=np.float32)
            action[idx] = 1.0
            obs, reward, done, info = env.step({"action": action})
            transition = {k: convert(v) for k, v in obs.items()}
            transition["action"] = action
            transition["logprob"] = np.float32(0.0)  # deterministic expert
            transition["reward"] = reward
            transition["discount"] = info.get(
                "discount", np.array(1 - float(done)))
            add_to_cache(cache, env.id, transition)
            ep_return += float(reward)
            step_i += 1

        save_episodes(directory, {env.id: cache[env.id]})
        returns.append(ep_return)
        failures.append(int(info.get("failures", 0)))
        if verbose and ((ep + 1) % 10 == 0 or ep + 1 == count):
            print(f"  demos[{start_offset}..{start_offset + count - 1}]: "
                  f"{ep + 1}/{count} return={np.mean(returns):.2f} "
                  f"failures={np.mean(failures):.2f} (running mean)",
                  flush=True)
    try:
        env.close()
    except Exception:
        pass
    return count


def collect_demos(directory, env_fn, episodes, verbose=True, workers=1):
    """Ensure `episodes` expert episodes exist in `directory`.

    env_fn(seed_offset) -> a train.make_env-style wrapped env (OneHotAction ->
    TimeLimit -> SelectAction -> UUID over PCBDreamerEnv). Returns the number
    of episodes generated on this call.

    workers > 1 forks that many processes over contiguous seed-offset slices
    (episodes are independent; ~30s each on hard 20-trace moat boards, so
    serial generation of 200 takes over an hour while 8 workers need ~15
    min). Fork-only: on non-Linux platforms this falls back to serial.
    """
    directory = pathlib.Path(directory).expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    existing = len(list(directory.glob("*.npz")))
    todo = max(0, int(episodes) - existing)
    if todo == 0:
        if verbose and episodes:
            print(f"Demos: {existing} episodes already in {directory}")
        return 0
    workers = max(1, min(int(workers or 1), todo))
    if sys.platform != "linux":
        workers = 1  # closures don't survive spawn; fork is Linux-only here
    if verbose:
        print(f"Demos: generating {todo} expert episodes "
              f"({existing} already in {directory}, "
              f"{workers} worker{'s' if workers > 1 else ''})...")

    if workers == 1:
        return _collect_range(directory, env_fn, existing, todo, verbose)

    ctx = mp.get_context("fork")  # children only run numpy/env code, no torch
    sizes = [todo // workers + (1 if i < todo % workers else 0)
             for i in range(workers)]
    procs, lo = [], 0
    for n in sizes:
        if n == 0:
            continue
        procs.append(ctx.Process(
            target=_collect_range,
            args=(directory, env_fn, existing + lo, n, False)))
        procs[-1].start()
        lo += n
    # Workers are silent; the parent reports GLOBAL progress (a single
    # worker's slice count reads as 8x slower than reality).
    t_last = time.monotonic()
    while any(p.is_alive() for p in procs):
        time.sleep(2)
        if verbose and time.monotonic() - t_last >= 30:
            t_last = time.monotonic()
            done_now = len(list(directory.glob("*.npz"))) - existing
            print(f"  demos: {done_now}/{todo} generated "
                  f"({len(procs)} workers)", flush=True)
    for p in procs:
        p.join()
    made = len(list(directory.glob("*.npz"))) - existing
    if any(p.exitcode != 0 for p in procs) or made < todo:
        raise RuntimeError(
            f"demo workers produced {made}/{todo} episodes "
            f"(exit codes {[p.exitcode for p in procs]})")
    if verbose:
        # Workers are silent in the parallel path, so read the stats back:
        # the mean return here is the BASELINE ANCHOR eval must reach.
        rets, clean, total = [], 0, 0
        for f in directory.glob("*.npz"):
            ep = np.load(f)
            rets.append(float(ep["reward"].sum()))
            total += 1
            if "log_routable" in ep:
                clean += float(ep["log_routable"].sum()) == 10.0
        print(f"Demos: {made} episodes from {len(procs)} workers -> {directory}")
        print(f"  demo anchor: return mean {np.mean(rets):.1f} "
              f"(min {np.min(rets):.1f}, max {np.max(rets):.1f}); "
              f"{clean}/{total} routed 20/20 planar")
    return made


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", type=str, default="./logdir/pcb")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_traces", type=int, default=8)
    parser.add_argument("--reward_mode", type=str, default="layer_aware",
                        choices=["layer_aware", "single_layer"])
    parser.add_argument("--boards", type=str, default="mixed",
                        choices=["mixed", "central"])
    parser.add_argument("--workers", type=int, default=None,
                        help="Parallel generator processes "
                             "(default: min(8, cpus-2); Linux only)")
    args = parser.parse_args()

    from train import make_env, DEMO_SEED_OFFSET  # lazy: avoids a cycle

    def env_fn(offset):
        return make_env("demo", 0, args.seed + DEMO_SEED_OFFSET + offset,
                        args.num_traces, args.reward_mode, boards=args.boards)

    import os
    workers = args.workers or max(1, min(8, (os.cpu_count() or 2) - 2))
    collect_demos(pathlib.Path(args.logdir).expanduser() / "demo_eps",
                  env_fn, args.episodes, workers=workers)


if __name__ == "__main__":
    main()
