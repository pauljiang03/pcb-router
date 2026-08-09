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
import pathlib

import numpy as np

from dreamerv3.tools import add_to_cache, convert, save_episodes
from envs.board import smart_placement


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


def collect_demos(directory, env_fn, episodes, verbose=True):
    """Ensure `episodes` expert episodes exist in `directory`.

    env_fn(seed_offset) -> a train.make_env-style wrapped env (OneHotAction ->
    TimeLimit -> SelectAction -> UUID over PCBDreamerEnv). Returns the number
    of episodes generated on this call.
    """
    directory = pathlib.Path(directory).expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    existing = len(list(directory.glob("*.npz")))
    todo = max(0, int(episodes) - existing)
    if todo == 0:
        if verbose and episodes:
            print(f"Demos: {existing} episodes already in {directory}")
        return 0
    if verbose:
        print(f"Demos: generating {todo} expert episodes "
              f"({existing} already in {directory})...")

    env = env_fn(existing)  # continue the board-seed stream on resume
    num_actions = env.action_space.shape[0]
    inner = env._inner  # underlying TPPlacementEnv (wrappers delegate reads)

    returns, failures = [], []
    for ep in range(todo):
        obs = env.reset()
        cache = {}
        first = {k: convert(v) for k, v in obs.items()}
        first["reward"] = 0.0
        first["discount"] = 1.0
        add_to_cache(cache, env.id, first)

        plan = smart_placement(inner.board, inner.num_traces)
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
        if verbose and ((ep + 1) % 10 == 0 or ep + 1 == todo):
            print(f"  demo {existing + ep + 1}/{episodes}: "
                  f"return={np.mean(returns):.2f} "
                  f"failures={np.mean(failures):.2f} (running mean)")
    try:
        env.close()
    except Exception:
        pass
    return todo


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
    args = parser.parse_args()

    from train import make_env, DEMO_SEED_OFFSET  # lazy: avoids a cycle

    def env_fn(offset):
        return make_env("demo", 0, args.seed + DEMO_SEED_OFFSET + offset,
                        args.num_traces, args.reward_mode, boards=args.boards)

    collect_demos(pathlib.Path(args.logdir).expanduser() / "demo_eps",
                  env_fn, args.episodes)


if __name__ == "__main__":
    main()
