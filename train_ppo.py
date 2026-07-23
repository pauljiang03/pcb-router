"""PPO model-free baseline on the fixed TE board (train.py's Dreamer default
also mixes challenge boards and random orientations).
Run: python train_ppo.py --steps 200000 --num_traces 8
"""

import argparse
import pathlib

import gymnasium as gym
import numpy as np

from envs.pcb_env import TPPlacementEnv


class SeededBoards(gym.Wrapper):
    """Regenerate the board each episode from an incrementing seed, as in Dreamer training."""

    def __init__(self, env, base_seed: int):
        super().__init__(env)
        self._next_seed = base_seed

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=self._next_seed, options=options)
        self._next_seed += 1
        return obs, info


def make_env(num_traces, base_seed, reward_mode):
    def _thunk():
        env = TPPlacementEnv(
            num_traces=num_traces, seed=base_seed, reward_mode=reward_mode,
            route_n_starts=1, route_max_iters=12,   # fast preset, as in training
        )
        return SeededBoards(env, base_seed)
    return _thunk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=200_000)
    ap.add_argument("--num_traces", type=int, default=8)
    ap.add_argument("--envs", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--logdir", type=str, default="./logdir/ppo")
    ap.add_argument("--reward_mode", type=str, default="layer_aware",
                    choices=["layer_aware", "single_layer"])
    ap.add_argument("--check_only", action="store_true",
                    help="Run SB3's env checker + build the model, no training")
    args = ap.parse_args()

    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.env_checker import check_env
        from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
        from stable_baselines3.common.callbacks import EvalCallback
        from stable_baselines3.common.monitor import Monitor
    except ImportError:
        raise SystemExit("stable-baselines3 not installed: pip install stable-baselines3")

    logdir = pathlib.Path(args.logdir).expanduser()
    logdir.mkdir(parents=True, exist_ok=True)

    check_env(make_env(args.num_traces, args.seed, args.reward_mode)(), warn=True)

    vec_cls = SubprocVecEnv if args.envs > 1 else DummyVecEnv
    venv = vec_cls([make_env(args.num_traces, args.seed + 10_000 * i, args.reward_mode)
                    for i in range(args.envs)])
    # Held-out eval boards: disjoint seed range, mirrors train.py.
    eval_env = DummyVecEnv([lambda: Monitor(
        make_env(args.num_traces, args.seed + 1_000_000, args.reward_mode)())])

    model = PPO(
        "CnnPolicy", venv, verbose=1, seed=args.seed,
        n_steps=max(64, 2 * args.num_traces), batch_size=64,
        tensorboard_log=str(logdir),
    )
    if args.check_only:
        print("--check_only: skipping training.")
        return

    cb = EvalCallback(eval_env, best_model_save_path=str(logdir),
                      log_path=str(logdir), eval_freq=5_000,
                      n_eval_episodes=10, deterministic=True)
    model.learn(total_timesteps=args.steps, callback=cb)
    model.save(logdir / "ppo_final")
    print(f"Saved {logdir}/ppo_final.zip")


if __name__ == "__main__":
    main()
