"""Train DreamerV3 on PCB Test Point Placement.
Run: python train.py --configs defaults [debug]
"""

import argparse
import functools
import os
import pathlib
import sys

os.environ["MUJOCO_GL"] = "osmesa"

import numpy as np
import ruamel.yaml as yaml
import torch
from torch import distributions as torchd

# Prevent PyTorch distribution validation errors on discrete actions
torch.distributions.Distribution.set_default_validate_args(False)

from dreamerv3 import exploration as expl
from dreamerv3 import models
from dreamerv3 import tools
from envs import wrappers
from envs.dreamer_wrapper import PCBDreamerEnv
from dreamerv3.parallel import Parallel, Damy
from dreamerv3.dreamer import Dreamer

to_np = lambda x: x.detach().cpu().numpy()


def mixed_board_factory(num_traces):
    """Per-episode board sampler: 50% TE board / 50% moat challenge, random orientation, deterministic per seed."""
    from envs.board import load_te_example, make_challenge, ChallengeSpec, rotate_board

    def factory(seed):
        rng = np.random.RandomState(seed)
        if rng.rand() < 0.5:
            board = load_te_example(num_traces=num_traces, seed=seed)
        else:
            spec = ChallengeSpec(
                board_size=float(rng.choice([100.0, 120.0, 140.0, 160.0])),
                num_traces=num_traces,
                n_gaps=int(rng.choice([2, 3, 4])),
                seed=seed,
            )
            board = make_challenge(spec)[0]  # board only; the agent places pads
        return rotate_board(board, int(rng.randint(4)), bool(rng.randint(2)))

    return factory


def make_env(mode, env_id, seed=0, num_traces=8, reward_mode="layer_aware",
             route_n_starts=1, route_max_iters=12, boards="mixed"):
    factory = mixed_board_factory(num_traces) if boards == "mixed" else None
    # Space workers 10k seeds apart; +env_id would replay ~the same board stream.
    env = PCBDreamerEnv(num_traces=num_traces, seed=seed + 10_000 * env_id,
                        reward_mode=reward_mode,
                        route_n_starts=route_n_starts,
                        route_max_iters=route_max_iters,
                        board_factory=factory)
    env = wrappers.OneHotAction(env)
    env = wrappers.TimeLimit(env, num_traces)
    env = wrappers.SelectAction(env, key="action")
    env = wrappers.UUID(env)
    return env


def count_steps(folder):
    return sum(int(str(n).split("-")[-1][:-4]) - 1 for n in folder.glob("*.npz"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", default=["defaults"])
    parser.add_argument("--logdir", type=str, default="./logdir/pcb")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_traces", type=int, default=8)
    parser.add_argument("--envs", type=int, default=None,
                        help="Number of parallel env workers (overrides config.envs). "
                             ">1 overlaps CPU routing with GPU training.")
    parser.add_argument("--parallel", action="store_true",
                        help="Run envs in separate processes (use with --envs>1).")
    parser.add_argument("--reward_mode", type=str, default="layer_aware",
                        choices=["layer_aware", "single_layer"])
    parser.add_argument("--boards", type=str, default="mixed",
                        choices=["mixed", "central"],
                        help="Training board distribution: 'mixed' samples the "
                             "central TE board and parametric moat challenge "
                             "boards 50/50; 'central' uses only the TE board.")
    args = parser.parse_args()

    config_path = pathlib.Path(__file__).parent / "configs.yaml"
    configs = yaml.YAML(typ="safe").load(config_path.read_text())
    config = {}
    for name in args.configs:
        assert name in configs, f"Config '{name}' not found in {list(configs.keys())}"
        config.update(configs[name])

    config["logdir"] = args.logdir
    config["seed"] = args.seed
    if args.device:
        config["device"] = args.device
    if config["device"] == "cuda:0" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        config["device"] = "cpu"

    if args.envs is not None:
        config["envs"] = args.envs
    if args.parallel:
        config["parallel"] = True

    config["time_limit"] = args.num_traces

    config = argparse.Namespace(**config)

    tools.set_seed_everywhere(config.seed)
    logdir = pathlib.Path(config.logdir).expanduser()
    config.traindir = config.traindir or logdir / "train_eps"
    config.evaldir = config.evaldir or logdir / "eval_eps"
    config.steps //= config.action_repeat
    config.eval_every //= config.action_repeat
    config.log_every //= config.action_repeat
    config.time_limit //= config.action_repeat

    print(f"logdir={logdir} device={config.device} "
          f"steps={int(config.steps)} traces={args.num_traces}")

    logdir.mkdir(parents=True, exist_ok=True)
    config.traindir.mkdir(parents=True, exist_ok=True)
    config.evaldir.mkdir(parents=True, exist_ok=True)

    step = count_steps(config.traindir)
    logger = tools.Logger(logdir, config.action_repeat * step)

    print(f"Creating {config.envs} environment(s) "
          f"({'parallel processes' if config.parallel else 'in-process'}, "
          f"reward_mode={args.reward_mode})...")

    def wrap_env(mode, env_id, base_seed):
        # Worker processes let CPU routing in env.step overlap GPU training.
        env = make_env(mode, env_id, base_seed, args.num_traces, args.reward_mode,
                       boards=args.boards)
        return Parallel(env, "process") if config.parallel else Damy(env)

    train_envs = [wrap_env("train", i, config.seed) for i in range(config.envs)]
    # Held-out eval boards: seed range disjoint from train's.
    eval_seed = config.seed + 1_000_000
    eval_envs = [wrap_env("eval", i, eval_seed) for i in range(config.envs)]

    acts = train_envs[0].action_space
    config.num_actions = acts.n if hasattr(acts, "n") else acts.shape[0]
    print(f"Action space: {config.num_actions} candidates")

    train_eps = tools.load_episodes(config.traindir, limit=config.dataset_size)
    eval_eps = tools.load_episodes(config.evaldir, limit=1)

    state = None
    prefill = max(0, config.prefill - count_steps(config.traindir))
    if prefill > 0:
        print(f"Prefilling ({prefill} steps)...")
        random_actor = tools.OneHotDist(
            torch.zeros(config.num_actions).repeat(config.envs, 1)
        )
        def random_agent(o, d, s):
            action = random_actor.sample()
            return {"action": action, "logprob": random_actor.log_prob(action)}, None

        state = tools.simulate(
            random_agent, train_envs, train_eps, config.traindir,
            logger, limit=config.dataset_size, steps=prefill,
        )
        logger.step += prefill * config.action_repeat
    dataset = tools.from_generator(
        tools.sample_episodes(train_eps, config.batch_length), config.batch_size
    )
    eval_dataset = tools.from_generator(
        tools.sample_episodes(eval_eps, config.batch_length), config.batch_size
    )

    agent = Dreamer(
        train_envs[0].observation_space,
        train_envs[0].action_space,
        config, logger, dataset,
    ).to(config.device)
    agent.requires_grad_(requires_grad=False)

    if (logdir / "latest.pt").exists():
        print("Resuming from checkpoint...")
        ckpt = torch.load(logdir / "latest.pt", map_location=config.device)
        agent.load_state_dict(ckpt["agent_state_dict"])
        tools.recursively_load_optim_state_dict(agent, ckpt["optims_state_dict"])
        agent._should_pretrain._once = False

    while agent._step < config.steps + config.eval_every:
        logger.write()

        if config.eval_episode_num > 0:
            print(f"\n[Step {agent._step}] Eval...")
            tools.simulate(
                functools.partial(agent, training=False),
                eval_envs, eval_eps, config.evaldir,
                logger, is_eval=True, episodes=config.eval_episode_num,
            )
            if config.video_pred_log:
                try:
                    video_pred = agent._wm.video_pred(next(eval_dataset))
                    logger.video("eval_openl", to_np(video_pred))
                except StopIteration:
                    pass

        print(f"[Step {agent._step}] Training...")
        state = tools.simulate(
            agent, train_envs, train_eps, config.traindir,
            logger, limit=config.dataset_size,
            steps=config.eval_every, state=state,
        )

        torch.save({
            "agent_state_dict": agent.state_dict(),
            "optims_state_dict": tools.recursively_collect_optim_state_dict(agent),
        }, logdir / "latest.pt")

    for env in train_envs + eval_envs:
        try: env.close()
        except: pass

    print("\nDone! tensorboard --logdir", logdir)


if __name__ == "__main__":
    main()