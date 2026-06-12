# pcb-router-world

World-model RL for PCB test point placement.

## Structure

```
pcb-dreamer/          # main project
├── train.py          # training entry point
├── eval.py           # baseline evaluation (random, greedy, spread)
├── test_env.py       # quick env sanity check
├── configs.yaml      # DreamerV3 hyperparameters
├── dreamer.py        # Dreamer agent class
├── models.py         # WorldModel + ImagBehavior
├── networks.py       # RSSM, encoders, decoders, MLP
├── tools.py          # utilities, distributions, logger
├── exploration.py    # exploration strategies
├── parallel.py       # process/thread env wrappers
├── requirements.txt
├── envs/
│   ├── board.py          # board geometry, obstacles, candidate grid
│   ├── routing.py        # A* trace router with rip-up-and-retry
│   ├── pcb_env.py        # Gymnasium environment
│   ├── dreamer_wrapper.py # DreamerV3-compatible env wrapper
│   ├── visualize.py      # matplotlib board plots
│   └── wrappers.py       # env composition wrappers
└── docs/
    ├── plan.md       # implementation plan and design decisions
    └── method.md     # technical approach writeup
```

## Usage

```bash
pip install -r pcb-dreamer/requirements.txt
cd pcb-dreamer

# Train
python train.py --configs defaults --logdir ./logdir/pcb

# Evaluate baselines
python eval.py --episodes 5 --num_traces 10

# Quick env check
python test_env.py
```

See `pcb-dreamer/README.md` for full documentation.
