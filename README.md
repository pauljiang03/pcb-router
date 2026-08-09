# World-Model-Based Test Point Placement for SI Fixture Routing

![tests](https://github.com/pauljiang03/pcb-router/actions/workflows/tests.yml/badge.svg)

DreamerV3 learns where to place test points on a PCB test fixture; a deterministic
octilinear (45°) A* router with negotiated rip-up-and-reroute then routes every
trace and equalizes lengths with serpentine meanders. Placement is the only
learned stage; routing and length equalization are deterministic and planar by
construction (no crossings, full 1.33 mm pitch, pad keep-outs enforced).

**Canonical result:** the TE AutoLayout Example01 board (135×90 mm, 20 traces,
`AutoLayout_Example01.xlsx`) routes **20/20 on a single copper layer, 20/20
length-matched** (spread 0.014) under a 5.98 mm pad keep-out, in ~1 s.
Figures for this and other boards: [`eval_results/equalized/`](eval_results/equalized/).

![canonical board](eval_results/equalized/canonical.png)

## Layout

- `envs/` - PCB environment: board loading/placement, router, length equalization, visualization
- `dreamerv3/` - forked DreamerV3 engine
- `scripts/` - `route_canonical.py` (reproduce the canonical result), `board_gallery.py` (figure gallery), `train_ordering.py` (net-ordering model, `models/ordering.npz`)
- `train.py` / `train_ppo.py` / `eval.py` - training and evaluation entry points
- `tests/` - unit tests (run in CI on every push)

## Quickstart

```bash
pip install -r requirements.txt

python -m pytest tests/ -q                              # unit tests
python eval.py --episodes 5 --num_traces 20 --no-plot   # baseline metrics (--fast: quick low-budget pass)
python scripts/route_canonical.py --mirror --figs       # canonical 20/20 board

python train.py --configs defaults --device cuda:0 --num_traces 20   # train Dreamer
python train_ppo.py --steps 200000 --num_traces 8                    # PPO baseline
```

## Cold start + dense reward (on by default in train.py)

The terminal routing reward is a cliff-shaped function of the whole placement,
and random exploration rarely produces a fully-routed episode for the reward
head to learn from. Training therefore uses three additions (each can be
disabled):

- **Expert demos** — `smart_placement` episodes are generated once into
  `<logdir>/demo_eps` (resumable; also standalone via `python demos.py`) and
  sampled into batches at a fixed fraction so the world model and reward head
  see high-reward placements from step 0. `--demos 0` disables.
- **Decayed behavior cloning** — demo steps add a BC term to the actor loss
  through the same masked policy dist that acts in the env, decaying linearly
  from `bc_scale` to 0 over `bc_decay` env steps (so RL can eventually beat
  the heuristic instead of anchoring to it). `--bc_scale 0` disables.
- **Potential-based reward shaping** — each step pays the delta of a cheap
  placement potential (`wire_estimate` lengths + crossing pin→TP chords) and
  the terminal step refunds it, so episode totals are *identical* to the
  unshaped env (train_return stays comparable) while credit lands on the
  placement that caused it. `shaping="none"` in the env restores the old
  per-step rewards; `eval.py` scoring is unaffected either way.
