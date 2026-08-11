# Plan: World-Model-Guided Placement Search

Status: IMPLEMENTED 2026-08-10 (`scripts/wm_guided_search.py`,
`tests/test_wm_search.py`, notebook cells). Every code-level claim below was
verified against the repo before implementing; four corrections were applied —
see "Implementation notes" at the end. Remaining step: the user runs the
Colab cell against the v4 static checkpoint and reads the results against the
decision rules.

## Goal

Use the trained DreamerV3 world model as a **millisecond-scale surrogate of the
A* router** to accelerate placement search on the static canonical board. The
claim to demonstrate: *guided search reaches the same placement quality as
blind search in N× fewer real router calls* (target N ≥ 3), plus a measured
**surrogate fidelity** (rank correlation between model-predicted and true
returns). This is the one result in this repo where the world model's
distinctive capability (learned reward prediction) visibly does work that
nothing else can.

## Context (state of the repo as of commit `ea256be`+)

- Task: place 20 test points on a PCB; a deterministic A* router
  (`envs/routing.py:route_all_traces`) then routes; a serpentine equalizer
  pads all traces to the max. Objective (lexicographic): 0 planar failures →
  min max trace → min total, and the placement must fully equalize
  ("matched = 20/20", checked via `envs/routing.py:equalize_lengths`).
- Board: **static canonical** = `envs/board.py:load_te_excel()` (the real
  135×90mm AutoLayout Example01 board from the xlsx). `train.py --boards
  canonical` trains on it exclusively. 156 real candidates on a 6.5mm grid,
  action space padded to 200 (`MAX_CANDIDATES`).
- Classical baseline (`smart_placement`): 0 failures, max **95mm**, total
  1332mm, matched 20/20, eq_spread 0.0138.
- `scripts/optimize_placement.py`: blind hill-climb (swap / rotate-3 / move
  ≤30mm mutations, real router scores every trial, matched-gated milestones).
  A 200s ungated probe reached 86mm/1073mm; gated true optimum is TBD.
- Trained checkpoint: the user's Colab Drive,
  `/content/drive/MyDrive/pcb-router-logs-v4/canonical/latest.pt` — a 60k-step
  run on the static board (config `defaults colab_a100`, single_layer reward,
  shaping="potential", vector obs). Its policy exactly clones the smart
  solution. Its **world model** is sharp on this board (reward_loss ≈ 0.4,
  vector_loss ≈ 0.1) and its replay included 5k random-placement prefill
  steps, so the reward head has seen diverse placements, not just the clone.
  There is no checkpoint on the local machine — real experiments run in Colab
  (the user's notebook `pcb_router_colab-v3.ipynb` clones this repo from
  GitHub main; local work must be pushed to be usable there).

## Core mechanism: scoring a placement WITHOUT the router

Key insight: **every observation component is pure geometry** — the 64×64
render, the valid-candidate mask, and the 443-dim vector depend only on the
board and the TPs placed so far. Routing happens only in the terminal
*reward*. So a full 21-row episode observation sequence for any candidate
placement can be built for ~free, fed through the world model's posterior
(`dynamics.observe`), and the reward head's predictions summed = **predicted
episode return**. Because the training run used `shaping="potential"`,
per-step rewards carry length/crossing signal and the terminal carries the
routing outcome, so the summed prediction is a surrogate of the full shaped
objective. No imagination rollouts needed — this is posterior scoring on real
observations.

### Episode-format conventions (MUST match training exactly)

From `dreamerv3/tools.py:simulate` / `envs/dreamer_wrapper.py:PCBDreamerEnv`:

- Row 0 = reset obs; `action[0]` = zeros; `is_first[0]=True`.
- Row t (1..20) = obs AFTER the t-th placement; `action[t]` = one-hot(200) of
  the index chosen at t (convention: `action[t]` is the action that *produced*
  obs t); `is_terminal[20]=True`.
- `mask` and `vector` rows are the POST-step values (the wrapper emits them
  after the placement).
- Build rows by driving the inner env manually WITHOUT triggering terminal
  routing: after `env.reset()`, for each action index do
  `inner.placed_tps.append(tuple(inner.candidates[idx]));
  inner.current_trace += 1; inner._update_candidate_mask()`, then collect
  `{"image": inner._render_obs(), "mask": env._mask(),
  "vector": env._vector(), ...}`.
  Wrapper-attr note: `train.make_env(...)` returns
  UUID(SelectAction(TimeLimit(OneHotAction(PCBDreamerEnv)))); attribute READS
  like `env._inner`, `env._mask()`, `env._vector()` delegate down the chain
  automatically (BaseWrapper.__getattr__).

### Scoring code sketch

```python
data = {k: np.stack(batch_of_B_sequences), ...}   # (B, 21, ...) incl. zeros-row-0 actions
obs = wm.preprocess(data)                          # needs is_first/is_terminal
embed = wm.encoder(obs)
post, _ = wm.dynamics.observe(embed, obs["action"], obs["is_first"])
feat = wm.dynamics.get_feat(post)
pred = wm.heads["reward"](feat).mode()             # -> reshape (B, 21)
scores = pred.reshape(B, 21).sum(-1)               # predicted returns
```

Checkpoint loading recipe: mirror `eval.py:run_dreamer_policy` — yaml
`defaults` (+`colab_a100`), `cfg["num_actions"]=200`,
`torch.distributions.Distribution.set_default_validate_args(False)`,
`Dreamer(env.observation_space, env.action_space, config, logger,
dataset=None)`, load `latest.pt` `agent_state_dict`, use `agent._wm`,
`torch.no_grad()`. Obs/act spaces from `train.make_env(..., boards="canonical")`.

## Script spec: `scripts/wm_guided_search.py`

Args: `--checkpoint` (required for real runs; if missing/empty, run with
random weights and print a loud RANDOM-WEIGHTS warning — plumbing smoke only),
`--configs defaults colab_a100`, `--device cuda:0` (fallback cpu),
`--minutes 10` (guided-arm budget), `--batch 48` (mutations scored per
iteration), `--topk 2` (real router calls per iteration), `--fidelity 32`
(random-mutation samples for the correlation measurement), `--seed`, `--out
<json>`.

State is a list of 20 ACTION INDICES (coords = `candidates[idx]`); mutations
mirror the blind optimizer: swap two, rotate three, or move one to a candidate
within 30mm passing `check_tp_spacing` against the others.

Phase 1 — **fidelity**: sample `--fidelity` random mutations of the smart
placement; for each compute (a) the surrogate score and (b) the TRUE shaped
return by replaying the action sequence through the real wrapped env
(`env.step({"action": onehot})`, which routes at the terminal). Report
Spearman (implement rank-correlation manually via ranks + Pearson — scipy is
NOT a dependency) and Pearson. These router calls are bookkept separately from
the search arms.

Phase 2 — **guided arm**: per iteration, generate `--batch` mutations of the
incumbent, score all with the surrogate in ONE batched forward, real-route
only the `--topk` by predicted score (fast budget `n_starts=1, max_iters=12,
repair_passes=1`), accept lexicographic improvements `(fails, max, total)`,
and gate max-milestones on full equalization exactly as
`optimize_placement.py` does (`fully_matched` + fallback snapshot). Count
router calls C.

Phase 3 — **blind arm**: identical mutation generator and acceptance, but
route every mutation, capped at the SAME router-call budget C. Report both
arms' best (max/total/matched) at equal calls; also report the surrogate's
per-iteration hit rate (how often a top-k candidate was a true improvement).

Final: quality-budget route + equalization verify of the guided best; save
JSON (same schema as `optimize_placement.py`: board, placement, failures,
max_mm, total_mm, matched) so `train.py --demo_placement` can distill it.

## Verification

1. **Unit test** (`tests/test_coldstart.py`, no checkpoint, CI-safe): build
   the obs sequence for a random valid action sequence with the manual
   builder, and separately roll the SAME actions through the real wrapped env
   (small board OK — `boards="central"`, `num_traces=4`, routing at terminal
   is fine in a test); assert image/mask/vector rows are exactly equal and
   flags match. This pins the format-alignment risk, which is the main way
   this feature silently breaks. Import via `from scripts.wm_guided_search
   import ...` (namespace package, conftest puts repo root on sys.path).
2. **Local smoke** (no checkpoint): `--minutes 0.3 --fidelity 4` with random
   weights — verifies plumbing end to end. Keep local runs to ~2-3 min; the
   user's machine is weak and real experiments belong in Colab.
3. **Colab cell** (add to `pcb_router_colab-v3.ipynb` via NotebookEdit — the
   file must be edited with NotebookEdit, and cell ids shift after
   insert/delete, so re-Read between structural edits): run with
   `--checkpoint "{RUN_DIR}/latest.pt"` — but NOTE the user's static run dir
   is `pcb-router-logs-v4/canonical` while the current notebook RUN_DIR is
   `{LOGROOT}/canonical-static`; the cell should let the user point at
   whichever checkpoint exists. Push to main first — Colab pulls from GitHub.

## Decision rules for the results

- Fidelity Spearman ρ ≥ 0.6: surrogate is genuinely informative; the speedup
  claim is expected to hold. 0.3–0.6: partial guidance, report honestly.
  < 0.3: the reward head does not rank off-policy mutations — the honest
  extension is fine-tuning the reward head on the blind optimizer's trial log
  (placement → true score pairs), which turns this into "learned surrogate
  from search data"; note it, don't silently ship a weak result.
- Speedup: success = guided reaches the blind arm's equal-call quality with
  ≥3× headroom (i.e., blind needs ≥3C calls to match guided's best). If the
  surrogate is good but speedup is small, the mutation generator (not the
  model) is the bottleneck — say so.

## Known pitfalls (learned the hard way in this repo)

- macOS has no `timeout`; `status` is a read-only zsh variable; `=` starts an
  expansion in zsh.
- `mp` fork paths are Linux-only (demo generation); nothing in this plan needs
  fork.
- `generate_candidate_grid(board, res, max_candidates=BIG)` pads a Python list
  to max_candidates — never pass a huge cap.
- The reward head's `.mode()` may carry a trailing dim — reshape defensively.
- eval.py memoizes `evaluate_placement` per (board, placement, budget) — not
  used here, but don't be surprised by instant repeats.
- CI installs a hand-picked pip list (see `.github/workflows/tests.yml`) — a
  new import needs adding there or must be optional.
- Don't run anything heavy locally; push and run in Colab.

## Deliverables checklist

- [x] `scripts/wm_guided_search.py` per spec
- [x] obs-builder equality unit test (CI-safe, no checkpoint) —
      `tests/test_wm_search.py` (new file, not test_coldstart.py: it tests
      the search script, not the cold-start pipeline); also covers mutation
      validity and the manual Spearman
- [x] local random-weights smoke run passes (fails=0 max=94.5 matched=20/20
      final verify; fidelity ρ = nan under random weights because
      reward_head outscale=0 zero-inits the output layer — expected)
- [x] notebook cells (markdown `6a78da1c` + code `137a4e6c`, after the
      distill cell) added via NotebookEdit
- [x] pushed to main (Colab consumes GitHub)
- [ ] user runs in Colab against the v4 static checkpoint; report fidelity ρ,
      guided-vs-blind table, and the matched-verified best placement JSON

## Implementation notes (deviations from the spec above)

1. `build_env` passes `reward_mode="single_layer"` explicitly —
   `train.make_env`'s DEFAULT is `layer_aware`, which is NOT what the static
   canonical checkpoint trained with; inheriting it would silently change the
   true-return function and poison fidelity.
2. Blind arm runs to `--blind_mult` × C (default 3, wall-clock capped by
   `--blind_minutes`, default 3×`--minutes`) instead of stopping at C: the
   ≥3× decision rule needs the 3C region observed, not extrapolated. Both
   arms record best-vs-calls curves; the JSON reports blind@C and
   `calls_to_match_guided` (0 = the shared seed already matched).
3. The seed is `smart_placement` SNAPPED to the candidate grid
   (spacing-aware greedy nearest-candidate) since smart coords are off-grid
   and the model only ever saw grid placements; both arms start from the
   same snapped seed and its score is reported (fast-budget: 94.5mm, not
   95mm — the snap landed slightly better).
4. Fidelity samples mutate at depths 1/2/4/8 (cycled) rather than depth-1
   only, and additionally report Spearman(pred, −max) on the 0-failure
   subset — the search objective, not just the shaped return.
