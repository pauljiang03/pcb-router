"""Local search for the best test-point placement on a fixed board.

Hill-climbs from the smart_placement solution with the real router in the
loop: swap two nets' TP assignment, rotate three, or move one TP to a nearby
candidate; keep any change that routes 20/20 planar with a shorter max trace
(tie-broken by total length). A 200s probe on the canonical board took the
classical 95mm/1332mm solution to 86mm/1073mm and was still improving.

The saved JSON can anchor training: train.py --demo_placement <out> makes the
demo episodes replay THIS placement, so behavior cloning distills the search
result into the policy (expert iteration: search finds, the network learns).

Run: python scripts/optimize_placement.py --board canonical --minutes 10
"""
import argparse
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from envs.board import (load_te_excel, load_te_example, smart_placement,
                        generate_candidate_grid, check_tp_spacing)
from envs.routing import route_all_traces


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--board", default="canonical",
                    choices=["canonical", "te"])
    ap.add_argument("--num_traces", type=int, default=20)
    ap.add_argument("--minutes", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="eval_results/best_placement.json")
    a = ap.parse_args()

    board = (load_te_excel() if a.board == "canonical"
             else load_te_example(num_traces=a.num_traces))
    n = min(a.num_traces, len(board.traces))
    cand, real = generate_candidate_grid(board, 6.5)
    cand = cand[:real]
    fast = dict(n_starts=1, max_iters=12, repair_passes=1)

    def score(placed):
        _p, lengths, fails = route_all_traces(board, placed, **fast)
        fin = [x for x in lengths if x < 1e9]
        return fails, (max(fin) if fin else 1e9), (sum(fin) if fin else 1e9)

    best = list(smart_placement(board, n))
    bf, bmax, btot = score(best)
    print(f"start (smart): fails={bf} max={bmax:.1f} total={btot:.0f}",
          flush=True)

    rng = np.random.RandomState(a.seed)
    t_end = time.monotonic() + a.minutes * 60
    trials = accepted = 0
    while time.monotonic() < t_end:
        trials += 1
        trial = list(best)
        r = rng.rand()
        if r < 0.35 and n >= 2:      # swap two nets' TP assignment
            i, j = rng.choice(n, 2, replace=False)
            trial[i], trial[j] = trial[j], trial[i]
        elif r < 0.5 and n >= 3:     # rotate three assignments
            i, j, k = rng.choice(n, 3, replace=False)
            trial[i], trial[j], trial[k] = trial[j], trial[k], trial[i]
        else:                        # move one TP to a nearby candidate
            i = rng.randint(n)
            d = np.hypot(cand[:, 0] - trial[i][0], cand[:, 1] - trial[i][1])
            near = np.where((d > 0.1) & (d < 30.0))[0]
            if not len(near):
                continue
            p = tuple(cand[rng.choice(near)])
            if not check_tp_spacing(
                    [q for m, q in enumerate(trial) if m != i], *p):
                continue
            trial[i] = p
        fails, m, tot = score(trial)
        if fails == 0 and (m < bmax - 1e-6
                           or (m < bmax + 1e-6 and tot < btot - 0.5)):
            best, bmax, btot = trial, m, tot
            accepted += 1
            print(f"  max={m:.1f} total={tot:.0f} (trial {trials})",
                  flush=True)

    # Verify the final placement at the quality budget eval.py scores with,
    # INCLUDING the serpentine equalization post-stage (the fixture pads
    # every trace to the max, so a placement only counts if it equalizes).
    paths, lengths, fails = route_all_traces(board, best)
    fin = [x for x in lengths if x < 1e9]
    qmax, qtot = (max(fin), sum(fin)) if fin else (1e9, 1e9)
    matched = 0
    if fails == 0:
        from envs.routing import equalize_lengths
        _eqp, _eqL, _t, matched = equalize_lengths(board, paths,
                                                   test_points=best)
    print(f"\nbest (quality-budget verify): fails={fails} max={qmax:.1f} "
          f"total={qtot:.0f} matched={matched}/{n}  "
          f"[{trials} trials, {accepted} accepted]")
    if fails == 0 and matched < n:
        print("WARNING: not all traces reach the equalization target; "
              "prefer a placement with matched == n")

    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"board": a.board, "num_traces": n,
               "placement": [[float(x), float(y)] for x, y in best],
               "failures": int(fails), "max_mm": float(qmax),
               "total_mm": float(qtot), "matched": int(matched)},
              out.open("w"), indent=1)
    print("saved ->", out)


if __name__ == "__main__":
    main()
