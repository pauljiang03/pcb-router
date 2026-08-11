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
from envs.routing import route_all_traces, equalize_lengths


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

    def fully_matched(placed):
        """Route + serpentine-equalize: True iff every trace reaches the
        padding target (the fixture requirement training only proxies)."""
        paths, _L, fails = route_all_traces(board, placed, **fast)
        if fails:
            return False
        _eqp, _eqL, _t, m = equalize_lengths(board, paths, test_points=placed)
        return m == n

    best = list(smart_placement(board, n))
    bf, bmax, btot = score(best)
    print(f"start (smart): fails={bf} max={bmax:.1f} total={btot:.0f}",
          flush=True)
    # Last max-improving placement verified to equalize 20/20; the saved
    # answer never regresses below this.
    best_matched = (list(best), bmax, btot) if fully_matched(best) else None

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
            improved_max = m < bmax - 1e-6
            best, bmax, btot = trial, m, tot
            accepted += 1
            note = ""
            if improved_max:  # gate max milestones on equalization fitting
                if fully_matched(best):
                    best_matched = (list(best), bmax, btot)
                else:
                    note = "  [not equalizable -- kept climbing, not saved]"
            print(f"  max={m:.1f} total={tot:.0f} (trial {trials}){note}",
                  flush=True)

    # Final answer must equalize: if the last hill-climb state doesn't fit
    # its meanders, fall back to the best max-milestone that did.
    if not fully_matched(best):
        if best_matched is not None:
            print(f"final state not equalizable; reverting to last matched "
                  f"milestone (max={best_matched[1]:.1f})")
            best, bmax, btot = (list(best_matched[0]), best_matched[1],
                                best_matched[2])
        else:
            print("WARNING: no fully-matched placement found; saving the "
                  "best routed one anyway")

    # Verify at the quality budget eval.py scores with, incl. equalization.
    paths, lengths, fails = route_all_traces(board, best)
    fin = [x for x in lengths if x < 1e9]
    qmax, qtot = (max(fin), sum(fin)) if fin else (1e9, 1e9)
    matched = 0
    if fails == 0:
        _eqp, _eqL, _t, matched = equalize_lengths(board, paths,
                                                   test_points=best)
    print(f"\nbest (quality-budget verify): fails={fails} max={qmax:.1f} "
          f"total={qtot:.0f} matched={matched}/{n}  "
          f"[{trials} trials, {accepted} accepted]")

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
