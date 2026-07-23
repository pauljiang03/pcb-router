"""Route the canonical TE board with no-training smart placement at every
orientation, single-layer and auto-layer, with length equalization.
Run: python scripts/route_canonical.py [--mirror] [--figs]
"""
import sys
import time
import argparse
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from envs.board import load_te_excel, smart_placement, rotate_board, rotate_points
from envs.routing import (route_all_traces, route_auto_layers, equalize_lengths,
                          route_to_length, optimize_layers_for_length,
                          octile_lower_bounds, count_crossings, min_trace_separation)


def spread(L):
    fin = [x for x in L if x < 1e9]
    return (max(fin) - min(fin)) / np.mean(fin) if len(fin) > 1 else 0.0


def equalize_layers(board, paths, lengths, layer_of, test_points):
    """Meander each layer separately against one global target; unrouted nets ride along as pad keep-outs."""
    n = len(paths)
    fin = [x for x in lengths if x < 1e9]
    if not fin:
        return paths, lengths, 0
    target_mm = max(fin) - board.traces[0].breakout_length
    unrouted = [i for i in range(n) if layer_of[i] < 0]
    eq = [None] * n
    eqL = [float("inf")] * n
    matched = 0
    for layer in sorted(set(l for l in layer_of if l >= 0)):
        idxs = [i for i in range(n) if layer_of[i] == layer] + unrouted
        ep, el, _t, m = equalize_lengths(
            board, [paths[i] for i in idxs], target_mm=target_mm,
            test_points=[test_points[i] for i in idxs])
        matched += m
        for k, i in enumerate(idxs):
            if layer_of[i] == layer:
                eq[i] = ep[k]
                eqL[i] = el[k]
    return eq, eqL, matched


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mirror", action="store_true",
                    help="also run the 4 mirrored orientations (8 total)")
    ap.add_argument("--figs", action="store_true",
                    help="render eval_results/router_canonical_*.png")
    ap.add_argument("--figs-only", action="store_true",
                    help="skip the orientation sweep; just render the figures")
    a = ap.parse_args()
    if a.figs_only:
        a.figs = True

    board = load_te_excel()
    n = len(board.traces)
    print(f"canonical board {board.width:.0f}x{board.height:.0f}mm, {n} traces")
    placed = smart_placement(board, n, verbose=True)

    orients = [] if a.figs_only else [
        (k, m) for m in ((False, True) if a.mirror else (False,))
        for k in range(4)]
    for k, mirror in orients:
        rb = rotate_board(board, k, mirror)
        rp = rotate_points(placed, board, k, mirror)

        t0 = time.time()
        paths, L, fails = route_all_traces(rb, rp)
        paths, L, _nd = route_to_length(rb, rp, paths, L)
        eq, eqL, _tgt, matched = equalize_lengths(rb, paths, test_points=rp)
        t1 = time.time() - t0
        x = count_crossings(eq)
        sep = min_trace_separation(eq)

        t0 = time.time()
        mp, mL, layer_of, mf, _ = route_auto_layers(rb, rp, max_layers=6)
        raw_max = max((x_ for x_ in mL if x_ < 1e9), default=0.0)
        mp, mL, layer_of, moves = optimize_layers_for_length(
            rb, rp, mp, mL, layer_of, max_layers=6)
        meq, meqL, mmatched = equalize_layers(rb, mp, mL, layer_of, rp)
        t2 = time.time() - t0
        used = sorted(set(l for l in layer_of if l >= 0))
        vias = sum(1 for l in layer_of if l >= 1)
        lb_max = max(octile_lower_bounds(rb, rp))
        opt_max = max((x_ for x_ in mL if x_ < 1e9), default=0.0)

        print(f"k={k} mirror={int(mirror)} | 1-layer {n-fails:2d}/{n} x={x} "
              f"sep={sep:.4f} spread {spread(L):.2f}->{spread(eqL):.2f} "
              f"matched {matched}/{n-fails} ({t1:.0f}s) | "
              f"auto {n-mf}/{n} layers={len(used)} vias={vias} "
              f"max {raw_max:.0f}->{opt_max:.0f}mm ({moves} moves, "
              f"padding {opt_max/lb_max:.2f}) "
              f"spread {spread(mL):.2f}->{spread(meqL):.2f} "
              f"matched {mmatched}/{n-mf} ({t2:.0f}s)",
              flush=True)

    if a.figs:
        from envs.visualize import render_board_png
        from envs.routing import CELL_SIZE, TP_CLEARANCE_CELLS
        KW = dict(labels=True, legend=True,
                  keepout_mm=TP_CLEARANCE_CELLS * CELL_SIZE)
        paths, L, fails = route_all_traces(board, placed)
        paths, L, _nd = route_to_length(board, placed, paths, L)
        eq, eqL, _t, matched = equalize_lengths(board, paths, test_points=placed)
        lbm = max(octile_lower_bounds(board, placed))
        finr = [x_ for x_ in L if x_ < 1e9]
        eq_label = ("LENGTH-EQUALIZED" if matched == n - fails else
                    f"NOT EQUALIZED ({matched}/{n-fails} at target)")
        render_board_png(
            board, placed, eq, "eval_results/router_canonical_smart.png", **KW,
            title=f"Canonical TE board, smart placement, SINGLE COPPER LAYER: "
                  f"{n-fails}/{n} routed, max {max(finr):.0f}mm "
                  f"(padding {max(finr)/lbm:.2f}), "
                  f"spread {spread(L):.2f}->{spread(eqL):.2f}, {eq_label}, "
                  f"min sep {min_trace_separation(eq):.2f}mm >= pitch"
                  + (", dotted = unrouted" if fails else ""))
        print("figure saved to eval_results/router_canonical_smart.png")


if __name__ == "__main__":
    main()
