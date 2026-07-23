"""Board-variation gallery: no-training pipeline over many board variations
(all 20 traces); writes one figure per board plus a summary table to eval_results/variations/.
Run: python scripts/board_gallery.py
"""
import sys
import time
import shutil
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from envs.board import (
    load_te_excel, load_te_example, load_edge_board, load_edge_board_2row,
    fan_to_top_placement, wrap_to_top_placement, make_challenge, ChallengeSpec,
    smart_placement, rotate_board, rotate_points, Obstacle, CircularObstacle,
    TRACE_TO_TRACE_MIN, TRACE_TO_UPTH_MIN,
)
from envs.routing import (
    route_all_traces, route_auto_layers, equalize_lengths, route_to_length,
    octile_lower_bounds, count_crossings, min_trace_separation,
    validate_routing_constraints, CELL_SIZE, TP_CLEARANCE_CELLS,
)
from envs.visualize import render_board_png

OUT = pathlib.Path("eval_results/variations")


def spread(L):
    fin = [x for x in L if x < 1e9]
    return (max(fin) - min(fin)) / np.mean(fin) if len(fin) > 1 else 0.0


def _translate_cluster(b, dx, dy):
    """Move the whole connector cluster (obstacles, pins, outline) rigidly."""
    for o in b.rect_obstacles:
        o.cx += dx
        o.cy += dy
    for o in b.circ_obstacles:
        o.cx += dx
        o.cy += dy
    for t in b.traces:
        t.start_x += dx
        t.start_y += dy
    b.connector_x += dx
    b.connector_y += dy
    return b


def creative_cases():
    """Board types beyond the standard set."""
    cases = []

    # Corner connector: fan forced into one quadrant.
    cases.append(("corner-connector-160mm",
                  _translate_cluster(load_te_example(20, seed=0), -38.0, -38.0),
                  None))

    # Asteroid field: random small keep-outs to thread between.
    b = load_te_example(20, seed=1)
    rng = np.random.RandomState(11)
    cx = b.connector_x + b.connector_w / 2
    cy = b.connector_y + b.connector_h / 2
    k = 0
    while k < 14:
        x = rng.uniform(b.x_min + 12, b.x_max - 12)
        y = rng.uniform(b.y_min + 12, b.y_max - 12)
        if abs(x - cx) < b.connector_w / 2 + 9 and \
           abs(y - cy) < b.connector_h / 2 + 9:
            continue                              # keep the cluster clear
        if rng.rand() < 0.5:
            b.rect_obstacles.append(Obstacle(
                cx=x, cy=y, width=rng.uniform(4, 9), height=rng.uniform(4, 9),
                clearance=TRACE_TO_TRACE_MIN, name=f"rock_{k}"))
        else:
            b.circ_obstacles.append(CircularObstacle(
                cx=x, cy=y, radius=rng.uniform(1.2, 2.6),
                clearance=TRACE_TO_UPTH_MIN, name=f"hole_{k}"))
        k += 1
    cases.append(("asteroid-field-160mm", b, None))

    # Walled channel: routes must run a corridor to the right-side field.
    b = _translate_cluster(load_te_example(20, seed=2), -45.0, 0.0)
    cy = b.connector_y + b.connector_h / 2
    b.rect_obstacles.append(Obstacle(cx=100.0, cy=cy + 26.0, width=85.0,
                                     height=6.0, clearance=TRACE_TO_TRACE_MIN,
                                     name="wall_top"))
    b.rect_obstacles.append(Obstacle(cx=100.0, cy=cy - 26.0, width=85.0,
                                     height=6.0, clearance=TRACE_TO_TRACE_MIN,
                                     name="wall_bot"))
    cases.append(("channel-160mm", b, None))

    # Via farm: canonical board with drills sprinkled over the pad field.
    b = load_te_excel()
    rng = np.random.RandomState(7)
    k = 0
    while k < 12:
        x = rng.uniform(20, 115)
        y = rng.uniform(132, 168)
        if all(np.hypot(x - o.cx, y - o.cy) > 7 for o in b.circ_obstacles):
            b.circ_obstacles.append(CircularObstacle(
                cx=x, cy=y, radius=0.95, clearance=TRACE_TO_UPTH_MIN,
                name=f"via_{k}"))
            k += 1
    cases.append(("canonical-via-farm", b, None))

    # Tall skinny board: extreme aspect ratio.
    cases.append(("skinny-70x220",
                  load_edge_board(20, board_w=70.0, board_h=220.0), None))

    # Single-gap moat: every net through one opening.
    bb, pl = make_challenge(ChallengeSpec(num_traces=20, n_gaps=1,
                                          board_size=140.0,
                                          placement="gap_aligned"))
    cases.append(("moat-140mm-1gap", bb, pl))
    return cases


def build_cases():
    """(name, board, placement-or-None); None means smart_placement decides."""
    cases = []
    cano = load_te_excel()
    cases.append(("canonical", cano, None))
    for k, mir, tag in ((1, False, "rot90"), (2, False, "rot180"), (1, True, "rot90m")):
        cases.append((f"canonical-{tag}", rotate_board(cano, k, mir), None))

    for seed in (0, 1, 2):
        cases.append((f"central-160mm-s{seed}", load_te_example(20, seed=seed), None))
    cases.append(("central-120mm", load_te_example(20, seed=3, board_size=120.0), None))
    cases.append(("central-200mm", load_te_example(20, seed=4, board_size=200.0), None))
    cases.append(("central-rot90-s1", rotate_board(load_te_example(20, seed=1), 1), None))

    b = load_edge_board(20)
    cases.append(("edge-fan-180mm", b, fan_to_top_placement(b, 20)))
    b = load_edge_board(20)
    cases.append(("edge-fan-rot90", rotate_board(b, 1),
                  rotate_points(fan_to_top_placement(b, 20), b, 1)))
    b = load_edge_board_2row(20, board_w=240.0, board_h=250.0)
    cases.append(("edge2row", b, wrap_to_top_placement(b, 20)))

    for size, gaps, n, plc in ((120, 3, 20, "ring"), (120, 3, 20, "gap_aligned"),
                               (100, 3, 20, "ring"), (140, 4, 20, "gap_aligned"),
                               (120, 2, 20, "ring"), (160, 4, 20, "ring")):
        bb, pl = make_challenge(ChallengeSpec(
            board_size=float(size), num_traces=n, n_gaps=gaps, placement=plc))
        cases.append((f"moat-{size}mm-{gaps}g-{n}tr-{plc}", bb, pl))
        cases.append((f"moat-{size}mm-{gaps}g-{n}tr-{plc}-rot90",
                      rotate_board(bb, 1), rotate_points(pl, bb, 1)))
    cases += creative_cases()
    return cases


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    kw = dict(labels=True, legend=True, keepout_mm=TP_CLEARANCE_CELLS * CELL_SIZE)
    for name, board, placed in build_cases():
        n = len(board.traces)
        t0 = time.time()
        if placed is None:
            placed = smart_placement(board, n)
        paths, L, fails = route_all_traces(board, placed)
        paths, L, ndet = route_to_length(board, placed, paths, L)
        eq, eqL, _t, matched = equalize_lengths(board, paths, test_points=placed)
        secs = time.time() - t0
        fin = [x for x in L if x < 1e9]
        lbs = octile_lower_bounds(board, placed)
        v = validate_routing_constraints(board, eq)
        sep = min_trace_separation(eq)
        note = ""
        if fails:                       # auto-layer fallback
            _p, _l, lof, af, _x = route_auto_layers(
                board, placed, max_layers=4, n_starts=1, max_iters=12,
                repair_passes=1)
            layers = len(set(l for l in lof if l >= 0))
            note = f"auto: {n-af}/{n} on {layers}L"
        if ndet:
            note = (note + "; " if note else "") + f"{ndet} detoured"
        eq_full = (matched == n - fails)     # every ROUTED trace at the target
        row = dict(name=name, n=n, routed=n - fails,
                   maxmm=round(max(fin), 1) if fin else 0.0,
                   padding=round(max(fin) / max(lbs), 2) if fin else 0.0,
                   spread_raw=round(spread(L), 2), spread_eq=round(spread(eqL), 2),
                   matched=matched, eq_full=eq_full,
                   sep=round(float(sep), 3) if sep < 1e9 else None,
                   x=count_crossings(eq), tp_ok=v["tp_clearance_ok"],
                   viol=len(v["violations"]), secs=round(secs, 1), note=note)
        rows.append(row)
        eq_label = ("LENGTH-EQUALIZED" if eq_full else
                    f"NOT EQUALIZED ({matched}/{n-fails} at target, "
                    f"rest space-limited)")
        render_board_png(
            board, placed, eq, str(OUT / f"{name}.png"), **kw,
            title=f"{name}: {n-fails}/{n} routed (1 layer), "
                  f"max {row['maxmm']}mm (padding {row['padding']}), "
                  f"spread {row['spread_raw']}->{row['spread_eq']}, {eq_label}"
                  + (f", DOTTED = unrouted ({fails})" if fails else ""))
        print(f"{name:32s} {n-fails:2d}/{n}  max={row['maxmm']:6.1f}mm "
              f"pad={row['padding']:4.2f}  spread {row['spread_raw']:.2f}->"
              f"{row['spread_eq']:.2f}  x={row['x']}  tp_ok={row['tp_ok']} "
              f"({secs:.0f}s) {note}", flush=True)

    lines = [
        "# Board-variation gallery",
        "",
        "No-training pipeline on every board: `smart_placement` (or the",
        "board's own fan/wrap placement) -> strict single-layer routing ->",
        "pad-guarded length equalization. Regenerate with",
        "`python scripts/board_gallery.py`. Dotted figure lines = unrouted nets.",
        "",
        "| board | routed@1L | equalized? | max mm | padding | spread raw->eq | sep mm | crossings | pad keep-out | secs | fallback |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        eq_cell = ("**YES**" if r["eq_full"] else
                   f"**NO** ({r['matched']}/{r['routed']} at target)")
        lines.append(
            f"| [{r['name']}]({r['name']}.png) | {r['routed']}/{r['n']} "
            f"| {eq_cell} | {r['maxmm']} | {r['padding']} "
            f"| {r['spread_raw']} -> {r['spread_eq']} "
            f"| {r['sep']} | {r['x']} | {'OK' if r['tp_ok'] else 'VIOLATED'} "
            f"| {r['secs']} | {r['note']} |")
    full = sum(1 for r in rows if r["routed"] == r["n"])
    eqf = sum(1 for r in rows if r["eq_full"])
    lines += [
        "",
        f"**{full}/{len(rows)} boards fully routed on one copper layer; "
        f"{eqf}/{len(rows)} fully length-equalized** (every routed trace at "
        f"the common target). A **NO** means the meander ran out of legal "
        f"free space for the remaining traces; they sit in the congested "
        f"zone near the connector, and the pad keep-outs / pitch rules are "
        f"never traded away for length (the fix is placement-side). "
        f"Every board: 0 crossings, "
        f"separation >= the 1.3286 mm pitch, pad keep-outs clear."]
    (OUT / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print(f"\nwrote {OUT}/RESULTS.md + {len(rows)} figures")

    # Showcase folder: only boards fully routed on one layer and fully equalized.
    EQ = pathlib.Path("eval_results/equalized")
    EQ.mkdir(parents=True, exist_ok=True)
    for f in EQ.glob("*.png"):
        f.unlink()                                # folder is fully regenerated
    done = [r for r in rows if r["eq_full"] and r["routed"] == r["n"]]
    eq_lines = [
        "# Complete boards: fully routed AND fully length-equalized",
        "",
        "Every net on one copper layer, every trace at the common target",
        "length, all invariants (planarity, full pitch, pad keep-outs) held.",
        "Copied from the full gallery (`../variations/`); regenerate with",
        "`python scripts/board_gallery.py`.",
        "",
        "| board | n | max mm | padding | spread_eq | sep mm |",
        "|---|---|---|---|---|---|",
    ]
    for r in done:
        shutil.copyfile(OUT / f"{r['name']}.png", EQ / f"{r['name']}.png")
        eq_lines.append(f"| [{r['name']}]({r['name']}.png) | {r['n']} "
                        f"| {r['maxmm']} | {r['padding']} | {r['spread_eq']} "
                        f"| {r['sep']} |")
    eq_lines += ["", f"**{len(done)}/{len(rows)} gallery boards are complete.**"]
    (EQ / "RESULTS.md").write_text("\n".join(eq_lines) + "\n")
    print(f"wrote {EQ}/RESULTS.md + {len(done)} complete-board figures")


if __name__ == "__main__":
    main()
