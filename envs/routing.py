"""
Negotiated octilinear A* routing on a cell-discretized PCB grid.
Produces planar, full-pitch routings; see route_all_traces for the pipeline.
"""

import numpy as np
import heapq
from typing import List, Tuple, Optional, Set, Dict
from envs.board import (
    BoardSpec, TRACE_WIDTH, TRACE_TO_TRACE_MIN,
    TRACE_TO_EDGE_MIN, TRACE_TO_UPTH_MIN, TRACE_TO_TABPAD_MIN,
)

CELL_SIZE = TRACE_WIDTH + TRACE_TO_TRACE_MIN  # 1.3286 mm

# Pad keep-out radius (cells) other nets may not enter around each test point.
# With TP_TO_TP_MIN = 13 mm, two min-spaced pads' disks leave < 1 routing lane, so traces must route AROUND pad clusters.
TP_CLEARANCE_CELLS = 4.5

# Max depth (cells) of a length-matching meander tooth.
_MAX_BUMP_DEPTH = 8


class RoutingGrid:

    def __init__(self, board: BoardSpec):
        self.board = board
        self.res = CELL_SIZE
        self.cols = int(np.ceil(board.width / self.res))
        self.rows = int(np.ceil(board.height / self.res))
        self.grid = np.zeros((self.rows, self.cols), dtype=np.uint8)

        self._rasterize_edge_clearance()
        self._rasterize_obstacles()
        self.obstacle_grid = self.grid.copy()

    # ---- coordinates ----

    def _world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        c = int(round((x - self.board.x_min) / self.res - 0.5))
        r = int(round((y - self.board.y_min) / self.res - 0.5))
        return int(np.clip(c, 0, self.cols - 1)), int(np.clip(r, 0, self.rows - 1))

    def _grid_to_world(self, col: int, row: int) -> Tuple[float, float]:
        return (self.board.x_min + (col + 0.5) * self.res,
                self.board.y_min + (row + 0.5) * self.res)

    # ---- initial rasterization ----

    def _rasterize_edge_clearance(self):
        n = max(1, int(np.ceil((TRACE_TO_EDGE_MIN + TRACE_WIDTH / 2) / self.res)))
        self.grid[:n, :] = 1
        self.grid[-n:, :] = 1
        self.grid[:, :n] = 1
        self.grid[:, -n:] = 1

    def _rasterize_obstacles(self):
        hw = TRACE_WIDTH / 2
        for obs in self.board.rect_obstacles:
            xn, yn, xx, yx = obs.bounds
            b = obs.clearance + hw
            c0, r0 = self._world_to_grid(xn - b, yn - b)
            c1, r1 = self._world_to_grid(xx + b, yx + b)
            self.grid[max(0, r0):min(self.rows, r1 + 1),
                      max(0, c0):min(self.cols, c1 + 1)] = 1
        for obs in self.board.circ_obstacles:
            b = obs.radius + obs.clearance + hw
            c0, r0 = self._world_to_grid(obs.cx - b, obs.cy - b)
            c1, r1 = self._world_to_grid(obs.cx + b, obs.cy + b)
            for r in range(max(0, r0), min(self.rows, r1 + 1)):
                for c in range(max(0, c0), min(self.cols, c1 + 1)):
                    wx, wy = self._grid_to_world(c, r)
                    if np.hypot(wx - obs.cx, wy - obs.cy) < b:
                        self.grid[r, c] = 1


# ------------------------------------------------------------------
# Negotiated-congestion router internals
# ------------------------------------------------------------------

# NB: two paths can cross without sharing a cell via the two complementary
# diagonals of one unit square; crossings must be handled separately.
_NEG_NBR = [(-1, 0), (1, 0), (0, -1), (0, 1),
            (-1, -1), (-1, 1), (1, -1), (1, 1)]


def _diag_key(r, c, nr, nc):
    """(unit-square, diagonal-type) for a diagonal move; type t crosses type 1-t."""
    return (min(r, nr), min(c, nc)), (0 if (nr - r) == (nc - c) else 1)


# Edge-adjacent unit squares (parallel-diagonal clearance rule).
_ADJ4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _diag_corners(sq, t):
    """The two cells of square `sq` not used by a type-`t` diagonal; a foreign trace
    through either sits exactly pitch/sqrt(2) from the diagonal (sub-pitch)."""
    (r, c) = sq
    return ((r, c + 1), (r + 1, c)) if t == 0 else ((r, c), (r + 1, c + 1))


def _build_context(board: BoardSpec, test_points: List[Tuple[float, float]]):
    """Shared router setup; returns (grid, rows, cols, blocked, cells, endpoints, owner).
    owner[r, c] == i means only net i may enter that cell (pad disk, start pin, stub)."""
    n = min(len(board.traces), len(test_points))
    grid = RoutingGrid(board)
    rows, cols = grid.rows, grid.cols
    blocked = grid.obstacle_grid > 0

    cells = []
    for i in range(n):
        t = board.traces[i]
        sc, sr = grid._world_to_grid(t.start_x, t.start_y)
        ec, er = grid._world_to_grid(*test_points[i])
        cells.append(((sr, sc), (er, ec)))
        blocked[sr, sc] = False
        blocked[er, ec] = False

    # Escape stubs must run PERPENDICULAR to the pin row (a stub along the row
    # ploughs through neighbouring pins), and may never be carved over a drilled
    # hole's body (a trace could then run over the hole itself).
    hole_body = np.zeros_like(blocked)
    hw = TRACE_WIDTH / 2.0
    for obs in board.circ_obstacles:
        b2 = obs.radius + hw
        c0, r0 = grid._world_to_grid(obs.cx - b2, obs.cy - b2)
        c1, r1 = grid._world_to_grid(obs.cx + b2, obs.cy + b2)
        for rr in range(max(0, r0), min(rows, r1 + 1)):
            for cc in range(max(0, c0), min(cols, c1 + 1)):
                wx, wy = grid._grid_to_world(cc, rr)
                if np.hypot(wx - obs.cx, wy - obs.cy) < b2:
                    hole_body[rr, cc] = True

    ccx = board.connector_x + board.connector_w / 2.0
    ccy = board.connector_y + board.connector_h / 2.0
    ccol, crow = grid._world_to_grid(ccx, ccy)
    starts = [s for s, _e in cells]
    vote = 0                       # >0: rows run horizontally (columns vary)
    for i, (sr, sc) in enumerate(starts):
        best = None
        for j, (orr, occ) in enumerate(starts):
            if j != i:
                d2 = (orr - sr) ** 2 + (occ - sc) ** 2
                if best is None or d2 < best[0]:
                    best = (d2, abs(occ - sc) >= abs(orr - sr))
        if best is not None:
            vote += 1 if best[1] else -1
    stub_dir = []                  # per-net (dr, dc) unit cardinal escape step
    for (sr, sc), _e in cells:
        if len(starts) > 1:
            vertical_escape = vote >= 0
        else:                      # lone pin: dominant offset from the center
            vertical_escape = abs(sr - crow) >= abs(sc - ccol)
        if vertical_escape:
            d = (1 if sr > crow else -1, 0)
        else:
            d = (0, 1 if sc > ccol else -1)
        stub_dir.append(d)
        for j in range(5):
            rr, cc = sr + d[0] * j, sc + d[1] * j
            if not (0 <= rr < rows and 0 <= cc < cols) or hole_body[rr, cc]:
                break                      # never carve over a drilled hole
            blocked[rr, cc] = False

    endpoints = set()
    for s, e in cells:
        endpoints.add(s)
        endpoints.add(e)

    owner = -np.ones((rows, cols), dtype=np.int32)
    rad = TP_CLEARANCE_CELLS
    R = int(np.ceil(rad))
    for i, (_s, (er, ec)) in enumerate(cells):
        for dr in range(-R, R + 1):
            for dc in range(-R, R + 1):
                if dr * dr + dc * dc <= rad * rad:
                    rr, cc = er + dr, ec + dc
                    if 0 <= rr < rows and 0 <= cc < cols:
                        owner[rr, cc] = i
    # Start + stub ownership is assigned LAST so it wins over any overlapping pad disk.
    for i, ((sr, sc), _e) in enumerate(cells):
        dr, dc = stub_dir[i]
        for j in range(6):
            rr, cc = sr + dr * j, sc + dc * j
            if not (0 <= rr < rows and 0 <= cc < cols) or hole_body[rr, cc]:
                break          # owner claims one cell beyond the carved stub
            owner[rr, cc] = i
    return grid, rows, cols, blocked, cells, endpoints, owner


def _astar_cost(blocked, cell_cost, diag_present, diag_hist, rows, cols, start, end,
                present_penalty, owner=None, net_id=-1, diagonal=True,
                hard_diag=None, strict=False, corner_present=None, kept_cells=None,
                adj_present=None):
    """Octilinear A* with congestion/ownership/strict-pitch rules; returns cells or None."""
    hard_corners = None
    if strict and hard_diag:
        hard_corners = set()
        for hsq, ht in hard_diag:
            hard_corners.update(_diag_corners(hsq, ht))
    g = {start: 0.0}
    came = {}
    pq = [(0.0, start)]
    seen = set()
    er, ec = end
    while pq:
        _, cur = heapq.heappop(pq)
        if cur in seen:
            continue
        seen.add(cur)
        if cur == end:
            path = [cur]
            while cur in came:
                cur = came[cur]
                path.append(cur)
            return path[::-1]
        r, c = cur
        for dr, dc in (_NEG_NBR if diagonal else _NEG_NBR[:4]):   # 4-conn = rectilinear
            nr, nc = r + dr, c + dc
            if not (0 <= nr < rows and 0 <= nc < cols) or blocked[nr, nc] or (nr, nc) in seen:
                continue
            if owner is not None and owner[nr, nc] >= 0 and owner[nr, nc] != net_id:
                continue          # keep clear of other nets' pads / pins / stubs
            if hard_corners is not None and (nr, nc) in hard_corners:
                continue          # cell sits pitch/sqrt(2) from a kept diagonal
            extra = 0.0
            if corner_present is not None:
                extra += present_penalty * corner_present.get((nr, nc), 0)
            if dr and dc:
                if blocked[r, nc] and blocked[nr, c]:
                    continue                       # don't cut the corner between obstacles
                sq, t = _diag_key(r, c, nr, nc)
                if hard_diag is not None and (sq, 1 - t) in hard_diag:
                    continue                       # would cross a kept net's diagonal
                if strict and hard_diag is not None:
                    if any(((sq[0] + ar, sq[1] + ac), t) in hard_diag
                           for ar, ac in _ADJ4):
                        continue                   # parallel to a kept diagonal, 1 lane
                    if kept_cells is not None and (
                            (r, nc) in kept_cells or (nr, c) in kept_cells):
                        continue                   # kept cell on my complementary corner
                step = 1.4142135624
                extra += present_penalty * diag_present.get((sq, 1 - t), 0) + diag_hist.get((sq, t), 0.0)
                if strict:
                    if adj_present is not None:    # parallel-diagonal pressure
                        extra += present_penalty * adj_present.get((sq, t), 0)
                    extra += cell_cost[r, nc] + cell_cost[nr, c]   # corner-cell pressure
            else:
                step = 1.0
            ng = g[cur] + step + cell_cost[nr, nc] + extra
            if ng < g.get((nr, nc), 1e18):
                g[(nr, nc)] = ng
                came[(nr, nc)] = cur
                dx, dy = abs(nr - er), abs(nc - ec)
                h = max(dx, dy) + 0.4142135624 * min(dx, dy)   # octile, admissible
                heapq.heappush(pq, (ng + h, (nr, nc)))
    return None


# ------------------------------------------------------------------
# Public API: negotiated-congestion rip-up-and-reroute
# ------------------------------------------------------------------

def _negotiate(blocked, rows, cols, cells, endpoints, order,
               max_iters, present_penalty, history_inc, owner=None, diagonal=True,
               strict=False):
    """One negotiated rip-up-and-reroute run; returns per-net cell paths (or None)."""
    n = len(cells)
    history = np.zeros((rows, cols))
    present = np.zeros((rows, cols), dtype=np.int32)
    diag_present = {}                 # (square, type) -> count
    diag_hist = {}                    # (square, type) -> history cost
    # Sparse dicts on purpose; measured faster than numpy scalar indexing in the A* hot loop.
    corner_present = {} if strict else None
    adj_present = {} if strict else None
    clear_conf = set()                # nets in a clearance conflict last pass (strict)
    routes = [None] * n
    net_diags = [None] * n           # per-net list of diagonal keys
    pp = present_penalty
    since_best, best_conf = 0, None

    def conflicted(i):
        if not routes[i]:
            return True
        if any(present[cell] > 1 for cell in routes[i]):
            return True
        if any(diag_present.get((sq, 1 - t), 0) > 0 for sq, t in net_diags[i]):
            return True
        return i in clear_conf

    def _corner_add(keys, d):
        for (sq, t) in keys:
            for cc in _diag_corners(sq, t):
                corner_present[cc] = corner_present.get(cc, 0) + d
            for ar, ac in _ADJ4:
                k2 = ((sq[0] + ar, sq[1] + ac), t)
                adj_present[k2] = adj_present.get(k2, 0) + d

    for it in range(max_iters):
        for i in order:
            if it and not conflicted(i):
                continue                      # settled net: leave it in place
            if routes[i]:
                for cell in routes[i]:
                    present[cell] -= 1
            if net_diags[i]:
                for key in net_diags[i]:
                    diag_present[key] -= 1
                if strict:
                    _corner_add(net_diags[i], -1)
            cost = history + pp * present
            p = _astar_cost(blocked, cost, diag_present, diag_hist, rows, cols,
                            cells[i][0], cells[i][1], pp, owner, i, diagonal,
                            strict=strict, corner_present=corner_present,
                            adj_present=adj_present)
            routes[i] = p
            net_diags[i] = []
            if p:
                for cell in p:
                    present[cell] += 1
                for k in range(len(p) - 1):
                    (r, c), (nr, nc) = p[k], p[k + 1]
                    if (nr - r) and (nc - c):
                        key = _diag_key(r, c, nr, nc)
                        diag_present[key] = diag_present.get(key, 0) + 1
                        net_diags[i].append(key)
                if strict:
                    _corner_add(net_diags[i], +1)
        cell_conf = [(r, c) for r, c in zip(*np.where(present > 1))
                     if (r, c) not in endpoints]
        squares = {}
        for (sq, t), cnt in diag_present.items():
            if cnt > 0:
                squares.setdefault(sq, set()).add(t)
        diag_conf = [sq for sq, ts in squares.items() if len(ts) == 2]
        strict_conf = 0
        if strict:
            # Identity-aware clearance conflicts; a net's OWN jogs never count.
            clear_conf = set()
            keyowners = {}
            for i2 in range(n):
                for key in (net_diags[i2] or ()):
                    keyowners.setdefault(key, set()).add(i2)
            cellowners = {}
            for i2 in range(n):
                if routes[i2]:
                    for cell in routes[i2]:
                        cellowners.setdefault(cell, set()).add(i2)
            for (sq, t), owns in keyowners.items():
                for ar, ac in _ADJ4:
                    o2 = keyowners.get(((sq[0] + ar, sq[1] + ac), t))
                    if o2 and len(owns | o2) > 1:         # different nets
                        strict_conf += 1
                        clear_conf |= owns | o2
                        diag_hist[(sq, t)] = diag_hist.get((sq, t), 0.0) + history_inc
                for cc in _diag_corners(sq, t):
                    oc = cellowners.get(cc)
                    if oc and (oc - owns):                 # foreign cell on corner
                        strict_conf += 1
                        clear_conf |= owns | (oc - owns)
                        diag_hist[(sq, t)] = diag_hist.get((sq, t), 0.0) + history_inc
                        if cc not in endpoints:            # a pin/pad cannot move
                            history[cc] += history_inc
        if not cell_conf and not diag_conf and not strict_conf:
            break
        conf = len(cell_conf) + len(diag_conf) + strict_conf
        if best_conf is None or conf < best_conf:
            best_conf, since_best = conf, 0
        else:
            since_best += 1       # counts every pass since the best conflict count
            if since_best == 3:
                pp = min(pp * 2.0, 8.0 * present_penalty)   # push overlaps apart
            elif since_best >= 8:
                break             # plateau: infeasible here; let repair passes try
        for cell in cell_conf:
            history[cell] += history_inc
        for sq in diag_conf:
            diag_hist[(sq, 0)] = diag_hist.get((sq, 0), 0.0) + history_inc
            diag_hist[(sq, 1)] = diag_hist.get((sq, 1), 0.0) + history_inc
    return routes


def _kept_blockers(routes, cells, blocked, skip):
    """Hard-block kept nets' cells (except net `skip`); returns (blocked, diag keys, cells)."""
    resc = blocked.copy()
    hard = set()
    kept = set()
    for j in range(len(routes)):
        if j == skip or not routes[j]:
            continue
        for (r, c) in routes[j]:
            resc[r, c] = True
            kept.add((r, c))
        for k in range(len(routes[j]) - 1):
            (r, c), (nr, nc) = routes[j][k], routes[j][k + 1]
            if (nr - r) and (nc - c):
                hard.add(_diag_key(r, c, nr, nc))
    return resc, hard, kept


def _diag_crossing_pairs(routes):
    """Crossing pairs among cell-disjoint unit-move paths: such paths can only cross
    via complementary diagonals of one unit square, so scanning diagonal keys is exact."""
    by_square = {}                    # square -> {type -> set(net)}
    for i, rt in enumerate(routes):
        if not rt:
            continue
        for k in range(len(rt) - 1):
            (r, c), (nr, nc) = rt[k], rt[k + 1]
            if (nr - r) and (nc - c):
                sq, t = _diag_key(r, c, nr, nc)
                by_square.setdefault(sq, ({}, {}))[t][i] = True
    pairs = set()
    for t0, t1 in by_square.values():
        for i in t0:
            for j in t1:
                if i != j:
                    pairs.add((min(i, j), max(i, j)))
    return pairs


def _sub_pitch_pairs(routes):
    """Pairs of cell-disjoint paths closer than one cell pitch: the only sub-pitch
    patterns are (X) crossing, (P) parallel diagonals one lane apart, and (C) a
    foreign cell on a diagonal's complementary corner; all exactly pitch/sqrt(2)."""
    keyowners = {}
    cellowners = {}
    for i, rt in enumerate(routes):
        if not rt:
            continue
        for cell in rt:
            cellowners.setdefault(cell, set()).add(i)
        for k in range(len(rt) - 1):
            (r, c), (nr, nc) = rt[k], rt[k + 1]
            if (nr - r) and (nc - c):
                keyowners.setdefault(_diag_key(r, c, nr, nc), set()).add(i)
    pairs = set()

    def _cross(a_set, b_set):
        for a in a_set:
            for b in b_set:
                if a != b:
                    pairs.add((min(a, b), max(a, b)))

    for (sq, t), owns in keyowners.items():
        comp = keyowners.get((sq, 1 - t))
        if comp:
            _cross(owns, comp)                            # (X) crossing
        for ar, ac in _ADJ4:
            o2 = keyowners.get(((sq[0] + ar, sq[1] + ac), t))
            if o2:
                _cross(owns, o2)                          # (P) parallel, 1 lane
        for cc in _diag_corners(sq, t):
            oc = cellowners.get(cc)
            if oc:
                _cross(owns, oc)                          # (C) corner cell
    return pairs


def _remove_crossings(routes, cells, rows, cols, grid,
                      blocked=None, owner=None, diagonal=True, strict=False):
    """Guarantee a planar result: drop minimal nets for cell overlaps, then reroute
    (or drop) crossing nets around the kept set."""
    n = len(routes)
    present = np.zeros((rows, cols), dtype=np.int32)
    for rt in routes:
        if rt:
            for cell in rt:
                present[cell] += 1
    while True:
        worst, worst_ov = -1, 0
        for i in range(n):
            if routes[i]:
                ov = sum(1 for c in routes[i] if present[c] > 1 and c not in cells[i])
                if ov > worst_ov:
                    worst, worst_ov = i, ov
        if worst < 0:
            break
        for cell in routes[worst]:
            present[cell] -= 1
        routes[worst] = None
    zero_cost = np.zeros((rows, cols))
    tried = set()
    while True:
        pairs = _sub_pitch_pairs(routes) if strict else _diag_crossing_pairs(routes)
        if not pairs:
            break
        deg = {}
        for i, j in pairs:
            deg[i] = deg.get(i, 0) + 1
            deg[j] = deg.get(j, 0) + 1
        # Prefer an untried net: a violation pinned to a START cell can only be
        # fixed by moving the OTHER net, so the partner gets the next attempt.
        worst = max(deg, key=lambda k: (k not in tried, deg[k]))
        rerouted = None
        if blocked is not None and worst not in tried:
            tried.add(worst)
            resc, hard, kept = _kept_blockers(routes, cells, blocked, worst)
            (sr, sc), (er, ec) = cells[worst]
            resc[sr, sc] = False
            resc[er, ec] = False
            rerouted = _astar_cost(resc, zero_cost, {}, {}, rows, cols,
                                   (sr, sc), (er, ec), 0.0, owner, worst,
                                   diagonal, hard_diag=hard,
                                   strict=strict, kept_cells=kept)
        routes[worst] = rerouted          # None (drop) only if no legal path
    return routes


def route_all_traces(
    board: BoardSpec,
    test_points: List[Tuple[float, float]],
    max_iters: int = 40,
    present_penalty: float = 4.0,
    history_inc: float = 1.0,
    n_starts: int = 6,
    diagonal: bool = True,
    repair_passes: int = 3,
    strict_clearance: bool = True,
    order_hint: Optional[List[int]] = None,
) -> Tuple[List[Optional[List[Tuple[float, float]]]], List[float], int]:
    """Route all traces (negotiated rip-up-and-reroute + multi-start); returns
    (paths_world, lengths, failures)."""
    n = min(len(board.traces), len(test_points))
    if n == 0:
        return [], [], 0

    grid, rows, cols, blocked, cells, endpoints, owner = _build_context(
        board, test_points)
    res = grid.res

    dist = [np.hypot(test_points[i][0] - board.traces[i].start_x,
                     test_points[i][1] - board.traces[i].start_y) for i in range(n)]
    base = list(range(n))
    informed = [base,
                sorted(base, key=lambda i: -dist[i]),   # longest first
                sorted(base, key=lambda i: dist[i])]    # shortest first
    # A caller-supplied routing order tries first.
    if order_hint is not None and sorted(order_hint) == base:
        informed.insert(0, list(order_hint))
    rng = np.random.RandomState(0)
    best_routes, best_fails = None, None

    def try_order(order):
        """One negotiated run + planarity/clearance pass; kept if best so far."""
        nonlocal best_routes, best_fails
        routes = _negotiate(blocked, rows, cols, cells, endpoints, order,
                            max_iters, present_penalty, history_inc, owner, diagonal,
                            strict=strict_clearance)
        routes = _remove_crossings(routes, cells, rows, cols, grid,
                                   blocked=blocked, owner=owner, diagonal=diagonal,
                                   strict=strict_clearance)
        fails = sum(1 for rt in routes if rt is None)
        if best_fails is None or fails < best_fails:
            best_fails, best_routes = fails, routes
        return fails

    for k in range(max(1, n_starts)):
        order = informed[k] if k < len(informed) else list(rng.permutation(n))
        if try_order(order) == 0:
            break

    # Extra restarts only when 1-2 nets are dropped (order is then the blocker);
    # with many drops the placement itself is infeasible and more orders burn time.
    k = max(1, n_starts)
    while best_fails and best_fails <= 2 and k < 2 * max(1, n_starts):
        try_order(list(rng.permutation(n)))
        k += 1

    # Failed-first repair: route still-dropped nets before the board fills around them.
    attempt = 0
    while best_fails and attempt < repair_passes:
        attempt += 1
        failed = [i for i in range(n) if best_routes[i] is None]
        rest = [i for i in range(n) if best_routes[i] is not None]
        rng.shuffle(rest)
        try_order(failed + rest)

    routes = best_routes

    # Rescue: reroute each dropped net with kept nets hard-blocked (planar by construction).
    dropped = sorted((i for i in range(n) if routes[i] is None), key=lambda i: dist[i])
    if dropped:
        zero_cost = np.zeros((rows, cols))
        resc, hard, kept = _kept_blockers(routes, cells, blocked, skip=-1)
        for i in dropped:
            (sr, sc), (er, ec) = cells[i][0], cells[i][1]
            resc[sr, sc] = False
            resc[er, ec] = False
            p = _astar_cost(resc, zero_cost, {}, {}, rows, cols, (sr, sc), (er, ec),
                            present_penalty, owner, i, diagonal, hard_diag=hard,
                            strict=strict_clearance, kept_cells=kept)
            if p is None:
                continue
            # Planar/full-pitch by construction; verify the invariant cheaply anyway.
            new_keys = [_diag_key(*p[k], *p[k + 1]) for k in range(len(p) - 1)
                        if (p[k + 1][0] - p[k][0]) and (p[k + 1][1] - p[k][1])]
            if any((sq, 1 - t) in hard for sq, t in new_keys):
                continue                                         # never expected
            routes[i] = p
            if strict_clearance and any(i in pr for pr in _sub_pitch_pairs(routes)):
                routes[i] = None                                 # never expected
                continue
            for (r, c) in p:
                resc[r, c] = True                                # block for later rescues
                kept.add((r, c))
            hard.update(new_keys)

    if strict_clearance:
        # Defensive final sweep (never expected to trigger): drop the worst
        # offender until no sub-pitch pair remains.
        while True:
            residual = _sub_pitch_pairs(routes)
            if not residual:
                break
            deg = {}
            for a, b in residual:
                deg[a] = deg.get(a, 0) + 1
                deg[b] = deg.get(b, 0) + 1
            routes[max(deg, key=lambda k_: deg[k_])] = None

    paths: List[Optional[List[Tuple[float, float]]]] = []
    lengths: List[float] = []
    failures = 0
    for i in range(n):
        rt = routes[i]
        if rt is None:
            paths.append(None)
            lengths.append(float('inf'))
            failures += 1
        else:
            world = [grid._grid_to_world(c, r) for (r, c) in rt]
            plen = sum(np.hypot(rt[k + 1][0] - rt[k][0], rt[k + 1][1] - rt[k][1])
                       for k in range(len(rt) - 1)) * res
            paths.append(world)
            lengths.append(plen + board.traces[i].breakout_length)
    return paths, lengths, failures


def route_auto_layers(board, test_points, max_layers=4, **kwargs):
    """Route with automatic layer assignment (push unroutable nets to the next layer);
    returns (paths_world, lengths, layer_of, failures, layer_crossings)."""
    import copy
    n = min(len(board.traces), len(test_points))
    paths = [None] * n
    lengths = [float('inf')] * n
    layer_of = [-1] * n
    remaining = list(range(n))
    layer_crossings = {}
    for layer in range(max_layers):
        if not remaining:
            break
        sub = copy.copy(board)
        sub.traces = [board.traces[i] for i in remaining]   # obstacles kept on every layer
        sp, sl, sf = route_all_traces(sub, [test_points[i] for i in remaining], **kwargs)
        layer_crossings[layer] = count_crossings(sp)
        still = []
        for k, i in enumerate(remaining):
            if sp[k] is None:
                still.append(i)
            else:
                paths[i] = sp[k]
                lengths[i] = sl[k]
                layer_of[i] = layer
        remaining = still
    return paths, lengths, layer_of, len(remaining), layer_crossings


def octile_lower_bounds(board, test_points):
    """Per-net lower bound on routed length (mm): no routing can be shorter."""
    lbs = []
    for i in range(min(len(board.traces), len(test_points))):
        t = board.traces[i]
        dx = abs(test_points[i][0] - t.start_x)
        dy = abs(test_points[i][1] - t.start_y)
        lbs.append(max(dx, dy) + 0.4142135624 * min(dx, dy) + t.breakout_length)
    return lbs


def optimize_layers_for_length(board, test_points, paths, lengths, layer_of,
                               max_layers=6, ratio=1.5, **kwargs):
    """Min-max-length layer reassignment; returns (paths, lengths, layer_of, moves).
    Only the receiving layer is re-routed (a subset of a legal layer is legal)."""
    import copy as _copy
    n = min(len(board.traces), len(test_points))
    paths = list(paths)
    lengths = list(lengths)
    layer_of = list(layer_of)
    lbs = octile_lower_bounds(board, test_points)
    moves = 0
    for _ in range(n):
        routed = [i for i in range(n) if paths[i] is not None]
        if not routed:
            break
        worst = max(routed, key=lambda i: lengths[i])
        cur_max = lengths[worst]
        if cur_max <= ratio * lbs[worst]:
            break                       # the max is inherent, not congestion
        used = sorted(set(l for l in layer_of if l >= 0))
        cand_layers = [L for L in used if L != layer_of[worst]]
        fresh = (max(used) + 1 if used else 0) if len(used) < max_layers else None
        if fresh is not None:
            cand_layers.append(fresh)
        # Prefer an EXISTING layer: a fresh layer always routes shortest but thin
        # layers are expensive; take one only when no existing layer reduces the max.
        best = None                # ((is_new, new_max), layer, sub_paths, sub_len, members)
        for L in cand_layers:
            members = [i for i in range(n) if layer_of[i] == L] + [worst]
            sub = _copy.copy(board)
            sub.traces = [board.traces[i] for i in members]
            sp, sl, sf = route_all_traces(
                sub, [test_points[i] for i in members], **kwargs)
            if sf:
                continue                # every affected net must stay routed
            trial = dict(zip(members, sl))
            new_max = max(trial.get(i, lengths[i]) for i in routed)
            if new_max >= cur_max - 1e-6:
                continue                # must strictly reduce the global max
            key = (L == fresh and L not in used, new_max)
            if best is None or key < best[0]:
                best = (key, L, sp, sl, members)
        if best is None:
            break                       # no move reduces the global maximum
        _key, L, sp, sl, members = best
        for k, i in enumerate(members):
            paths[i] = sp[k]
            lengths[i] = sl[k]
            layer_of[i] = L
        moves += 1

    # Consolidation: merge layers pairwise (highest into lowest) when the combined
    # set routes with no fails and without raising the global maximum.
    while True:
        used = sorted(set(l for l in layer_of if l >= 0))
        if len(used) < 2:
            break
        routed = [i for i in range(n) if paths[i] is not None]
        cur_max = max(lengths[i] for i in routed)
        merged = False
        for hi in reversed(used):
            for lo_ in used:
                if lo_ >= hi:
                    break
                members = [i for i in range(n) if layer_of[i] in (hi, lo_)]
                sub = _copy.copy(board)
                sub.traces = [board.traces[i] for i in members]
                sp, sl, sf = route_all_traces(
                    sub, [test_points[i] for i in members], **kwargs)
                if sf:
                    continue
                trial = dict(zip(members, sl))
                if max(trial.get(i, lengths[i]) for i in routed) > cur_max + 1e-6:
                    continue
                for k, i in enumerate(members):
                    paths[i] = sp[k]
                    lengths[i] = sl[k]
                    layer_of[i] = lo_
                merged = True
                break
            if merged:
                break
        if not merged:
            break
    # Renumber layers densely (0..k) after merges.
    used = sorted(set(l for l in layer_of if l >= 0))
    remap = {L: k for k, L in enumerate(used)}
    layer_of = [remap.get(l, -1) for l in layer_of]
    return paths, lengths, layer_of, moves


def route_to_length(board, test_points, paths_world, lengths, target_mm=None,
                    tol=1.5, margin=4.0, tries=8):
    """Bounded-detour rerouting toward the length target for nets the meander cannot
    finish; run AFTER routing, BEFORE equalize_lengths. Returns (paths, lengths, n_extended)."""
    n = min(len(board.traces), len(test_points))
    paths_world = list(paths_world)
    lengths = list(lengths)
    fin = [x for x in lengths if x < 1e9]
    if not fin:
        return paths_world, lengths, 0
    if target_mm is None:
        target_mm = max(fin)

    # Probe: which routed nets can the meander alone NOT bring to target?
    _eqp, eqL, _t, _m = equalize_lengths(board, paths_world,
                                         target_mm=target_mm - board.traces[0].breakout_length,
                                         test_points=test_points)
    unmatched = [i for i in range(n)
                 if paths_world[i] is not None and eqL[i] < target_mm - tol]
    if not unmatched:
        return paths_world, lengths, 0

    grid, rows, cols, blocked, cells, endpoints, owner = _build_context(
        board, test_points)
    res = grid.res

    def to_cells(p):
        cp = []
        for (x, y) in p:
            c, r = grid._world_to_grid(x, y)
            if not cp or cp[-1] != (r, c):
                cp.append((r, c))
        return cp

    routes = [to_cells(p) if p else None for p in paths_world]
    zero = np.zeros((rows, cols))
    n_ext = 0
    for i in sorted(unmatched, key=lambda j: eqL[j]):        # neediest first
        resc, hard, kept = _kept_blockers(routes, cells, blocked, skip=i)
        hard_corners = set()
        for hsq, ht in hard:
            hard_corners.update(_diag_corners(hsq, ht))
        (sr, sc), (er, ec) = cells[i]
        resc[sr, sc] = False
        resc[er, ec] = False
        want = (target_mm - margin - board.traces[i].breakout_length) / res
        # Waypoint candidates: free open cells whose pin->W->pad octile estimate
        # lands nearest the wanted length.
        cand = []
        for rr in range(1, rows - 1, 2):
            for cc in range(1, cols - 1, 2):
                if (not resc[rr, cc] and owner[rr, cc] < 0
                        and (rr, cc) not in hard_corners):
                    est = (max(abs(rr - sr), abs(cc - sc))
                           + 0.4142135624 * min(abs(rr - sr), abs(cc - sc))
                           + max(abs(rr - er), abs(cc - ec))
                           + 0.4142135624 * min(abs(rr - er), abs(cc - ec)))
                    if est <= want + tol / res:
                        cand.append((abs(est - want), (rr, cc)))
        cand.sort(key=lambda t_: t_[0])
        old_len = lengths[i]

        def attempt(ws):
            """Route pin -> waypoints -> pad, each leg hard-blocked against kept
            nets AND the previous legs; returns the full cell path or None."""
            resc_t = resc.copy()
            hard_t = set(hard)
            kept_t = set(kept)
            pts = [(sr, sc)] + list(ws) + [(er, ec)]
            full = None
            for a, bpt in zip(pts, pts[1:]):
                leg = _astar_cost(resc_t, zero, {}, {}, rows, cols, a, bpt,
                                  0.0, owner, i, True, hard_diag=hard_t,
                                  strict=True, kept_cells=kept_t)
                if not leg:
                    return None
                for (r, c) in leg[:-1]:              # leg end stays enterable
                    resc_t[r, c] = True
                    kept_t.add((r, c))
                for k2 in range(len(leg) - 1):
                    (r, c), (nr, nc) = leg[k2], leg[k2 + 1]
                    if (nr - r) and (nc - c):
                        hard_t.add(_diag_key(r, c, nr, nc))
                full = leg if full is None else full + leg[1:]
            return full

        def accept(full):
            nonlocal n_ext
            plen = sum(np.hypot(full[k + 1][0] - full[k][0],
                                full[k + 1][1] - full[k][1])
                       for k in range(len(full) - 1)) * res \
                + board.traces[i].breakout_length
            if plen > target_mm + tol or plen <= old_len + 1e-6:
                return False
            routes[i] = full
            paths_world[i] = [grid._grid_to_world(c, r) for (r, c) in full]
            lengths[i] = plen
            n_ext += 1
            return True

        done = False
        for _score, W in cand[:tries]:
            full = attempt([W])
            if full is not None and accept(full):
                done = True
                break
        if not done and len(cand) >= 2:
            # Two-waypoint S-detours when no single waypoint fits.
            def octd_c(a, bpt):
                dr, dc = abs(a[0] - bpt[0]), abs(a[1] - bpt[1])
                return max(dr, dc) + 0.4142135624 * min(dr, dc)
            pool = [w for _s, w in cand[:6]]
            pairs = []
            for wi in range(len(pool)):
                for wj in range(len(pool)):
                    if wi == wj:
                        continue
                    w1, w2 = pool[wi], pool[wj]
                    est = (octd_c((sr, sc), w1) + octd_c(w1, w2)
                           + octd_c(w2, (er, ec)))
                    if est <= want + tol / res:
                        pairs.append((abs(est - want), (w1, w2)))
            pairs.sort(key=lambda t_: t_[0])
            for _score, (w1, w2) in pairs[:tries]:
                full = attempt([w1, w2])
                if full is not None and accept(full):
                    break
    return paths_world, lengths, n_ext


def equalize_lengths(board, paths_world, passes=24, tol=1.0, target_mm=None,
                     test_points=None):
    """Length matching, run strictly AFTER all routing: meander shorter traces toward
    the target; clearance- and keep-out-safe by construction. Pass `test_points` so
    unrouted nets' pads are protected too. Returns (paths, lengths, target_mm, n_matched)."""
    grid = RoutingGrid(board)
    rows, cols, res = grid.rows, grid.cols, grid.res
    blocked = grid.obstacle_grid > 0

    # World paths -> dedup'd (row, col) cell paths.
    cell_paths = []
    for p in paths_world:
        if p is None:
            cell_paths.append(None)
            continue
        cp = []
        for (x, y) in p:
            c, r = grid._world_to_grid(x, y)
            if not cp or cp[-1] != (r, c):
                cp.append((r, c))
        cell_paths.append(cp)

    occ = set()
    for cp in cell_paths:
        if cp:
            occ.update(cp)

    # Full-pitch guard: a tooth cell must not sit on a foreign diagonal's
    # complementary corner (exactly pitch/sqrt(2) from the diagonal).
    corner_owner: Dict[Tuple[int, int], Set[int]] = {}
    for i2, cp in enumerate(cell_paths):
        if not cp:
            continue
        for k in range(len(cp) - 1):
            (r, c), (nr, nc) = cp[k], cp[k + 1]
            if (nr - r) and (nc - c):
                for cc in _diag_corners(*_diag_key(r, c, nr, nc)):
                    corner_owner.setdefault(cc, set()).add(i2)

    # A meander tooth may never enter another net's pad keep-out disk.
    pad_owner: Dict[Tuple[int, int], Set[int]] = {}
    rad = TP_CLEARANCE_CELLS
    R_pad = int(np.ceil(rad))
    pads = []
    if test_points is not None:
        pads = [(i2, tp) for i2, tp in enumerate(test_points[:len(cell_paths)])
                if tp is not None]
    else:
        pads = [(i2, p[-1]) for i2, p in enumerate(paths_world) if p]
    for i2, (px, py) in pads:
        pc, pr = grid._world_to_grid(px, py)
        for dr in range(-R_pad, R_pad + 1):
            for dc in range(-R_pad, R_pad + 1):
                if dr * dr + dc * dc <= rad * rad:
                    rr, cc = pr + dr, pc + dc
                    if 0 <= rr < rows and 0 <= cc < cols:
                        pad_owner.setdefault((rr, cc), set()).add(i2)

    def _tooth_ok(cell, net):
        own = corner_owner.get(cell)
        if own and not own <= {net}:
            return False
        pads_here = pad_owner.get(cell)
        return not pads_here or pads_here <= {net}

    def plen(p):
        return sum(np.hypot(p[k + 1][0] - p[k][0], p[k + 1][1] - p[k][1])
                   for k in range(len(p) - 1))

    routed = [p for p in cell_paths if p]
    if not routed:
        return paths_world, [float('inf')] * len(paths_world), 0.0, 0
    target = (target_mm / res) if target_mm is not None else max(plen(p) for p in routed)

    _STAIR = 2.0 - float(np.sqrt(2))            # length gained per L-step

    def extend_net(i, tgt):
        """Grow net i toward `tgt` (cells): serpentine teeth, then staircase
        densification of diagonal steps. Returns True if the net grew."""
        p = cell_paths[i]
        need = tgt - plen(p)                    # cells of length still to add
        if need <= tol:
            return False
        added_total = 0.0
        for pass_no in range(passes):
            if added_total >= need - tol:
                break
            # Cap tooth depth per pass so length distributes as accordion
            # meanders instead of a few deep first-fit combs.
            depth_cap = min(pass_no + 1, _MAX_BUMP_DEPTH)
            new = [p[0]]
            progressed = False
            endpt, startpt = p[-1], p[0]
            keep_e = TP_CLEARANCE_CELLS + 1     # clean approach; never wrap a pad
            keep_s = 2                          # the pin needs no pad-size moat
            for k in range(len(p) - 1):
                A, N = p[k], p[k + 1]
                dr, dc = N[0] - A[0], N[1] - A[1]
                near_end = (max(abs(A[0] - endpt[0]), abs(A[1] - endpt[1])) <= keep_e or
                            max(abs(N[0] - endpt[0]), abs(N[1] - endpt[1])) <= keep_e or
                            max(abs(A[0] - startpt[0]), abs(A[1] - startpt[1])) <= keep_s or
                            max(abs(N[0] - startpt[0]), abs(N[1] - startpt[1])) <= keep_s)
                if added_total < need - tol and (dr == 0) != (dc == 0) and not near_end:
                    perps = [(1, 0), (-1, 0)] if dr == 0 else [(0, 1), (0, -1)]
                    for pr, pc in perps:
                        # Deepest free excursion; each unit of depth adds 2 cells of length.
                        D = 0
                        while D < depth_cap:
                            t = D + 1
                            B = (A[0] + t * pr, A[1] + t * pc)
                            C = (N[0] + t * pr, N[1] + t * pc)
                            if (0 <= B[0] < rows and 0 <= B[1] < cols and
                                    0 <= C[0] < rows and 0 <= C[1] < cols and
                                    not blocked[B] and not blocked[C] and
                                    B not in occ and C not in occ and
                                    _tooth_ok(B, i) and _tooth_ok(C, i)):
                                D = t
                            else:
                                break
                        if D >= 1 and (need - added_total) >= 2.0:
                            # FLOOR the depth; never overshoot with teeth, or the
                            # re-base rounds chase their own tail; the staircase finishes.
                            d = min(D, max(1, int((need - added_total) // 2)))
                            tooth = ([(A[0] + t * pr, A[1] + t * pc) for t in range(1, d + 1)] +
                                     [(N[0] + t * pr, N[1] + t * pc) for t in range(d, 0, -1)])
                            if tooth[0] != new[-1]:
                                new += tooth
                                for cc in tooth:
                                    occ.add(cc)
                                added_total += 2.0 * d
                                progressed = True
                                break
                new.append(N)
            p = new
            if not progressed:
                break
        # Staircase densification: 0.586-cell top-up; the L stays inside its own
        # unit square, so it can never wrap a pad or leave the corridor.
        if added_total < need - tol:
            new = [p[0]]
            for k in range(len(p) - 1):
                A, N = p[k], p[k + 1]
                if (added_total < need - tol
                        and A[0] != N[0] and A[1] != N[1]
                        and abs(A[0] - N[0]) == 1 and abs(A[1] - N[1]) == 1):
                    for corner in ((A[0], N[1]), (N[0], A[1])):
                        if (0 <= corner[0] < rows and 0 <= corner[1] < cols and
                                not blocked[corner] and corner not in occ and
                                _tooth_ok(corner, i) and corner != new[-1]):
                            new.append(corner)
                            occ.add(corner)
                            added_total += _STAIR
                            break
                new.append(N)
            p = new
        cell_paths[i] = p
        return added_total > 0

    # RE-BASE: teeth can overshoot, raising the true common maximum; re-extend
    # stragglers against the FINAL max until stable.
    for _round in range(4):
        target = max(target, max(plen(p) for p in cell_paths if p))
        grew = False
        for i in range(len(cell_paths)):
            if cell_paths[i]:
                grew |= extend_net(i, target)
        final = max(plen(p) for p in cell_paths if p)
        if not grew or all(plen(p) >= final - tol
                           for p in cell_paths if p):
            target = max(target, final)
            break

    target = max(target, max(plen(p) for p in cell_paths if p))
    new_paths = [[grid._grid_to_world(c, r) for (r, c) in p] if p else None
                 for p in cell_paths]
    new_lengths = [plen(p) * res + board.traces[i].breakout_length if p else float('inf')
                   for i, p in enumerate(cell_paths)]
    n_matched = sum(1 for p in cell_paths if p and plen(p) >= target - tol)
    return new_paths, new_lengths, target * res, n_matched


def _resample_path(arr: np.ndarray, n: int = 200) -> np.ndarray:
    """Uniformly resample a path to at most n points (keeps endpoints).
    Uniform sampling, unlike strided slicing, never drops the final point."""
    if len(arr) <= n:
        return arr
    idx = np.linspace(0, len(arr) - 1, n).round().astype(int)
    return arr[idx]


def _ccw(ax, ay, bx, by, cx, cy):
    return (cy - ay) * (bx - ax) - (by - ay) * (cx - ax)


# Orientation deadband (mm^2): without it, float noise (~1e-14) classifies
# collinear 45-deg segments as phantom "proper crossings", dropping healthy nets.
_CCW_TOL = 1e-6


def _ccw_sign(d):
    """-1 / 0 / +1 with the collinearity deadband (works on scalars & arrays)."""
    return (d > _CCW_TOL) * 1 + (d < -_CCW_TOL) * -1


def _seg_arrays(paths):
    """Flatten polylines into segment arrays: (S, N) with S[:, 0:4] = x1,y1,x2,y2
    and N the owning net index per segment. Returns (None, None) if no segments."""
    segs, nets = [], []
    for idx, p in enumerate(paths):
        if p is not None and len(p) >= 2:
            a = np.asarray(p, dtype=np.float64)
            segs.append(np.concatenate([a[:-1], a[1:]], axis=1))
            nets.append(np.full(len(a) - 1, idx, dtype=np.int64))
    if not segs:
        return None, None
    return np.concatenate(segs), np.concatenate(nets)


def _crossing_pairs(paths):
    """Net pairs whose polylines properly cross (touching endpoints and collinear
    segments do not count). Vectorized in chunks; the pure-Python O(segments^2)
    loop cost seconds per call."""
    S, N = _seg_arrays(paths)
    if S is None:
        return set()
    n = len(S)
    Ax, Ay, Bx, By = S[:, 0], S[:, 1], S[:, 2], S[:, 3]
    idx = np.arange(n)
    pairs = set()
    for lo in range(0, n, 256):
        hi = min(lo + 256, n)
        ax, ay = Ax[lo:hi, None], Ay[lo:hi, None]
        bx, by = Bx[lo:hi, None], By[lo:hi, None]
        cx, cy, dx, dy = Ax[None, :], Ay[None, :], Bx[None, :], By[None, :]
        s1 = _ccw_sign(_ccw(cx, cy, dx, dy, ax, ay))
        s2 = _ccw_sign(_ccw(cx, cy, dx, dy, bx, by))
        s3 = _ccw_sign(_ccw(ax, ay, bx, by, cx, cy))
        s4 = _ccw_sign(_ccw(ax, ay, bx, by, dx, dy))
        cross = (s1 * s2 < 0) & (s3 * s4 < 0)
        cross &= N[lo:hi, None] != N[None, :]        # different nets only
        cross &= idx[None, :] > idx[lo:hi, None]     # upper triangle (no dupes)
        for r, c in zip(*np.where(cross)):
            i, j = int(N[lo + r]), int(N[c])
            pairs.add((min(i, j), max(i, j)))
    return pairs


def count_crossings(paths) -> int:
    """Number of trace pairs that properly cross; a valid single-layer routing has zero."""
    return len(_crossing_pairs(paths))


def _seg_seg_dist(A, B, C, D):
    """Exact minimum distance between 2-D segments AB and CD (closest points)."""
    A, B, C, D = (np.asarray(p, float) for p in (A, B, C, D))
    d1, d2, r = B - A, D - C, A - C
    a, e, f = d1 @ d1, d2 @ d2, d2 @ r
    cl = lambda x: max(0.0, min(1.0, x))
    if a <= 1e-12 and e <= 1e-12:
        return float(np.hypot(*(A - C)))
    if a <= 1e-12:
        s, t = 0.0, cl(f / e)
    else:
        c = d1 @ r
        if e <= 1e-12:
            t, s = 0.0, cl(-c / a)
        else:
            b = d1 @ d2
            den = a * e - b * b
            s = cl((b * f - c * e) / den) if den > 1e-12 else 0.0
            t = (b * s + f) / e
            if t < 0:
                t, s = 0.0, cl(-c / a)
            elif t > 1:
                t, s = 1.0, cl((b - c) / a)
    return float(np.hypot(*((A + d1 * s) - (C + d2 * t))))


def _seg_obstacle_clear(grid, A, B) -> bool:
    """True if straight segment A->B stays out of all (inflated) obstacle cells."""
    n = max(2, int(np.hypot(B[0] - A[0], B[1] - A[1]) / (0.3 * grid.res)))
    for tt in np.linspace(0.0, 1.0, n):
        c, r = grid._world_to_grid(A[0] + (B[0] - A[0]) * tt, A[1] + (B[1] - A[1]) * tt)
        if grid.obstacle_grid[r, c]:
            return False
    return True


def min_trace_separation(paths) -> float:
    """Exact min centerline distance between different nets' segments (mm); catches
    the sub-pitch parallel-diagonal case that a resampled point check misses."""
    S, N = _seg_arrays(paths)
    if S is None or len(S) < 2:
        return float('inf')
    n = len(S)
    A, B = S[:, 0:2], S[:, 2:4]
    D1 = B - A
    idx = np.arange(n)
    eps = 1e-12
    m = float('inf')

    def cl(x):
        return np.clip(x, 0.0, 1.0)

    for lo in range(0, n, 256):
        hi = min(lo + 256, n)
        A1, d1 = A[lo:hi, None, :], D1[lo:hi, None, :]     # (m,1,2)
        C2, d2 = A[None, :, :], D1[None, :, :]             # (1,n,2)
        r = A1 - C2
        a = (d1 * d1).sum(-1)
        e = (d2 * d2).sum(-1)
        f = (d2 * r).sum(-1)
        c_ = (d1 * r).sum(-1)
        b = (d1 * d2).sum(-1)
        den = a * e - b * b
        s = np.where(den > eps, cl((b * f - c_ * e) / np.where(den > eps, den, 1.0)), 0.0)
        t = np.where(e > eps, (b * s + f) / np.where(e > eps, e, 1.0), 0.0)
        # clamp t, then recompute s for the clamped rows (Ericson's algorithm)
        s = np.where(t < 0, cl(-c_ / np.where(a > eps, a, 1.0)), s)
        s = np.where(t > 1, cl((b - c_) / np.where(a > eps, a, 1.0)), s)
        t = cl(t)
        # degenerate segments (points); same branches as _seg_seg_dist
        s = np.where(a <= eps, 0.0, s)
        t = np.where((a <= eps) & (e > eps), cl(f / np.where(e > eps, e, 1.0)), t)
        t = np.where(e <= eps, 0.0, t)
        s = np.where((e <= eps) & (a > eps), cl(-c_ / np.where(a > eps, a, 1.0)), s)
        diff = (A1 + d1 * s[..., None]) - (C2 + d2 * t[..., None])
        dist = np.sqrt((diff * diff).sum(-1))
        valid = (N[lo:hi, None] != N[None, :]) & (idx[None, :] > idx[lo:hi, None])
        if valid.any():
            m = min(m, float(dist[valid].min()))
    return m


def sub_clearance_pairs(paths, threshold: float) -> Set[Tuple[int, int]]:
    """Net pairs whose exact centerline separation is below `threshold` mm."""
    S, N = _seg_arrays(paths)
    if S is None or len(S) < 2:
        return set()
    n = len(S)
    A, B = S[:, 0:2], S[:, 2:4]
    D1 = B - A
    idx = np.arange(n)
    eps = 1e-12
    pairs: Set[Tuple[int, int]] = set()

    def cl(x):
        return np.clip(x, 0.0, 1.0)

    for lo in range(0, n, 256):
        hi = min(lo + 256, n)
        A1, d1 = A[lo:hi, None, :], D1[lo:hi, None, :]
        C2, d2 = A[None, :, :], D1[None, :, :]
        r = A1 - C2
        a = (d1 * d1).sum(-1)
        e = (d2 * d2).sum(-1)
        f = (d2 * r).sum(-1)
        c_ = (d1 * r).sum(-1)
        b = (d1 * d2).sum(-1)
        den = a * e - b * b
        s = np.where(den > eps, cl((b * f - c_ * e) / np.where(den > eps, den, 1.0)), 0.0)
        t = np.where(e > eps, (b * s + f) / np.where(e > eps, e, 1.0), 0.0)
        s = np.where(t < 0, cl(-c_ / np.where(a > eps, a, 1.0)), s)
        s = np.where(t > 1, cl((b - c_) / np.where(a > eps, a, 1.0)), s)
        t = cl(t)
        s = np.where(a <= eps, 0.0, s)
        t = np.where((a <= eps) & (e > eps), cl(f / np.where(e > eps, e, 1.0)), t)
        t = np.where(e <= eps, 0.0, t)
        s = np.where((e <= eps) & (a > eps), cl(-c_ / np.where(a > eps, a, 1.0)), s)
        diff = (A1 + d1 * s[..., None]) - (C2 + d2 * t[..., None])
        dist = np.sqrt((diff * diff).sum(-1))
        hit = (N[lo:hi, None] != N[None, :]) & (idx[None, :] > idx[lo:hi, None]) \
            & (dist < threshold)
        for rr, cc in zip(*np.where(hit)):
            i, j = int(N[lo + rr]), int(N[cc])
            pairs.add((min(i, j), max(i, j)))
    return pairs


def validate_routing_constraints(
    board: BoardSpec,
    paths: List[Optional[List[Tuple[float, float]]]],
) -> dict:
    """Check all hard constraints on routed traces."""
    violations = []
    t2t_min = float('inf')
    t2e_min = float('inf')
    t2o_min = float('inf')
    hw = TRACE_WIDTH / 2

    vp, vi = [], []
    for i, p in enumerate(paths):
        if p is not None:
            vp.append(np.array(p))
            vi.append(i)

    for idx, pts in zip(vi, vp):
        de = min((pts[:, 0] - board.x_min).min() - hw,
                 (board.x_max - pts[:, 0]).min() - hw,
                 (pts[:, 1] - board.y_min).min() - hw,
                 (board.y_max - pts[:, 1]).min() - hw)
        t2e_min = min(t2e_min, de)
        if de < TRACE_TO_EDGE_MIN - CELL_SIZE:
            violations.append(("trace_to_edge", idx, None,
                               f"Trace {idx}: edge dist {de:.3f}mm"))

    for idx, pts in zip(vi, vp):
        for obs in board.rect_obstacles:
            xn, yn, xx, yx = obs.bounds
            dx = np.maximum(np.maximum(xn - pts[:, 0], 0), pts[:, 0] - xx)
            dy = np.maximum(np.maximum(yn - pts[:, 1], 0), pts[:, 1] - yx)
            d = np.where((dx > 0) | (dy > 0), np.hypot(dx, dy), 0.0) - hw
            dm = float(d.min())
            t2o_min = min(t2o_min, dm)
            if dm < obs.clearance - CELL_SIZE:
                violations.append(("trace_to_obstacle", idx, obs.name,
                                   f"Trace {idx}: {obs.name} dist {dm:.3f}mm"))
        for obs in board.circ_obstacles:
            d = np.hypot(pts[:, 0] - obs.cx, pts[:, 1] - obs.cy) - obs.radius - hw
            dm = float(d.min())
            t2o_min = min(t2o_min, dm)
            if dm < obs.clearance - CELL_SIZE:
                violations.append(("trace_to_obstacle", idx, obs.name,
                                   f"Trace {idx}: {obs.name} dist {dm:.3f}mm"))

    # Point-sampled edge-to-edge estimate; report metric ONLY; the authoritative
    # trace-to-trace check is the exact centerline separation below.
    for a in range(len(vp)):
        for b in range(a + 1, len(vp)):
            sa = _resample_path(vp[a])
            sb = _resample_path(vp[b])
            d = np.sqrt(((sa[:, None, :] - sb[None, :, :]) ** 2).sum(2))
            mcc = float(d.min())
            mee = mcc - TRACE_WIDTH
            t2t_min = min(t2t_min, mee)

    crossings = count_crossings(paths)

    # Pad keep-out check: exact point-to-SEGMENT distance (vertex-only sampling
    # under-measures on equalized / any-angle paths).
    def _pt_polyline(pt, arr):
        if len(arr) < 2:
            return float(np.hypot(arr[0, 0] - pt[0], arr[0, 1] - pt[1]))
        A, B = arr[:-1], arr[1:]
        d = B - A
        L2 = (d * d).sum(1)
        t = np.clip(((pt - A) * d).sum(1) / np.where(L2 > 1e-12, L2, 1.0), 0.0, 1.0)
        proj = A + d * t[:, None]
        return float(np.sqrt(((proj - pt) ** 2).sum(1)).min())

    tp_keepout_mm = TP_CLEARANCE_CELLS * CELL_SIZE
    tp2trace_min = float('inf')
    for i, pi in enumerate(paths):
        if not pi:
            continue
        tp = np.asarray(pi[-1], dtype=float)
        for j, pj in enumerate(paths):
            if pj is None or j == i:
                continue
            dj = _pt_polyline(tp, np.asarray(pj, dtype=float))
            tp2trace_min = min(tp2trace_min, dj)
            if dj < tp_keepout_mm - CELL_SIZE - 0.05:
                violations.append(("tp_keepout", i, j,
                                   f"Pad {i}: net {j} at {dj:.2f}mm "
                                   f"(keep-out {tp_keepout_mm:.2f}mm)"))
    tp_clearance_ok = tp2trace_min >= tp_keepout_mm - CELL_SIZE - 0.05

    # Exact centerline separation; the true min; the resampled loop above can
    # miss sub-pitch spacing such as parallel 45s one lane apart.
    t2t_center_min = min_trace_separation(paths)
    clearance_ok = t2t_center_min >= CELL_SIZE - 0.05   # >= trace pitch
    for i, j in sorted(sub_clearance_pairs(paths, CELL_SIZE - 0.05)):
        violations.append(("trace_to_trace", i, j,
                           f"Traces {i},{j}: centerline below pitch "
                           f"({CELL_SIZE:.4f}mm)"))

    return {"violations": violations, "trace_to_trace_min": t2t_min,
            "trace_to_trace_center_min": t2t_center_min, "clearance_ok": clearance_ok,
            "trace_to_edge_min": t2e_min, "trace_to_obstacle_min": t2o_min,
            "crossings": crossings, "tp_to_trace_min": tp2trace_min,
            "tp_clearance_ok": tp_clearance_ok,
            "all_valid": len(violations) == 0 and crossings == 0}


def any_angle_shortcut(paths, board, clearance: float = None):
    """String-pull octilinear routes into any-angle routes; every shortcut is verified
    to clear obstacles and keep >= `clearance` from other nets, so it can never
    introduce a spacing violation. Returns new paths_world."""
    if clearance is None:
        clearance = CELL_SIZE
    grid = RoutingGrid(board)
    out = [list(p) if p else None for p in paths]
    for i in range(len(out)):
        p = out[i]
        if not p or len(p) < 3:
            continue
        new = [p[0]]
        k = 0
        while k < len(p) - 1:
            best = k + 1
            for m in range(len(p) - 1, k + 1, -1):       # farthest reachable first
                A, B = p[k], p[m]
                if not _seg_obstacle_clear(grid, A, B):
                    continue
                ok = True
                for j in range(len(out)):
                    if j == i or not out[j]:
                        continue
                    q = out[j]
                    for t in range(len(q) - 1):
                        if _seg_seg_dist(A, B, q[t], q[t + 1]) < clearance - 1e-9:
                            ok = False
                            break
                    if not ok:
                        break
                if ok:
                    best = m
                    break
            new.append(p[best])
            k = best
        out[i] = new
    return out