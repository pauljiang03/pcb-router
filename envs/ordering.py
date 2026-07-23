"""Learned net-ordering: a tiny numpy-only MLP predicts a routing order,
distilled from the multi-start search's best orders (models/ordering.npz)."""
import pathlib
from typing import List, Optional

import numpy as np

_DEFAULT = pathlib.Path(__file__).resolve().parents[1] / "models" / "ordering.npz"


def net_features(board, test_points) -> np.ndarray:
    """(n, 8) per-net geometric/congestion features."""
    from envs.board import wire_estimate
    from envs.routing import _ccw, _ccw_sign
    n = min(len(board.traces), len(test_points))
    diag = float(np.hypot(board.width, board.height))
    segs = [(board.traces[i].start_x, board.traces[i].start_y,
             test_points[i][0], test_points[i][1]) for i in range(n)]

    def xings(i):
        ax, ay, bx, by = segs[i]
        c = 0
        for j in range(n):
            if j == i:
                continue
            cx, cy, dx, dy = segs[j]
            s1 = _ccw_sign(_ccw(cx, cy, dx, dy, ax, ay))
            s2 = _ccw_sign(_ccw(cx, cy, dx, dy, bx, by))
            s3 = _ccw_sign(_ccw(ax, ay, bx, by, cx, cy))
            s4 = _ccw_sign(_ccw(ax, ay, bx, by, dx, dy))
            c += (s1 * s2 < 0) and (s3 * s4 < 0)
        return c

    feats = []
    for i in range(n):
        t = board.traces[i]
        px, py = test_points[i]
        d = np.hypot(px - t.start_x, py - t.start_y)
        w = wire_estimate(board, t, (px, py))
        feats.append([
            (t.start_x - board.x_min) / board.width,
            (t.start_y - board.y_min) / board.height,
            (px - board.x_min) / board.width,
            (py - board.y_min) / board.height,
            d / diag,
            np.arctan2(py - t.start_y, px - t.start_x) / np.pi,
            (w - d) / diag,                      # wrap surcharge
            xings(i) / max(n - 1, 1),            # direct-crossing congestion
        ])
    return np.asarray(feats, dtype=np.float64)


def load_model(path: Optional[str] = None):
    """Load ordering.npz -> dict of weights, or None if not trained yet."""
    p = pathlib.Path(path) if path else _DEFAULT
    if not p.exists():
        return None
    z = np.load(p)
    return {k: z[k] for k in z.files}


def predict_order(board, test_points, model=None,
                  path: Optional[str] = None) -> Optional[List[int]]:
    """Routing order (first-routed first) from the learned scorer, or None if no model."""
    if model is None:
        model = load_model(path)
    if model is None:
        return None
    x = (net_features(board, test_points) - model["mu"]) / model["sd"]
    h = np.maximum(x @ model["W1"] + model["b1"], 0.0)
    h = np.maximum(h @ model["W2"] + model["b2"], 0.0)
    s = (h @ model["W3"] + model["b3"]).ravel()
    return list(np.argsort(-s))


def route_fast(board, test_points, model=None, hint_starts=3, **kwargs):
    """Route with the predicted order first; fall back to full multi-start if any net fails."""
    from envs.routing import route_all_traces
    od = predict_order(board, test_points, model=model)
    if od is not None:
        probe = dict(kwargs)
        probe["n_starts"] = min(hint_starts, kwargs.get("n_starts", 6))
        probe["repair_passes"] = min(1, kwargs.get("repair_passes", 3))
        paths, lengths, fails = route_all_traces(
            board, test_points, order_hint=od, **probe)
        if fails == 0:
            return paths, lengths, fails
    return route_all_traces(board, test_points, order_hint=od, **kwargs)
