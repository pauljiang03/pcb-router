"""2D visualization of PCB boards with test points and routed traces."""

import numpy as np
from typing import List, Tuple, Optional
from envs.board import BoardSpec, TP_TO_TP_MIN


def render_board_png(board: BoardSpec, test_points=None, paths=None,
                     filename: str = "board.png", scale: int = 6,
                     title: Optional[str] = None, labels: bool = False,
                     keepout_mm: Optional[float] = None,
                     path_layers=None, legend: bool = False) -> str:
    """Render board, test points and routed traces to a PNG using PIL; returns filename."""
    from PIL import Image, ImageDraw
    import colorsys

    pad = 24
    head = 28 if title else 10
    _has_inner = bool(path_layers) and any(l >= 1 for l in path_layers)
    _will_unrouted = bool(paths) and any(
        p is None and test_points is not None and i < len(test_points)
        and test_points[i] is not None
        for i, p in enumerate(paths))
    foot = (96 if _has_inner else 74) if legend else 10
    if legend and _will_unrouted:
        foot += 22
    W = int(board.width * scale)
    H = int(board.height * scale)
    img = Image.new("RGB", (W + 2 * pad, H + head + foot), "white")
    d = ImageDraw.Draw(img)

    def wx(x):
        return pad + (x - board.x_min) * scale

    def wy(y):  # flip vertical so +y is up
        return head + (board.height - (y - board.y_min)) * scale

    d.rectangle([wx(board.x_min), wy(board.y_max), wx(board.x_max), wy(board.y_min)],
                outline="black", width=2)
    for o in board.rect_obstacles:
        xn, yn, xx, yx = o.bounds
        d.rectangle([wx(xn), wy(yx), wx(xx), wy(yn)], fill=(255, 175, 175),
                    outline=(190, 0, 0))
    for o in board.circ_obstacles:
        d.ellipse([wx(o.cx - o.radius), wy(o.cy + o.radius),
                   wx(o.cx + o.radius), wy(o.cy - o.radius)],
                  fill=(255, 150, 150), outline=(150, 0, 0))
    if board.connector_w > 0:
        d.rectangle([wx(board.connector_x), wy(board.connector_y + board.connector_h),
                     wx(board.connector_x + board.connector_w), wy(board.connector_y)],
                    outline=(150, 0, 190), width=2)

    n = max(len(board.traces), 1)
    colors = [tuple(int(255 * c) for c in colorsys.hsv_to_rgb(i / n, 0.85, 0.9))
              for i in range(n)]

    def dotted(p0, p1, col):
        seg = ((p1[0] - p0[0]) ** 2 + (p1[1] - p0[1]) ** 2) ** 0.5
        if seg < 1:
            return
        for j in range(int(seg // 7) + 1):
            t = j * 7 / seg
            x, y = p0[0] + (p1[0] - p0[0]) * t, p0[1] + (p1[1] - p0[1]) * t
            d.ellipse([x - 1.5, y - 1.5, x + 1.5, y + 1.5], fill=col)

    # Inner-layer paths drawn dashed; unrouted nets drawn as dotted pin->pad lines.
    _has_unrouted = False
    if paths:
        for i, p in enumerate(paths):
            if not p:
                if (test_points is not None and i < len(test_points)
                        and test_points[i] is not None and i < len(board.traces)):
                    t = board.traces[i]
                    dotted((wx(t.start_x), wy(t.start_y)),
                           (wx(test_points[i][0]), wy(test_points[i][1])),
                           colors[i % n])
                    _has_unrouted = True
                continue
            pts = [(wx(x), wy(y)) for x, y in p]
            col = colors[i % n]
            if path_layers and i < len(path_layers) and path_layers[i] >= 1:
                for k in range(len(pts) - 1):
                    (x0, y0), (x1, y1) = pts[k], pts[k + 1]
                    seg = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
                    if seg < 1:
                        continue
                    for j in range(int(seg // 8) + 1):
                        t0 = j * 8 / seg
                        t1 = min((j * 8 + 5) / seg, 1.0)
                        d.line([(x0 + (x1 - x0) * t0, y0 + (y1 - y0) * t0),
                                (x0 + (x1 - x0) * t1, y0 + (y1 - y0) * t1)], fill=col, width=2)
            else:
                d.line(pts, fill=col, width=2)
    for t in board.traces:
        x, y = wx(t.start_x), wy(t.start_y)
        d.rectangle([x - 3, y - 3, x + 3, y + 3], fill="black")
    if test_points:
        for i, (tx, ty) in enumerate(test_points):
            x, y = wx(tx), wy(ty)
            if keepout_mm:
                rr = keepout_mm * scale
                d.ellipse([x - rr, y - rr, x + rr, y + rr], outline=(110, 110, 110))
            d.ellipse([x - 5, y - 5, x + 5, y + 5], fill=colors[i % n], outline="black")
            if labels:
                d.text((x + 6, y - 6), str(i), fill="black")
    if title:
        d.text((pad, 9), title, fill="black")
    if legend:
        ly = head + H + 18
        d.rectangle([pad, ly - 4, pad + 8, ly + 4], fill="black")
        d.text((pad + 14, ly - 5), "start (connector pin)", fill="black")
        d.ellipse([pad + 200, ly - 5, pad + 210, ly + 5], fill=(0, 150, 200), outline="black")
        d.text((pad + 216, ly - 5), "test point / pad (numbered)", fill="black")
        d.line([pad + 430, ly, pad + 460, ly], fill=(0, 150, 200), width=2)
        d.text((pad + 466, ly - 5), "routed trace", fill="black")
        ly2 = ly + 22
        d.rectangle([pad, ly2 - 4, pad + 8, ly2 + 4], fill=(255, 175, 175), outline=(190, 0, 0))
        d.text((pad + 14, ly2 - 5), "obstacle / keep-out zone", fill="black")
        d.rectangle([pad + 200, ly2 - 4, pad + 208, ly2 + 4], outline=(150, 0, 190))
        d.text((pad + 216, ly2 - 5), "connector outline", fill="black")
        if keepout_mm:
            d.ellipse([pad + 430, ly2 - 6, pad + 442, ly2 + 6], outline=(110, 110, 110))
            d.text((pad + 448, ly2 - 5), "pad keep-out radius", fill="black")
        if _has_inner:
            ly3 = ly2 + 22
            for sx in range(0, 30, 9):
                d.line([pad + sx, ly3, pad + min(sx + 5, 30), ly3], fill=(0, 150, 200), width=2)
            d.text((pad + 36, ly3 - 5),
                   "dashed = inner layer (2nd copper; inter-layer crossings are vias)", fill="black")
        if _has_unrouted:
            ly4 = (ly2 + 44) if _has_inner else (ly2 + 22)
            for sx in range(0, 30, 7):
                d.ellipse([pad + sx - 1.5, ly4 - 1.5, pad + sx + 1.5, ly4 + 1.5],
                          fill=(0, 150, 200))
            d.text((pad + 36, ly4 - 5),
                   "dotted = UNROUTED net (direct pin->pad line)", fill="black")
    img.save(filename)
    return filename


def plot_board(
    board: BoardSpec,
    test_points: Optional[List[Tuple[float, float]]] = None,
    paths: Optional[List[Optional[List[Tuple[float, float]]]]] = None,
    candidates: Optional[np.ndarray] = None,
    candidate_mask: Optional[np.ndarray] = None,
    title: str = "PCB Test Point Placement",
    filename: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 10),
):
    """Plot the board with obstacles, starting points, test points, and traces."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    board_rect = patches.Rectangle(
        (board.x_min, board.y_min), board.width, board.height,
        linewidth=2, edgecolor='black', facecolor='#f5f5f5', zorder=0,
    )
    ax.add_patch(board_rect)

    from envs.board import TP_TO_EDGE_MIN
    inner_rect = patches.Rectangle(
        (board.x_min + TP_TO_EDGE_MIN, board.y_min + TP_TO_EDGE_MIN),
        board.width - 2 * TP_TO_EDGE_MIN,
        board.height - 2 * TP_TO_EDGE_MIN,
        linewidth=1, edgecolor='gray', facecolor='none',
        linestyle='--', zorder=1, label='TP edge clearance',
    )
    ax.add_patch(inner_rect)

    if board.connector_w > 0:
        conn_rect = patches.Rectangle(
            (board.connector_x, board.connector_y),
            board.connector_w, board.connector_h,
            linewidth=1.5, edgecolor='purple', facecolor='#e8d5f5',
            alpha=0.5, zorder=2, label='Connector',
        )
        ax.add_patch(conn_rect)

    for obs in board.rect_obstacles:
        xmin, ymin, xmax, ymax = obs.bounds
        rect = patches.Rectangle(
            (xmin, ymin), xmax - xmin, ymax - ymin,
            linewidth=1, edgecolor='red', facecolor='#ffcccc',
            alpha=0.7, zorder=3,
        )
        ax.add_patch(rect)
        ax.text(obs.cx, obs.cy, obs.name, fontsize=6, ha='center', va='center',
                color='red', zorder=10)

    for obs in board.circ_obstacles:
        circle = patches.Circle(
            (obs.cx, obs.cy), obs.radius,
            linewidth=1, edgecolor='darkred', facecolor='#ff9999',
            alpha=0.7, zorder=3,
        )
        ax.add_patch(circle)
        ax.text(obs.cx, obs.cy - obs.radius - 0.5, obs.name,
                fontsize=6, ha='center', color='darkred', zorder=10)

    if candidates is not None:
        if candidate_mask is not None:
            valid = candidates[candidate_mask[:len(candidates)] > 0]
            invalid = candidates[candidate_mask[:len(candidates)] == 0]
            if len(invalid) > 0:
                ax.scatter(invalid[:, 0], invalid[:, 1], c='lightgray',
                          s=8, marker='.', zorder=2, alpha=0.5)
            if len(valid) > 0:
                ax.scatter(valid[:, 0], valid[:, 1], c='lightblue',
                          s=12, marker='.', zorder=2, alpha=0.7, label='Valid candidates')
        else:
            ax.scatter(candidates[:, 0], candidates[:, 1], c='lightblue',
                      s=12, marker='.', zorder=2, alpha=0.7, label='Candidates')

    trace_colors = plt.cm.tab20(np.linspace(0, 1, len(board.traces)))
    for i, trace in enumerate(board.traces):
        ax.plot(trace.start_x, trace.start_y, 's', color=trace_colors[i],
                markersize=5, zorder=5)

    if paths is not None:
        for i, path in enumerate(paths):
            if path is None:
                continue
            color = trace_colors[i % len(trace_colors)]
            xs = [p[0] for p in path]
            ys = [p[1] for p in path]
            ax.plot(xs, ys, '-', color=color, linewidth=1.0, alpha=0.8, zorder=4)

    if test_points is not None:
        for i, (tx, ty) in enumerate(test_points):
            color = trace_colors[i % len(trace_colors)]
            ax.plot(tx, ty, 'o', color=color, markersize=8,
                    markeredgecolor='black', markeredgewidth=0.5, zorder=6)
            excl = patches.Circle(
                (tx, ty), TP_TO_TP_MIN / 2,
                linewidth=0.5, edgecolor=color, facecolor='none',
                linestyle=':', alpha=0.3, zorder=2,
            )
            ax.add_patch(excl)

    ax.plot([], [], 's', color='gray', markersize=5, label='Starting points')
    if test_points:
        ax.plot([], [], 'o', color='gray', markersize=8,
                markeredgecolor='black', label='Test points')

    ax.set_xlim(board.x_min - 2, board.x_max + 2)
    ax.set_ylim(board.y_min - 2, board.y_max + 2)
    ax.set_aspect('equal')
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_title(title)
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Saved: {filename}")
    plt.close()
    return fig