"""Training formulations for the PCB placement+routing problem, plus the
shared terminal reward and a CPU evaluator for the elected formulation."""
from dataclasses import dataclass
import copy
import numpy as np

from envs.routing import route_all_traces, count_crossings, min_trace_separation, CELL_SIZE


@dataclass
class Formulation:
    key: str
    controls: str          # what the policy outputs
    state: str
    action: str
    reward: str
    targets_bottleneck: str
    cost: str


# The forced-crossing bottleneck is decided by routing order + layer assignment.
FORMULATIONS = {
    "placement": Formulation(
        "placement", "where each test point goes (candidate cell)",
        "64x64x3 board image + index of net to place", "candidate grid cell",
        "terminal: routable + length + spread + compactness",
        "YES on boards with room+gaps (measured: gap-aligned placement -> single-layer "
        "at 16/18 traces, 19/20 at 20); only topology-bound on tight/few-gap boards",
        "ALREADY BUILT (envs/pcb_env.py:TPPlacementEnv); needs GPU to train"),
    "net_ordering": Formulation(
        "net_ordering", "the order nets are routed",
        "board + congestion (present/history) + routed-so-far mask", "pick next net",
        "-failures - w*crossings - w*length (or +1 per conflict-free net)",
        "directly (order decides which nets fit before congestion locks them out)",
        "ordering trainer exists (scripts/train_ordering.py, CPU minutes)"),
    "layer_assignment": Formulation(
        "layer_assignment", "which copper layer each net routes on",
        "board + crossing graph", "layer per net (sequential)",
        "-layers - vias - failures",
        "directly (sets how many nets spill to inner layers = the via count)",
        "moderate build; GPU to train"),
    "ordered_layer": Formulation(
        "ordered_layer", "jointly: the next net AND its layer",
        "board + per-layer congestion + remaining-nets mask", "(net, layer)",
        "-layers - vias - failures - w*length (see routing_reward)",
        "HEAD-ON: order+layer together determine the forced crossings -> layer count",
        "moderate build; subsumes net_ordering+layer_assignment; GPU to train"),
    "waypoint": Formulation(
        "waypoint", "intermediate points that steer each net's A* (your original idea)",
        "board + current partial route", "next waypoint",
        "routed + length",
        "per-net (escapes A* traps); weak on the global ordering bottleneck",
        "moderate build; GPU to train"),
    "joint_worldmodel": Formulation(
        "joint_worldmodel", "placement + order + layer together, in a world model",
        "full board", "combined", "global quality (routing_reward)",
        "everything (ultimate); ties to the DreamerV3 thesis",
        "largest build; GPU to train"),
}

# Elected primary lever: placement; deeper lever for hard boards: ordered_layer.
ELECTED = "placement"
DEEPER = "ordered_layer"


def routing_reward(board, paths, lengths, layer_of, *,
                   w_route=10.0, w_layer=3.0, w_via=0.5, w_len=2.0,
                   w_maxlen=6.0, w_cross=100.0):
    """Shared terminal reward for any formulation (higher is better)."""
    n = len(paths)
    routed = sum(1 for p in paths if p)
    used = sorted(set(l for l in layer_of if l >= 0))
    layers = len(used) if used else 1
    vias = sum(1 for l in layer_of if l >= 1)
    fin = [x for x in lengths if x < 1e9]
    total_len = float(sum(fin))
    max_len = float(max(fin)) if fin else 0.0
    diag = float(np.hypot(board.width, board.height))
    same_layer_x = max((count_crossings([paths[i] for i in range(n) if layer_of[i] == L])
                        for L in used), default=0)
    # Equalization pads every trace to the max, so final board length = n * max.
    reward = (w_route * routed / max(n, 1)
              - w_layer * (layers - 1)
              - w_via * vias
              - w_len * total_len / max(n * diag, 1e-6)
              - w_maxlen * max_len / max(diag, 1e-6)
              - w_cross * same_layer_x)
    return reward, dict(routed=routed, layers=layers, vias=vias,
                        same_layer_crossings=same_layer_x,
                        total_len=total_len, max_len=max_len)


def route_fixed_layers(board, placed, layer_of, **kw):
    """Route each layer planar and independently given a fixed per-net layer assignment."""
    n = min(len(board.traces), len(placed))
    paths = [None] * n
    lengths = [float('inf')] * n
    failures = 0
    for L in sorted(set(layer_of[:n])):
        idx = [i for i in range(n) if layer_of[i] == L]
        sub = copy.copy(board)
        sub.traces = [board.traces[i] for i in idx]
        sp, sl, sf = route_all_traces(sub, [placed[i] for i in idx], **kw)
        failures += sf
        for k, i in enumerate(idx):
            paths[i] = sp[k]
            lengths[i] = sl[k]
    return paths, lengths, failures


def evaluate_ordered_layer(board, placed, layer_of, **kw):
    """Score a layer assignment by routing it and applying the shared reward."""
    paths, lengths, _ = route_fixed_layers(board, placed, layer_of, **kw)
    return routing_reward(board, paths, lengths, layer_of)
