"""Unit tests for the PCB environment, board, and A* router."""
import numpy as np

from envs.board import (
    load_te_example, generate_candidate_grid, check_tp_spacing,
    _is_valid_tp_position, MAX_CANDIDATES, TP_TO_TP_MIN, TP_TO_EDGE_MIN,
)
from envs.routing import route_all_traces, validate_routing_constraints, count_crossings
from envs.pcb_env import TPPlacementEnv


# ---------------------------------------------------------------- board / grid

def test_candidate_grid_fixed_size_and_real_count():
    board = load_te_example(num_traces=10, seed=0)
    cand, real = generate_candidate_grid(board, 6.5)
    assert cand.shape == (MAX_CANDIDATES, 2)
    assert 0 < real <= MAX_CANDIDATES


def test_all_real_candidates_are_valid_positions():
    board = load_te_example(num_traces=10, seed=0)
    cand, real = generate_candidate_grid(board, 6.5)
    assert all(_is_valid_tp_position(board, x, y) for x, y in cand[:real])


def test_check_tp_spacing():
    assert check_tp_spacing([(0.0, 0.0)], 0.0, TP_TO_TP_MIN + 1.0)      # far enough
    assert not check_tp_spacing([(0.0, 0.0)], 0.0, TP_TO_TP_MIN - 1.0)  # too close


def test_edge_clearance_enforced():
    board = load_te_example(num_traces=4, seed=0)
    # A point one mm inside the edge violates the TP-to-edge minimum.
    assert not _is_valid_tp_position(board, board.x_min + 1.0, board.y_min + 30.0)
    # A point well inside (away from obstacles) should be valid.
    assert _is_valid_tp_position(board, board.x_min + TP_TO_EDGE_MIN + 20.0,
                                 board.y_min + TP_TO_EDGE_MIN + 20.0)


# ---------------------------------------------------------------------- router

def test_astar_routes_a_greedy_placement():
    board = load_te_example(num_traces=8, seed=0)
    cand, real = generate_candidate_grid(board, 6.5)
    cand = cand[:real]
    placed = []
    for t in board.traces:
        d = np.hypot(cand[:, 0] - t.start_x, cand[:, 1] - t.start_y)
        for idx in np.argsort(d):
            if check_tp_spacing(placed, *cand[idx]):
                placed.append(tuple(cand[idx])); break
    paths, lengths, failures = route_all_traces(board, placed)
    assert len(paths) == len(board.traces)
    assert failures < len(board.traces)                 # not everything fails
    assert any(l < float('inf') for l in lengths)       # some real lengths


def test_validate_returns_expected_schema():
    board = load_te_example(num_traces=6, seed=1)
    cand, real = generate_candidate_grid(board, 6.5)
    cand = cand[:real]
    placed = [tuple(cand[i]) for i in range(0, 6)]
    paths, _, _ = route_all_traces(board, placed)
    v = validate_routing_constraints(board, paths)
    for key in ("violations", "trace_to_trace_min", "trace_to_edge_min",
                "trace_to_obstacle_min", "all_valid"):
        assert key in v
    assert isinstance(v["all_valid"], bool)


# ------------------------------------------------------------------------- env

def test_reset_regenerates_board_per_seed():
    """Different seeds -> different layouts; same seed -> reproducible."""
    env = TPPlacementEnv(num_traces=6, seed=1)
    env.reset(seed=1); a = env.board.rect_obstacles[0].cx
    env.reset(seed=2); b = env.board.rect_obstacles[0].cx
    env.reset(seed=1); a2 = env.board.rect_obstacles[0].cx
    assert a != b          # randomization actually takes effect
    assert a == a2         # deterministic for a given seed


def test_invalid_action_never_places_corner_tp():
    """An invalid (masked) action snaps to a valid candidate, never the junk corner."""
    env = TPPlacementEnv(num_traces=4, seed=3)
    env.reset(seed=3)
    env.step(0)                                   # place one TP
    masked = [i for i in range(env._real_count) if not env.candidate_mask[i]]
    assert masked, "expected some candidates masked out after a placement"
    env.step(masked[0])                           # pick an invalid candidate
    placed = env.placed_tps[-1]
    assert placed != (env.board.x_min, env.board.y_min)   # not the junk corner
    reals = {tuple(env.candidates[i]) for i in range(env._real_count)}
    assert tuple(placed) in reals                 # snapped to a real candidate


def test_env_uses_astar_by_default():
    """A* is the default router."""
    assert TPPlacementEnv(num_traces=4, seed=0).use_freerouting is False
    from envs.dreamer_wrapper import PCBDreamerEnv
    assert PCBDreamerEnv(num_traces=4, seed=0)._inner.use_freerouting is False


def test_episode_exposes_reward_components():
    """Per-component reward breakdown is logged in info."""
    env = TPPlacementEnv(num_traces=4, seed=5)
    env.reset(seed=5)
    info = {}
    for _ in range(env.num_traces):
        valid = np.where(env.candidate_mask[:env._real_count])[0]
        a = int(valid[0]) if len(valid) else 0
        _, _, term, _, info = env.step(a)
    assert term
    assert "reward_components" in info
    assert "routable" in info["reward_components"]


def test_training_env_wrapper_chain():
    """The dreamerv3 wrapper stack steps a full episode and returns reward components."""
    from envs.dreamer_wrapper import PCBDreamerEnv
    from envs import wrappers
    env = PCBDreamerEnv(num_traces=5, seed=7)
    env = wrappers.OneHotAction(env)
    env = wrappers.TimeLimit(env, 5)
    env = wrappers.SelectAction(env, key="action")
    env = wrappers.UUID(env)
    obs = env.reset()
    assert "image" in obs
    n = env.action_space.shape[0]      # OneHotAction Box: shape (num_candidates,)
    assert obs["mask"].shape == (n,) and obs["mask"].dtype == np.float32
    assert 0 < obs["mask"].sum() <= n
    done, info, steps = False, {}, 0
    while not done and steps < 10:
        onehot = np.zeros(n, dtype=np.float32)
        onehot[np.random.randint(n)] = 1.0
        obs, reward, done, info = env.step({"action": onehot})
        assert obs["mask"].shape == (n,)
        steps += 1
    assert done
    assert "reward_components" in info


def test_masked_actor_dist_suppresses_invalid_actions():
    """_apply_mask drives masked probabilities to ~zero while keeping log_probs finite."""
    torch = __import__("torch")
    import types
    from dreamerv3 import tools as dv3_tools
    from dreamerv3.models import ImagBehavior

    logits = torch.zeros(4, 8)                        # uniform actor
    dist = dv3_tools.OneHotDist(logits, unimix_ratio=0.01)
    mask = torch.ones(4, 8)
    mask[:, :6] = 0.0                                 # only actions 6,7 valid
    self = types.SimpleNamespace(_config=types.SimpleNamespace(
        actor={"dist": "onehot"}))
    masked = ImagBehavior._apply_mask(self, dist, mask)
    probs = masked.probs if hasattr(masked, "probs") else masked._dist.probs
    assert float(probs[:, :6].max()) < 1e-4           # invalid ~zero
    assert torch.allclose(probs.sum(-1), torch.ones(4), atol=1e-5)
    sample = masked.sample()
    assert float(sample[:, :6].sum()) == 0.0          # never samples invalid
    assert torch.isfinite(masked.log_prob(
        torch.nn.functional.one_hot(torch.tensor([0, 1, 6, 7]), 8).float())).all()


def test_pad_keepout_from_other_traces():
    """Each test pad's keep-out stays clear of every other trace's body."""
    from envs.routing import validate_routing_constraints, CELL_SIZE
    b = load_te_example(num_traces=16, seed=1)
    c, rc = generate_candidate_grid(b, 6.5); c = c[:rc]
    ccx = b.connector_x + b.connector_w / 2; ccy = b.connector_y + b.connector_h / 2
    chosen = []
    for idx in np.argsort(-np.hypot(c[:, 0] - ccx, c[:, 1] - ccy)):
        if len(chosen) >= 16: break
        if check_tp_spacing(chosen, *c[idx]): chosen.append(tuple(c[idx]))
    tps = sorted(chosen, key=lambda p: np.arctan2(p[1] - ccy, p[0] - ccx))
    pins = sorted(range(16), key=lambda i: np.arctan2(b.traces[i].start_y - ccy,
                                                      b.traces[i].start_x - ccx))
    placed = [None] * 16
    for k, i in enumerate(pins):
        placed[i] = tps[k]
    paths, _, _ = route_all_traces(b, placed)
    v = validate_routing_constraints(b, paths)
    assert v["tp_to_trace_min"] >= 1.5 * CELL_SIZE   # pad keep-out (> 1 cell)


def test_length_matching_is_crossing_safe_and_equalizes():
    """equalize_lengths meanders without introducing crossings or worsening spread."""
    from envs.routing import equalize_lengths
    b = load_te_example(num_traces=20, seed=1)
    c, rc = generate_candidate_grid(b, 6.5); c = c[:rc]
    ccx = b.connector_x + b.connector_w / 2; ccy = b.connector_y + b.connector_h / 2
    chosen = []
    for idx in np.argsort(-np.hypot(c[:, 0] - ccx, c[:, 1] - ccy)):
        if len(chosen) >= 20: break
        if check_tp_spacing(chosen, *c[idx]): chosen.append(tuple(c[idx]))
    tps = sorted(chosen, key=lambda p: np.arctan2(p[1] - ccy, p[0] - ccx))
    pins = sorted(range(20), key=lambda i: np.arctan2(b.traces[i].start_y - ccy,
                                                      b.traces[i].start_x - ccx))
    placed = [None] * 20
    for k, i in enumerate(pins):
        placed[i] = tps[k]
    paths, L, _ = route_all_traces(b, placed)
    fin0 = [x for x in L if x < float('inf')]
    s0 = (max(fin0) - min(fin0)) / np.mean(fin0)
    paths2, L2, _, _ = equalize_lengths(b, paths)
    fin1 = [x for x in L2 if x < float('inf')]
    s1 = (max(fin1) - min(fin1)) / np.mean(fin1)
    assert count_crossings(paths2) == 0          # meander never crosses
    assert s1 <= s0 + 0.05                        # equalizes (never worsens spread)


def test_edge_board_routes_to_top():
    """Edge board planar fan: every TP above every start, nearly all route, 0 crossings."""
    from envs.board import load_edge_board, fan_to_top_placement
    b = load_edge_board(num_traces=20, board_w=180.0, board_h=180.0)
    placed = fan_to_top_placement(b, 20)
    start_top = max(t.start_y for t in b.traces)
    assert all(p[1] > start_top for p in placed)      # all traces end above the start
    paths, _, f = route_all_traces(b, placed)
    assert count_crossings(paths) == 0
    assert (20 - f) >= 18                              # nearly all (measured 20/20)


def test_two_row_edge_wrap_and_equalize():
    """2-row edge wrap: routed traces are crossing-free and equalization reduces spread."""
    from envs.board import load_edge_board_2row, wrap_to_top_placement
    from envs.routing import equalize_lengths
    b = load_edge_board_2row(num_traces=12, board_w=220.0, board_h=230.0)
    placed = wrap_to_top_placement(b, 12)
    start_top = max(t.start_y for t in b.traces)
    assert all(p[1] > start_top for p in placed)      # all TPs above the connector
    paths, L, f = route_all_traces(b, placed)
    assert count_crossings(paths) == 0
    assert (12 - f) >= 7                                # most route (single-layer wrap is hard)
    fin0 = [x for x in L if x < float('inf')]
    s0 = (max(fin0) - min(fin0)) / np.mean(fin0)
    eq_paths, eqL, _, _ = equalize_lengths(b, paths)
    fin1 = [x for x in eqL if x < float('inf')]
    s1 = (max(fin1) - min(fin1)) / np.mean(fin1)
    assert count_crossings(eq_paths) == 0              # equalization stays planar
    assert s1 < s0                                      # length-matching reduces spread


def test_auto_layers_planar_per_layer_and_respects_nrz():
    """route_auto_layers: every layer planar, nothing through the NRZ, nearly all route."""
    from envs.board import load_te_example, spread_placement
    from envs.routing import route_auto_layers
    b = load_te_example(num_traces=12, seed=6)
    placed = spread_placement(b, 12)
    paths, L, layer_of, f, lx = route_auto_layers(b, placed, max_layers=4)
    assert all(x == 0 for x in lx.values())              # every layer planar (key invariant)
    assert (12 - f) >= 10                                 # spread placement routes (nearly) all

    nrz = [o for o in b.rect_obstacles if o.name == "non_routing_zone"]
    def deep_in_nrz(x, y):                                # inside the body (not boundary rounding)
        return any(abs(x - o.cx) <= o.width / 2 - 1.5 and abs(y - o.cy) <= o.height / 2 - 1.5
                   for o in nrz)
    for p in paths:
        if p:
            assert not any(deep_in_nrz(x, y) for (x, y) in p)   # never through the non-routing zone


def test_equal_length_placement_more_uniform_radius():
    """equal_length_placement yields a more uniform TP radius than spread_placement."""
    from envs.board import load_te_example, spread_placement, equal_length_placement
    b = load_te_example(num_traces=16, seed=6)
    ccx = b.connector_x + b.connector_w / 2
    ccy = b.connector_y + b.connector_h / 2
    def radius_cv(pl):
        r = [np.hypot(x - ccx, y - ccy) for (x, y) in pl]
        return float(np.std(r) / np.mean(r))
    assert radius_cv(equal_length_placement(b, 16)) < radius_cv(spread_placement(b, 16))


def test_any_angle_shortens_without_reducing_clearance():
    """any_angle_shortcut never lengthens, never reduces clearance, stays planar."""
    from envs.board import load_te_example, equal_length_placement
    from envs.routing import any_angle_shortcut, min_trace_separation
    b = load_te_example(num_traces=16, seed=6)
    placed = equal_length_placement(b, 16)
    p, L, f = route_all_traces(b, placed)
    plen = lambda P: sum(np.hypot(P[k+1][0]-P[k][0], P[k+1][1]-P[k][1]) for k in range(len(P)-1))
    base_len = sum(plen(x) for x in p if x)
    base_sep = min_trace_separation(p)
    aa = any_angle_shortcut(p, b)
    assert sum(plen(x) for x in aa if x) <= base_len + 1e-6     # never longer
    assert min_trace_separation(aa) >= base_sep - 1e-6          # clearance not reduced
    assert count_crossings(aa) == 0                             # stays planar


def test_strict_clearance_guarantees_full_pitch_at_45deg():
    """strict_clearance keeps every net pair >= full pitch, surviving post-processing."""
    from envs.board import equal_length_placement
    from envs.routing import (equalize_lengths, any_angle_shortcut,
                              min_trace_separation, CELL_SIZE)
    b = load_te_example(num_traces=20, seed=0)
    placed = equal_length_placement(b, 20)

    paths, _, fails = route_all_traces(b, placed)          # strict by default
    assert fails == 0                                       # no capacity loss
    assert min_trace_separation(paths) >= CELL_SIZE - 0.05
    v = validate_routing_constraints(b, paths)
    assert v["clearance_ok"]
    assert not [x for x in v["violations"] if x[0] == "trace_to_trace"]
    eq, _, _, _ = equalize_lengths(b, paths)                # meander keeps pitch
    assert min_trace_separation(eq) >= CELL_SIZE - 0.05
    assert count_crossings(eq) == 0
    aa = any_angle_shortcut(paths, b)                       # any-angle keeps pitch
    assert min_trace_separation(aa) >= CELL_SIZE - 0.05

    legacy, _, lf = route_all_traces(b, placed, strict_clearance=False)
    assert lf == 0
    if min_trace_separation(legacy) < CELL_SIZE - 0.05:     # sub-pitch present
        lv = validate_routing_constraints(b, legacy)
        assert not lv["clearance_ok"]
        assert [x for x in lv["violations"] if x[0] == "trace_to_trace"]


def test_rectilinear_base_guarantees_pitch_clearance():
    """diagonal=False keeps every trace pair >= pitch; any-angle preserves it."""
    from envs.board import load_te_example, equal_length_placement
    from envs.routing import any_angle_shortcut, min_trace_separation, CELL_SIZE
    b = load_te_example(num_traces=14, seed=6, board_size=120.0)
    placed = equal_length_placement(b, 14)
    p, L, f = route_all_traces(b, placed, diagonal=False)
    assert (14 - f) >= 12                                       # rectilinear still routes
    assert min_trace_separation(p) >= CELL_SIZE - 0.05         # >= pitch (clearance ensured)
    aa = any_angle_shortcut(p, b)
    assert min_trace_separation(aa) >= CELL_SIZE - 0.05        # any-angle preserves it
    assert count_crossings(aa) == 0


def test_parametric_challenge_and_formulation_reward():
    """make_challenge places all pads; the formulation reward scores a layer assignment."""
    from envs.board import make_challenge, ChallengeSpec
    from envs.formulations import ELECTED, FORMULATIONS, evaluate_ordered_layer
    for placement in ("ring", "gap_aligned"):
        b, pl = make_challenge(ChallengeSpec(num_traces=12, n_gaps=3, board_size=120.0,
                                             placement=placement))
        assert sum(1 for p in pl if p) == 12                  # parametric board fully placed
    assert ELECTED in FORMULATIONS                            # elected formulation is registered
    reward, comp = evaluate_ordered_layer(b, pl, [0] * 12)    # all on one layer
    assert comp["routed"] >= 8 and comp["same_layer_crossings"] == 0
    assert isinstance(reward, float)


def test_env_layer_aware_reward_logs_layers_and_vias():
    """reward_mode='layer_aware' logs layer/via/routing reward components."""
    from envs.pcb_env import TPPlacementEnv
    env = TPPlacementEnv(num_traces=6, seed=0, reward_mode="layer_aware")
    env.reset(seed=0)
    done = False
    info = {}
    while not done:
        valid = np.where(env.candidate_mask[:env._real_count])[0]
        _, r, done, _, info = env.step(int(valid[env.current_trace % len(valid)]))
    comp = info.get("reward_components", {})
    assert "layers" in comp and "vias" in comp and "routing" in comp


def test_router_output_never_crosses():
    """route_all_traces output is always planar, regardless of placement."""
    b = load_te_example(num_traces=16, seed=2)
    c, rc = generate_candidate_grid(b, 6.5); c = c[:rc]
    placed = []                                   # greedy (crossing-prone) placement
    for t in b.traces:
        d = np.hypot(c[:, 0] - t.start_x, c[:, 1] - t.start_y)
        for i in np.argsort(d):
            if check_tp_spacing(placed, *c[i]):
                placed.append(tuple(c[i])); break
    paths, _, _ = route_all_traces(b, placed)
    assert count_crossings(paths) == 0


def test_vectorized_geometry_matches_scalar_reference():
    """Vectorized crossing test and min-separation agree with scalar references."""
    from envs.routing import (_crossing_pairs, _seg_seg_dist, min_trace_separation,
                              _ccw, _ccw_sign)
    rng = np.random.RandomState(7)
    paths = [[tuple(q) for q in rng.uniform(0, 50, size=(rng.randint(2, 8), 2))]
             for _ in range(6)]

    # scalar crossing reference (pair loop, same deadband orientation signs)
    segs = [(i, p[k], p[k + 1]) for i, p in enumerate(paths)
            for k in range(len(p) - 1)]
    ref_pairs = set()
    for x in range(len(segs)):
        ia, A, B = segs[x]
        for y in range(x + 1, len(segs)):
            ib, C, D = segs[y]
            if ia == ib:
                continue
            s1 = _ccw_sign(_ccw(C[0], C[1], D[0], D[1], A[0], A[1]))
            s2 = _ccw_sign(_ccw(C[0], C[1], D[0], D[1], B[0], B[1]))
            s3 = _ccw_sign(_ccw(A[0], A[1], B[0], B[1], C[0], C[1]))
            s4 = _ccw_sign(_ccw(A[0], A[1], B[0], B[1], D[0], D[1]))
            if s1 * s2 < 0 and s3 * s4 < 0:
                ref_pairs.add((min(ia, ib), max(ia, ib)))
    assert _crossing_pairs(paths) == ref_pairs
    assert ref_pairs                                   # random tangles do cross

    # collinear-but-disjoint 45 deg segments are NOT crossings
    colin = [[(0.0, 0.0), (1.3286, 1.3286)],
             [(50.1234, 50.1234), (51.452, 51.452)]]
    assert _crossing_pairs(colin) == set()

    # scalar min-separation reference
    ref = min(_seg_seg_dist(A, B, C, D)
              for x, (ia, A, B) in enumerate(segs)
              for ib, C, D in (segs[y] for y in range(x + 1, len(segs)))
              if ia != ib)
    assert abs(min_trace_separation(paths) - ref) < 1e-9


def test_rotate_board_geometry_and_placement_follow():
    """rotate_board stays well-formed and rotate_points preserves pin->TP distances."""
    from envs.board import rotate_board, rotate_points, equal_length_placement
    b = load_te_example(num_traces=12, seed=3)
    placed = equal_length_placement(b, 12)
    d0 = [np.hypot(px - t.start_x, py - t.start_y)
          for t, (px, py) in zip(b.traces, placed)]
    for k in range(4):
        for mirror in (False, True):
            rb = rotate_board(b, k, mirror)
            rp = rotate_points(placed, b, k, mirror)
            if k % 2:
                assert (rb.width, rb.height) == (b.height, b.width)
            else:
                assert (rb.width, rb.height) == (b.width, b.height)
            assert len(rb.traces) == 12 and len(rb.rect_obstacles) == len(b.rect_obstacles)
            for t in rb.traces:
                assert rb.x_min < t.start_x < rb.x_max
                assert rb.y_min < t.start_y < rb.y_max
            assert rb.connector_x > rb.x_min and rb.connector_y > rb.y_min
            d1 = [np.hypot(px - t.start_x, py - t.start_y)
                  for t, (px, py) in zip(rb.traces, rp)]
            assert np.allclose(d0, d1)                 # rigid transform


def test_router_orientation_invariant():
    """Rotated/mirrored boards route 20/20 with zero crossings at full pitch."""
    from envs.board import rotate_board, rotate_points, equal_length_placement
    from envs.routing import min_trace_separation
    for seed in (0, 1):
        b = load_te_example(num_traces=20, seed=seed)
        placed = equal_length_placement(b, 20)
        for k, mirror in ((0, False), (1, False), (2, False), (3, False), (1, True)):
            rb = rotate_board(b, k, mirror)
            rp = rotate_points(placed, b, k, mirror)
            paths, _, fails = route_all_traces(rb, rp)
            assert fails == 0, \
                f"seed {seed} rot {k*90}deg mirror={mirror}: {fails}/20 failed"
            assert count_crossings(paths) == 0
            from envs.routing import CELL_SIZE
            assert min_trace_separation(paths) >= CELL_SIZE - 0.05  # full pitch


def test_env_accepts_rotated_board_factory():
    """TPPlacementEnv works with a rotating board factory across resets."""
    from envs.board import load_te_example, rotate_board
    factory = lambda s: rotate_board(
        load_te_example(num_traces=5, seed=s), s % 4, bool(s % 2))
    env = TPPlacementEnv(num_traces=5, seed=9, board_factory=factory)
    env.reset(seed=9)          # 90 deg: connector taller than wide
    assert env.board.connector_h > env.board.connector_w
    env.reset(seed=10)         # 180 deg: horizontal pin rows again
    assert env.board.connector_w > env.board.connector_h
    done, info = False, {}
    while not done:
        valid = np.where(env.candidate_mask[:env._real_count])[0]
        _, _, done, _, info = env.step(int(valid[0]))
    assert "reward_components" in info


def test_smart_placement_no_training_baseline():
    """smart_placement is spacing-valid and routes fully at full pitch."""
    from envs.board import load_te_example, smart_placement
    from envs.routing import min_trace_separation, CELL_SIZE
    b = load_te_example(num_traces=12, seed=0)
    placed = smart_placement(b, 12)
    assert len(placed) == 12 and all(p is not None for p in placed)
    for i in range(12):
        assert check_tp_spacing(placed[:i], *placed[i])
    paths, _, fails = route_all_traces(b, placed)
    assert fails == 0
    assert count_crossings(paths) == 0
    assert min_trace_separation(paths) >= CELL_SIZE - 0.05


def test_canonical_excel_board_routes_all_on_single_layer():
    """Canonical TE board: smart placement routes all 20 nets on one layer at full pitch."""
    from envs.board import load_te_excel, smart_placement, TRACE_MIN_CENTER_TO_CENTER
    from envs.routing import (min_trace_separation, octile_lower_bounds,
                              validate_routing_constraints, CELL_SIZE)
    b = load_te_excel()
    assert (b.width, b.height) == (135.0, 90.0)
    assert len(b.traces) == 20
    nrz = next(o for o in b.rect_obstacles if o.name == "non_routing_zone")
    for row_y in (113.9436, 120.7446):             # original ys + 6mm shift
        xs = sorted(t.start_x for t in b.traces if abs(t.start_y - row_y) < 0.01)
        assert len(xs) == 10
        gaps = np.diff(xs)
        assert gaps.min() >= TRACE_MIN_CENTER_TO_CENTER - 1e-6   # legal pitch
        assert gaps.max() > 3.5                    # designed group break kept
        assert abs((xs[0] + xs[-1]) / 2 - nrz.cx) < 1e-6  # row centered on NRZ
    placed = smart_placement(b, 20)
    paths, L, fails = route_all_traces(b, placed)
    assert fails == 0                              # ALL nets, ONE copper layer
    assert count_crossings(paths) == 0
    assert min_trace_separation(paths) >= CELL_SIZE - 0.05
    fin = [x for x in L if x < 1e9]
    assert max(fin) <= 1.6 * max(octile_lower_bounds(b, placed))  # no wraps
    v = validate_routing_constraints(b, paths)
    assert v["all_valid"] and v["tp_clearance_ok"]
    # Full length-equalization: every trace within one cell of the common target.
    from envs.routing import route_to_length, equalize_lengths
    p2, L2, _nd = route_to_length(b, placed, paths, L)
    eq, eqL, _tgt, matched = equalize_lengths(b, p2, test_points=placed)
    assert matched == 20                           # all at the common target
    fin2 = [x for x in eqL if x < 1e9]
    assert (max(fin2) - min(fin2)) / np.mean(fin2) < 0.05
    v2 = validate_routing_constraints(b, eq)
    assert v2["tp_clearance_ok"] and count_crossings(eq) == 0
    assert min_trace_separation(eq) >= CELL_SIZE - 0.05
    for p in paths:                                # never over a drill body
        if p:
            for (x, y) in p:
                for o in b.circ_obstacles:
                    assert np.hypot(x - o.cx, y - o.cy) > o.radius


def test_equalization_respects_pad_keepouts_and_min_max_layers():
    """Meander never enters foreign pad keep-outs; layer moves collapse the length target."""
    from envs.board import load_te_excel, equal_length_placement
    from envs.routing import (route_auto_layers, optimize_layers_for_length,
                              octile_lower_bounds, equalize_lengths,
                              validate_routing_constraints, min_trace_separation,
                              CELL_SIZE)
    # Shift the connector cluster back down 6mm to recreate the capacity-bound board.
    b = load_te_excel()
    for o in b.rect_obstacles:
        o.cy -= 6.0
    for o in b.circ_obstacles:
        o.cy -= 6.0
    for t in b.traces:
        t.start_y -= 6.0
    b.connector_y -= 6.0
    placed = equal_length_placement(b, 20)
    fast = dict(n_starts=1, max_iters=12, repair_passes=1)
    mp, mL, lo, mf, _ = route_auto_layers(b, placed, max_layers=6, **fast)
    assert mf == 0
    raw_max = max(x for x in mL if x < 1e9)

    op, oL, olo, moves = optimize_layers_for_length(
        b, placed, mp, mL, lo, max_layers=6, **fast)
    opt_max = max(x for x in oL if x < 1e9)
    lbs = octile_lower_bounds(b, placed)
    assert moves >= 1
    assert opt_max < 0.6 * raw_max               # target collapses (280->~90)
    assert opt_max <= 2.0 * max(lbs)             # near the geometric bound

    target_mm = opt_max - b.traces[0].breakout_length
    used = sorted(set(l for l in olo if l >= 0))
    for L in used:                               # equalize + validate per layer
        idxs = [i for i in range(20) if olo[i] == L]
        ep, el, _t, _m = equalize_lengths(
            b, [op[i] for i in idxs], target_mm=target_mm,
            test_points=[placed[i] for i in idxs])
        v = validate_routing_constraints(b, ep)
        assert v["crossings"] == 0
        assert v["tp_clearance_ok"], v["violations"]   # keep-outs survive
        assert not [x for x in v["violations"] if x[0] == "tp_keepout"]
        if sum(1 for p in ep if p) > 1:
            assert min_trace_separation(ep) >= CELL_SIZE - 0.05


def test_wire_estimate_is_wrap_aware():
    """wire_estimate: a pad across the NRZ costs > air distance; same-side costs ~air."""
    from envs.board import load_te_excel, wire_estimate
    b = load_te_excel()
    t_low = next(t for t in b.traces if t.start_y < 117)   # lower escape row
    across = wire_estimate(b, t_low, (t_low.start_x, t_low.start_y + 40.0))
    same = wire_estimate(b, t_low, (t_low.start_x - 30.0, t_low.start_y - 5.0))
    assert across > 44.0                    # air 40 + wrap surcharge
    assert abs(same - 32.07) < 1.0          # ~octile air distance


def test_detour_extension_and_order_hint():
    """route_to_length gets more nets to the target; order_hint never hurts routing."""
    from envs.board import load_te_excel, smart_placement
    from envs.routing import (route_to_length, min_trace_separation,
                              validate_routing_constraints, CELL_SIZE)
    from envs.ordering import load_model, predict_order
    b = load_te_excel()
    placed = smart_placement(b, 20)
    fast = dict(n_starts=1, max_iters=12, repair_passes=1)
    paths, L, f = route_all_traces(b, placed, **fast)
    _eq0, eqL0, _t0, nm0 = equalize_from(b, paths, placed)
    p2, L2, next_ = route_to_length(b, placed, paths, L)
    eq, eqL, _t, nm = equalize_from(b, p2, placed)
    assert nm >= nm0                        # never fewer traces at target
    v = validate_routing_constraints(b, eq)
    assert v["crossings"] == 0 and v["tp_clearance_ok"]
    assert min_trace_separation(eq) >= CELL_SIZE - 0.05

    hint = list(reversed(range(20)))
    _p, _L, fh = route_all_traces(b, placed, order_hint=hint, **fast)
    assert fh <= f + 1                      # hint is additive, never harmful
    model = load_model()
    if model is not None:
        od = predict_order(b, placed, model=model)
        assert sorted(od) == list(range(20))
        _p, _L, fo = route_all_traces(b, placed, order_hint=od, n_starts=3)
        assert fo == 0


def equalize_from(b, paths, placed):
    from envs.routing import equalize_lengths
    return equalize_lengths(b, paths, test_points=placed)


def test_fuzz_invariants_random_boards_random_placements():
    """Constructive guarantees hold on random boards x random legal placements."""
    from envs.board import (load_te_example, make_challenge, ChallengeSpec,
                            load_te_excel, rotate_board, generate_candidate_grid)
    from envs.routing import (equalize_lengths, min_trace_separation,
                              validate_routing_constraints, CELL_SIZE)
    rng = np.random.RandomState(42)
    fast = dict(n_starts=1, max_iters=12, repair_passes=1)

    def rand_board(case):
        if case % 4 == 3:
            b, _ = make_challenge(ChallengeSpec(
                num_traces=int(rng.choice([12, 16])),
                n_gaps=int(rng.choice([2, 3])), board_size=120.0,
                seed=int(rng.randint(50))))
        elif case % 4 == 2:
            b = load_te_excel()
        else:
            b = load_te_example(int(rng.choice([10, 14, 20])),
                                seed=int(rng.randint(50)),
                                board_size=float(rng.choice([120.0, 160.0])))
        return rotate_board(b, int(rng.randint(4)), bool(rng.randint(2)))

    for case in range(8):
        b = rand_board(case)
        n = len(b.traces)
        c, rc = generate_candidate_grid(b, 6.5)
        c = c[:rc]
        placed = []
        for _ in range(n):                       # random legal placement
            for idx in rng.permutation(rc):
                if check_tp_spacing(placed, *c[idx]):
                    placed.append(tuple(c[idx]))
                    break
        paths, L, fails = route_all_traces(b, placed, **fast)
        eq, _eqL, _t, _m = equalize_lengths(b, paths, test_points=placed)
        for tag, ps in (("routed", paths), ("equalized", eq)):
            routed = [p for p in ps if p]
            assert count_crossings(ps) == 0, f"case {case} {tag}: crossing"
            if len(routed) > 1:
                assert min_trace_separation(ps) >= CELL_SIZE - 0.05, \
                    f"case {case} {tag}: sub-pitch"
            v = validate_routing_constraints(b, ps)
            assert v["tp_clearance_ok"], f"case {case} {tag}: pad keep-out"
            assert not [x for x in v["violations"]
                        if x[0] in ("trace_to_trace", "tp_keepout")], \
                f"case {case} {tag}: {v['violations'][:3]}"
            for p in routed:                     # never over a drill body
                for (x, y) in p:
                    for o in b.circ_obstacles:
                        assert np.hypot(x - o.cx, y - o.cy) > o.radius, \
                            f"case {case} {tag}: over drill {o.name}"


def test_no_route_through_foreign_pin_or_pad():
    """No routed trace passes through another net's pin cell, stub, or pad cell."""
    from envs.routing import _build_context, min_trace_separation
    for seed in (0, 1):
        b = load_te_example(num_traces=20, seed=seed)
        c, rc = generate_candidate_grid(b, 6.5); c = c[:rc]
        placed = []                               # clustered, worst-case placement
        for t in b.traces:
            d = np.hypot(c[:, 0] - t.start_x, c[:, 1] - t.start_y)
            for i in np.argsort(d):
                if check_tp_spacing(placed, *c[i]):
                    placed.append(tuple(c[i])); break
        paths, _, fails = route_all_traces(b, placed)
        grid, rows, cols, _bl, cells, _ep, _ow = _build_context(b, placed)
        owned = {}                                # cell -> owner net (starts + TPs)
        for i, (s, e) in enumerate(cells):
            owned[s] = i
            owned[e] = i
        for i, p in enumerate(paths):
            if not p:
                continue
            for (x, y) in p:
                cc, rr = grid._world_to_grid(x, y)
                assert owned.get((rr, cc), i) == i, \
                    f"seed {seed}: net {i} passes through net {owned[(rr, cc)]}'s pad/pin"
        # No two routed traces may touch (share a geometric point).
        routed = [p for p in paths if p]
        if len(routed) > 1:
            assert min_trace_separation(paths) > 0.5   # mm; touching would be ~0


if __name__ == "__main__":
    import sys
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn(); print(f"PASS {fn.__name__}")
        except Exception as e:
            failed += 1; print(f"FAIL {fn.__name__}: {e!r}")
    print(f"\n{len(fns)-failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
