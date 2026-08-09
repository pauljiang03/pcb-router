"""Wrapper for DreamerV3 compatibility (old-style 4-return step + dict obs)."""

import gymnasium.spaces as spaces
import numpy as np
from envs.pcb_env import TPPlacementEnv


class PCBDreamerEnv:
    metadata = {}

    # Reward components exported as log_* keys; tools.simulate sums each over
    # the episode and logs it to tensorboard. Fixed set keeps cache arrays aligned.
    _LOG_KEYS = ("routing", "routable", "routed", "layers", "vias",
                 "same_layer_crossings", "length", "length_max", "max_len",
                 "spread", "compactness", "phi")

    def __init__(self, num_traces=8, seed=0, reward_mode="layer_aware",
                 route_n_starts=1, route_max_iters=12, route_repair_passes=1,
                 board_factory=None, shaping="none"):
        # Fast routing preset for training; failures are placement- not
        # search-determined, so quality settings are left to eval.
        self._inner = TPPlacementEnv(
            num_traces=num_traces, seed=seed, use_freerouting=False,
            reward_mode=reward_mode, route_n_starts=route_n_starts,
            route_max_iters=route_max_iters,
            route_repair_passes=route_repair_passes, board_factory=board_factory,
            shaping=shaping)
        self._seed = seed
        self.reward_range = [-np.inf, np.inf]

    @property
    def observation_space(self):
        return spaces.Dict({
            "image": spaces.Box(0, 255, (64, 64, 3), dtype=np.uint8),
            # Ground-truth valid-action mask (1 = candidate still legal).
            "mask": spaces.Box(0, 1, (self._inner.num_candidates,),
                               dtype=np.float32),
            # Exact geometry for the encoder/decoder MLP branch (see _vector).
            "vector": spaces.Box(-1, 1, (self._vector_len(),),
                                 dtype=np.float32),
            "is_first": spaces.Box(0, 1, (), dtype=np.uint8),
            "is_last": spaces.Box(0, 1, (), dtype=np.uint8),
            "is_terminal": spaces.Box(0, 1, (), dtype=np.uint8),
        })

    @property
    def action_space(self):
        space = spaces.Box(low=0, high=1,
                           shape=(self._inner.num_candidates,),
                           dtype=np.float32)
        space.discrete = True
        space.n = self._inner.num_candidates
        return space

    def _mask(self):
        return self._inner.candidate_mask.astype(np.float32)

    def _vector_len(self):
        # progress + current-pin xy + placed-TP xys + candidate xys; sized by
        # the REQUESTED trace count so the shape is board-independent.
        return 3 + 2 * self._inner._num_traces_requested \
            + 2 * self._inner.num_candidates

    def _vector(self):
        """Exact geometric state for the encoder's MLP branch -- coordinates
        the 64x64 render only carries at ~2mm/pixel: [episode progress,
        current pin xy, placed TP xys, candidate xys], all normalized to the
        board bounds, -1 = empty slot. The candidate block grounds each
        action index to its board position, which the image alone leaves to
        single dim pixels."""
        inner = self._inner
        b = inner.board
        w, h = max(b.width, 1e-6), max(b.height, 1e-6)
        nx = lambda x: (x - b.x_min) / w
        ny = lambda y: (y - b.y_min) / h
        n_req = inner._num_traces_requested
        n = inner.num_traces
        vec = np.full(self._vector_len(), -1.0, dtype=np.float32)
        vec[0] = inner.current_trace / max(n, 1)
        if inner.current_trace < n:
            t = b.traces[inner.current_trace]
            vec[1], vec[2] = nx(t.start_x), ny(t.start_y)
        for i, (px, py) in enumerate(inner.placed_tps[:n_req]):
            vec[3 + 2 * i], vec[4 + 2 * i] = nx(px), ny(py)
        base = 3 + 2 * n_req
        for i in range(inner._real_count):
            cx, cy = inner.candidates[i]
            vec[base + 2 * i], vec[base + 2 * i + 1] = nx(cx), ny(cy)
        return vec

    def reset(self):
        obs, _ = self._inner.reset(seed=self._seed)
        self._seed += 1
        return {"image": obs, "mask": self._mask(), "vector": self._vector(),
                "is_first": True, "is_last": False, "is_terminal": False}

    def step(self, action):
        obs, reward, terminated, truncated, info = self._inner.step(int(action))
        done = terminated or truncated
        out = {"image": obs, "mask": self._mask(), "vector": self._vector(),
               "is_first": False, "is_last": done, "is_terminal": terminated}
        comp = info.get("reward_components", {})
        # Zero mid-episode, so each episode sum equals the terminal value.
        for k in self._LOG_KEYS:
            out[f"log_{k}"] = np.float32(comp.get(k, 0.0))
        return out, np.float32(reward), done, info

    def render(self):
        return self._inner.render()

    def close(self):
        pass