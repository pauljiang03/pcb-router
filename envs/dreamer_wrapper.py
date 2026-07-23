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
                 "spread", "compactness")

    def __init__(self, num_traces=8, seed=0, reward_mode="layer_aware",
                 route_n_starts=1, route_max_iters=12, route_repair_passes=1,
                 board_factory=None):
        # Fast routing preset for training; failures are placement- not
        # search-determined, so quality settings are left to eval.
        self._inner = TPPlacementEnv(
            num_traces=num_traces, seed=seed, use_freerouting=False,
            reward_mode=reward_mode, route_n_starts=route_n_starts,
            route_max_iters=route_max_iters,
            route_repair_passes=route_repair_passes, board_factory=board_factory)
        self._seed = seed
        self.reward_range = [-np.inf, np.inf]

    @property
    def observation_space(self):
        return spaces.Dict({
            "image": spaces.Box(0, 255, (64, 64, 3), dtype=np.uint8),
            # Ground-truth valid-action mask (1 = candidate still legal).
            "mask": spaces.Box(0, 1, (self._inner.num_candidates,),
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

    def reset(self):
        obs, _ = self._inner.reset(seed=self._seed)
        self._seed += 1
        return {"image": obs, "mask": self._mask(),
                "is_first": True, "is_last": False, "is_terminal": False}

    def step(self, action):
        obs, reward, terminated, truncated, info = self._inner.step(int(action))
        done = terminated or truncated
        out = {"image": obs, "mask": self._mask(),
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