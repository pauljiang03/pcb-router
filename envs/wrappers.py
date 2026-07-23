"""Lightweight env wrappers for DreamerV3 (no gym dependency;
uses gymnasium.spaces only)."""

import datetime
import uuid
import numpy as np
import gymnasium.spaces as spaces


class BaseWrapper:
    """Minimal env wrapper (replaces gym.Wrapper without the dependency)."""

    def __init__(self, env):
        self.env = env

    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def action_space(self):
        return self.env.action_space

    @property
    def reward_range(self):
        return getattr(self.env, "reward_range", (-np.inf, np.inf))

    @property
    def metadata(self):
        return getattr(self.env, "metadata", {})

    def step(self, action):
        return self.env.step(action)

    def reset(self):
        return self.env.reset()

    def render(self):
        return self.env.render()

    def close(self):
        if hasattr(self.env, "close"):
            return self.env.close()

    def __getattr__(self, name):
        # Pickle-safe: never recurse when `env` is absent (mid-unpickle) and
        # never delegate dunder lookups, or __getattr__('env') recurses forever.
        if name.startswith("__"):
            raise AttributeError(name)
        try:
            env = self.__dict__["env"]
        except KeyError:
            raise AttributeError(name)
        return getattr(env, name)


class TimeLimit(BaseWrapper):
    def __init__(self, env, duration):
        super().__init__(env)
        self._duration = duration
        self._step = None

    def step(self, action):
        assert self._step is not None, "Must reset environment."
        obs, reward, done, info = self.env.step(action)
        self._step += 1
        if self._step >= self._duration:
            done = True
            if "discount" not in info:
                info["discount"] = np.array(1.0).astype(np.float32)
            self._step = None
        return obs, reward, done, info

    def reset(self):
        self._step = 0
        return self.env.reset()


class NormalizeActions(BaseWrapper):
    def __init__(self, env):
        super().__init__(env)
        self._mask = np.logical_and(
            np.isfinite(env.action_space.low), np.isfinite(env.action_space.high)
        )
        self._low = np.where(self._mask, env.action_space.low, -1)
        self._high = np.where(self._mask, env.action_space.high, 1)
        low = np.where(self._mask, -np.ones_like(self._low), self._low)
        high = np.where(self._mask, np.ones_like(self._low), self._high)
        self._action_space = spaces.Box(low, high, dtype=np.float32)

    @property
    def action_space(self):
        return self._action_space

    def step(self, action):
        original = (action + 1) / 2 * (self._high - self._low) + self._low
        original = np.where(self._mask, original, action)
        return self.env.step(original)


class OneHotAction(BaseWrapper):
    def __init__(self, env):
        assert hasattr(env.action_space, "n"), "OneHotAction requires a discrete action space"
        super().__init__(env)
        self._random = np.random.RandomState()
        shape = (self.env.action_space.n,)
        space = spaces.Box(low=0, high=1, shape=shape, dtype=np.float32)
        space.discrete = True
        self._action_space = space

    @property
    def action_space(self):
        return self._action_space

    def step(self, action):
        index = np.argmax(action).astype(int)
        reference = np.zeros_like(action)
        reference[index] = 1
        if not np.allclose(reference, action):
            raise ValueError(f"Invalid one-hot action:\n{action}")
        return self.env.step(index)

    def reset(self):
        return self.env.reset()

    def _sample_action(self):
        actions = self.env.action_space.n
        index = self._random.randint(0, actions)
        reference = np.zeros(actions, dtype=np.float32)
        reference[index] = 1.0
        return reference


class RewardObs(BaseWrapper):
    def __init__(self, env):
        super().__init__(env)
        obs_spaces = dict(self.env.observation_space.spaces)
        if "obs_reward" not in obs_spaces:
            obs_spaces["obs_reward"] = spaces.Box(
                -np.inf, np.inf, shape=(1,), dtype=np.float32
            )
        self._observation_space = spaces.Dict(obs_spaces)

    @property
    def observation_space(self):
        return self._observation_space

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if "obs_reward" not in obs:
            obs["obs_reward"] = np.array([reward], dtype=np.float32)
        return obs, reward, done, info

    def reset(self):
        obs = self.env.reset()
        if "obs_reward" not in obs:
            obs["obs_reward"] = np.array([0.0], dtype=np.float32)
        return obs


class SelectAction(BaseWrapper):
    def __init__(self, env, key):
        super().__init__(env)
        self._key = key

    def step(self, action):
        return self.env.step(action[self._key])


class UUID(BaseWrapper):
    def __init__(self, env):
        super().__init__(env)
        timestamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        self.id = f"{timestamp}-{str(uuid.uuid4().hex)}"

    def reset(self):
        timestamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        self.id = f"{timestamp}-{str(uuid.uuid4().hex)}"
        return self.env.reset()