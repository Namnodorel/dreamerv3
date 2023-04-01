import collections

import numpy as np

from .basics import convert


class Driver:

  _CONVERSION = {
      np.floating: np.float32,
      np.signedinteger: np.int32,
      np.uint8: np.uint8,
      bool: bool,
  }

  def __init__(self, env, **kwargs):
    assert len(env) > 0
    self._env = env
    self._kwargs = kwargs
    self._on_steps = []
    self._on_episodes = []
    self.reset()

  def reset(self):
    self._acts = {
        k: convert(np.zeros((len(self._env),) + v.shape, v.dtype))
        for k, v in self._env.act_space.items()}
    self._acts['reset'] = np.ones(len(self._env), bool)
    self._eps = [collections.defaultdict(list) for _ in range(len(self._env))]
    self._state = None

  def on_step(self, callback):
    self._on_steps.append(callback)

  def on_episode(self, callback):
    self._on_episodes.append(callback)

  def __call__(self, policy, steps=0, episodes=0):
    step, episode = 0, 0
    while step < steps or episode < episodes:
      step, episode = self._step(policy, step, episode)

  def _step(self, policy, step, episode):
    assert all(len(x) == len(self._env) for x in self._acts.values())

    # Perform stored action and retrieve observation
    acts = {k: v for k, v in self._acts.items() if not k.startswith('log_')}
    obs = self._env.step(acts)
    obs = {k: convert(v) for k, v in obs.items()}

    assert all(len(x) == len(self._env) for x in obs.values()), obs

    # Compute the next action and internal state based on the observation
    acts, self._state = policy(obs, self._state, **self._kwargs)

    acts = {k: convert(v) for k, v in acts.items()}

    # Store reset actions for all observations that indicated that the environment ended the episode
    if obs['is_last'].any():
      mask = 1 - obs['is_last']
      acts = {k: v * self._expand(mask, len(v.shape)) for k, v in acts.items()}
    acts['reset'] = obs['is_last'].copy()

    # Store action for the next step
    self._acts = acts

    transitions = {**obs, **acts}

    # Clear episode history for any environment that was just restarted
    if obs['is_first'].any():
      for env_index, first in enumerate(obs['is_first']):
        if first:
          self._eps[env_index].clear()

    # Call on step listeners
    for env_index in range(len(self._env)):
      transition = {k: v[env_index] for k, v in transitions.items()}
      [self._eps[env_index][k].append(v) for k, v in transition.items()]
      [fn(transition, self._state, env_index, **self._kwargs) for fn in self._on_steps]
      # [fn(transition, env_index, **self._kwargs) for fn in self._on_steps]
      step += 1

    # Increase episode counter if necessary
    if obs['is_last'].any():
      for env_index, done in enumerate(obs['is_last']):
        if done:
          ep = {k: convert(v) for k, v in self._eps[env_index].items()}
          [fn(ep.copy(), env_index, **self._kwargs) for fn in self._on_episodes]
          episode += 1

    return step, episode

  def _expand(self, value, dims):
    while len(value.shape) < dims:
      value = value[..., None]
    return value
