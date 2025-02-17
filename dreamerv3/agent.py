import embodied
import jax
import jax.numpy as jnp
import ruamel.yaml as yaml
tree_map = jax.tree_util.tree_map
stop_gradient = lambda x: tree_map(jax.lax.stop_gradient, x)

import logging
logger = logging.getLogger()
class CheckTypesFilter(logging.Filter):
  def filter(self, record):
    return 'check_types' not in record.getMessage()
logger.addFilter(CheckTypesFilter())

from . import behaviors
from . import jaxagent
from . import jaxutils
from . import nets
from . import ninjax as nj


@jaxagent.Wrapper
class Agent(nj.Module):

  configs = yaml.YAML(typ='safe').load(
      (embodied.Path(__file__).parent / 'configs.yaml').read())

  def __init__(self, obs_space, act_space, step, config):
    self.config = config

    self.obs_space = obs_space
    self.act_space = act_space['action']

    self.step = step

    self.world_model = WorldModel(obs_space, act_space, config, name='wm')

    self.task_behavior = getattr(behaviors, config.task_behavior)(
        self.world_model, self.act_space, self.config, name='task_behavior')

    if config.exploration_behavior == 'None':
      self.exploration_behavior = self.task_behavior
    else:
      self.exploration_behavior = getattr(behaviors, config.exploration_behavior)(
          self.world_model, self.act_space, self.config, name='exploration_behavior')

  def initialize_policy(self, batch_size):
    return (
        self.world_model.initialize(batch_size),
        self.task_behavior.initialize(batch_size),
        self.exploration_behavior.initialize(batch_size))

  def train_initial(self, batch_size):
    return self.world_model.initialize(batch_size)

  def policy(self, observation, state, mode='train'):
    self.config.jax.jit and print('Tracing policy function.')

    (prev_latent, prev_action), task_state, exploration_state = state

    # Preprocess and encode observation
    observation = self.preprocess(observation)
    embedded_observation = self.world_model.encoder(observation)

    # Perform a step
    latent, _ = self.world_model.rssm.observation_step(
        prev_latent, prev_action, embedded_observation, observation['is_first']
    )

    self.exploration_behavior.policy(latent, exploration_state)
    task_outs, task_state = self.task_behavior.policy(latent, task_state)
    exploration_outs, exploration_state = self.exploration_behavior.policy(latent, exploration_state)

    if mode == 'eval':
      outs = task_outs
      outs['action'] = outs['action'].sample(seed=nj.rng())
      outs['log_entropy'] = jnp.zeros(outs['action'].shape[:1])
    elif mode == 'explore':
      outs = exploration_outs
      outs['log_entropy'] = outs['action'].entropy()
      outs['action'] = outs['action'].sample(seed=nj.rng())
    elif mode == 'train':
      outs = task_outs
      outs['log_entropy'] = outs['action'].entropy()
      outs['action'] = outs['action'].sample(seed=nj.rng())

    state = ((latent, outs['action']), task_state, exploration_state)

    return outs, state

  def train(self, data, state):
    self.config.jax.jit and print('Tracing train function.')

    metrics = {}
    data = self.preprocess(data)

    # Train World Model
    state, wm_outs, mets = self.world_model.train(data, state)
    metrics.update(mets)

    # Train Task Behavior
    context = {**data, **wm_outs['post']}
    start = tree_map(lambda x: x.reshape([-1] + list(x.shape[2:])), context)
    _, mets = self.task_behavior.train(self.world_model.imagine, start, context)
    metrics.update(mets)

    # Train Exploration Behavior, if configured
    if self.config.exploration_behavior != 'None':
      _, mets = self.exploration_behavior.train(self.world_model.imagine, start, context)
      metrics.update({'expl_' + key: value for key, value in mets.items()})

    outs = {}
    return outs, state, metrics

  def report(self, data):
    self.config.jax.jit and print('Tracing report function.')
    data = self.preprocess(data)
    report = {}
    report.update(self.world_model.report(data))
    mets = self.task_behavior.report(data)
    report.update({f'task_{k}': v for k, v in mets.items()})
    if self.exploration_behavior is not self.task_behavior:
      mets = self.exploration_behavior.report(data)
      report.update({f'expl_{k}': v for k, v in mets.items()})
    return report

  def preprocess(self, observation):
    observation = observation.copy()
    for key, value in observation.items():
      if key.startswith('log_') or key in ('key',):
        continue
      if len(value.shape) > 3 and value.dtype == jnp.uint8:
        value = jaxutils.cast_to_compute(value) / 255.0
      else:
        value = value.astype(jnp.float32)
      observation[key] = value
    observation['cont'] = 1.0 - observation['is_terminal'].astype(jnp.float32)
    return observation


class WorldModel(nj.Module):

  def __init__(self, obs_space, act_space, config):
    self.obs_space = obs_space
    self.act_space = act_space['action']
    self.config = config

    shapes = {k: tuple(v.shape) for k, v in obs_space.items()}
    shapes = {k: v for k, v in shapes.items() if not k.startswith('log_')}

    self.encoder = nets.MultiEncoder(shapes, **config.encoder, name='enc')
    self.rssm = nets.RSSM(**config.rssm, name='rssm')
    self.heads = {
        'decoder': nets.MultiDecoder(shapes, **config.decoder, name='dec'),
        'reward': nets.MLP((), **config.reward_head, name='rew'),
        'cont': nets.MLP((), **config.cont_head, name='cont')
    }

    self.optimizer = jaxutils.Optimizer(name='model_opt', **config.model_opt)

    loss_scales = self.config.loss_scales.copy()
    image, vector = loss_scales.pop('image'), loss_scales.pop('vector')
    loss_scales.update({k: image for k in self.heads['decoder'].cnn_shapes})
    loss_scales.update({k: vector for k in self.heads['decoder'].mlp_shapes})
    self.loss_scales = loss_scales

  def initialize(self, batch_size):
    prev_latent = self.rssm.initialize(batch_size)
    prev_action = jnp.zeros((batch_size, *self.act_space.shape))
    return prev_latent, prev_action

  def train(self, data, state):
    modules = [self.encoder, self.rssm, *self.heads.values()]
    mets, (state, outs, metrics) = self.optimizer(
        modules, self.loss, data, state, has_aux=True
    )
    metrics.update(mets)
    return state, outs, metrics

  def loss(self, data, state):
    embedded_observation = self.encoder(data)
    prev_latent, prev_action = state
    prev_actions = jnp.concatenate([
        prev_action[:, None], data['action'][:, :-1]], 1
    )
    # Perform RSSM observation
    posterior, prior = self.rssm.observe(
        embedded_observation, prev_actions, data['is_first'], prev_latent
    )

    # Predict output observation, reward, continue distributions
    distributions = {}
    feats = {**posterior, 'embed': embedded_observation}
    for name, head in self.heads.items():
      out = head(feats if name in self.config.grad_heads else stop_gradient(feats))
      out = out if isinstance(out, dict) else {name: out}
      distributions.update(out)

    losses = {}

    # Compute representation/dynamics losses
    losses['dyn'] = self.rssm.dynamics_loss(posterior, prior, **self.config.dynamics_loss)
    losses['rep'] = self.rssm.representation_loss(posterior, prior, **self.config.representation_loss)

    # Compute prediction loss
    for key, dist in distributions.items():
      loss = -dist.log_prob(data[key].astype(jnp.float32))
      assert loss.shape == embedded_observation.shape[:2], (key, loss.shape)
      losses[key] = loss

    # Scale each loss by their configured weight and sum them up
    scaled_losses = {k: v * self.loss_scales[k] for k, v in losses.items()}
    model_loss = sum(scaled_losses.values())

    out = {'embed':  embedded_observation, 'post': posterior, 'prior': prior}
    out.update({f'{k}_loss': v for k, v in losses.items()})

    last_latent_state = {k: v[:, -1] for k, v in posterior.items()}
    last_action = data['action'][:, -1]
    state = last_latent_state, last_action

    metrics = self._metrics(data, distributions, posterior, prior, losses, model_loss)

    return model_loss.mean(), (state, out, metrics)

  def imagine(self, policy, start, horizon):
    first_continue = (1.0 - start['is_terminal']).astype(jnp.float32)
    keys = list(self.rssm.initialize(1).keys())
    start = {k: v for k, v in start.items() if k in keys}
    start['action'] = policy(start)

    # Define a step: Predict the next recurrent state and compute an action using the current policy
    def step(prev_state, _):
      prev_state = prev_state.copy()
      state = self.rssm.imagine_step(prev_state, prev_state.pop('action'))
      return {**state, 'action': policy(state)}

    # Perform steps for the full prediction horizon
    trajectory = jaxutils.scan(
        step, jnp.arange(horizon), start, self.config.imag_unroll
    )
    trajectory = {
        k: jnp.concatenate([start[k][None], v], 0) for k, v in trajectory.items()
    }

    cont = self.heads['cont'](trajectory).mode()
    trajectory['cont'] = jnp.concatenate([first_continue[None], cont[1:]], 0)

    discount = 1 - 1 / self.config.horizon
    trajectory['weight'] = jnp.cumprod(discount * trajectory['cont'], 0) / discount

    return trajectory

  def report(self, data):
    state = self.initialize(len(data['is_first']))
    report = {}
    # Update report with metrics
    report.update(self.loss(data, state)[-1][-1])

    # Only use the first six entries from the batch and first five from the length (?)

    posterior, prior = self.rssm.observe(
        self.encoder(data)[:6, :5],
        data['action'][:6, :5],
        data['is_first'][:6, :5]
    )

    start = {k: v[:, -1] for k, v in posterior.items()}
    reconstructed_observation = self.heads['decoder'](posterior)
    open_loop_predictions = self.heads['decoder'](
        self.rssm.imagine(data['action'][:6, 5:], start)
    )
    
    for key in self.heads['decoder'].cnn_shapes.keys():
      # Take the first six entries from the batch
      truth = data[key][:6].astype(jnp.float32)
      # Add the reconstructed observation from the posterior and the freely (open-loop) predicted one
      model = jnp.concatenate([reconstructed_observation[key].mode()[:, :5], open_loop_predictions[key].mode()], 1)
      error = (model - truth + 1) / 2

      truth = (255*(truth - jnp.min(truth))/jnp.ptp(truth)).astype(int)    

      # Place truth, model and error in a column
      video = jnp.concatenate([truth], 2)
      report[f'openl_{key}'] = jaxutils.video_grid(video)
    return report

  def _metrics(self, data, dists, post, prior, losses, model_loss):
    entropy = lambda feat: self.rssm.get_distribution(feat).entropy()
    metrics = {}
    metrics.update(jaxutils.tensorstats(entropy(prior), 'prior_ent'))
    metrics.update(jaxutils.tensorstats(entropy(post), 'post_ent'))
    metrics.update({f'{k}_loss_mean': v.mean() for k, v in losses.items()})
    metrics.update({f'{k}_loss_std': v.std() for k, v in losses.items()})
    metrics['model_loss_mean'] = model_loss.mean()
    metrics['model_loss_std'] = model_loss.std()
    metrics['reward_max_data'] = jnp.abs(data['reward']).max()
    metrics['reward_max_pred'] = jnp.abs(dists['reward'].mean()).max()
    if 'reward' in dists and not self.config.jax.debug_nans:
      stats = jaxutils.balance_stats(dists['reward'], data['reward'], 0.1)
      metrics.update({f'reward_{k}': v for k, v in stats.items()})
    if 'cont' in dists and not self.config.jax.debug_nans:
      stats = jaxutils.balance_stats(dists['cont'], data['cont'], 0.5)
      metrics.update({f'cont_{k}': v for k, v in stats.items()})
    return metrics


class ImagActorCritic(nj.Module):

  def __init__(self, critics, scales, act_space, config):
    critics = {k: v for k, v in critics.items() if scales[k]}
    for key, scale in scales.items():
      assert not scale or key in critics, key
    self.critics = {k: v for k, v in critics.items() if scales[k]}
    self.scales = scales
    self.act_space = act_space
    self.config = config
    disc = act_space.discrete
    self.grad = config.actor_grad_disc if disc else config.actor_grad_cont
    self.actor = nets.MLP(
        name='actor', dims='deter', shape=act_space.shape, **config.actor,
        dist=config.actor_dist_disc if disc else config.actor_dist_cont)
    self.retnorms = {
        k: jaxutils.Moments(**config.retnorm, name=f'retnorm_{k}')
        for k in critics}
    self.optimizer = jaxutils.Optimizer(name='actor_opt', **config.actor_opt)

  def initialize(self, batch_size):
    return {}

  def policy(self, state, carry):
    return {'action': self.actor(state)}, carry

  def train(self, imagine, start, context):
    def loss(start):
      policy = lambda s: self.actor(stop_gradient(s)).sample(seed=nj.rng())
      traj = imagine(policy, start, self.config.imag_horizon)
      loss, metrics = self.loss(traj)
      return loss, (traj, metrics)
    mets, (traj, metrics) = self.optimizer(self.actor, loss, start, has_aux=True)
    metrics.update(mets)
    for key, critic in self.critics.items():
      mets = critic.train(traj, self.actor)
      metrics.update({f'{key}_critic_{k}': v for k, v in mets.items()})
    return traj, metrics

  def loss(self, traj):
    metrics = {}
    advs = []
    total = sum(self.scales[k] for k in self.critics)
    for key, critic in self.critics.items():
      rew, ret, base = critic.score(traj, self.actor)
      offset, invscale = self.retnorms[key](ret)
      normed_ret = (ret - offset) / invscale
      normed_base = (base - offset) / invscale
      advs.append((normed_ret - normed_base) * self.scales[key] / total)
      metrics.update(jaxutils.tensorstats(rew, f'{key}_reward'))
      metrics.update(jaxutils.tensorstats(ret, f'{key}_return_raw'))
      metrics.update(jaxutils.tensorstats(normed_ret, f'{key}_return_normed'))
      metrics[f'{key}_return_rate'] = (jnp.abs(ret) >= 0.5).mean()
    adv = jnp.stack(advs).sum(0)
    policy = self.actor(stop_gradient(traj))
    logpi = policy.log_prob(stop_gradient(traj['action']))[:-1]
    loss = {'backprop': -adv, 'reinforce': -logpi * stop_gradient(adv)}[self.grad]
    entropy = policy.entropy()[:-1]
    loss -= self.config.actent * entropy
    loss *= stop_gradient(traj['weight'])[:-1]
    loss *= self.config.loss_scales.actor
    metrics.update(self._metrics(traj, policy, logpi, entropy, adv))
    return loss.mean(), metrics

  def _metrics(self, traj, policy, logpi, ent, adv):
    metrics = {}
    ent = policy.entropy()[:-1]
    rand = (ent - policy.minent) / (policy.maxent - policy.minent)
    rand = rand.mean(range(2, len(rand.shape)))
    act = traj['action']
    act = jnp.argmax(act, -1) if self.act_space.discrete else act
    metrics.update(jaxutils.tensorstats(act, 'action'))
    metrics.update(jaxutils.tensorstats(rand, 'policy_randomness'))
    metrics.update(jaxutils.tensorstats(ent, 'policy_entropy'))
    metrics.update(jaxutils.tensorstats(logpi, 'policy_logprob'))
    metrics.update(jaxutils.tensorstats(adv, 'adv'))
    metrics['imag_weight_dist'] = jaxutils.subsample(traj['weight'])
    return metrics


class ValueFunction(nj.Module):

  def __init__(self, reward_function, config):
    self.reward_function = reward_function
    self.config = config
    self.net = nets.MLP((), name='net', dims='deter', **self.config.critic)
    self.slow = nets.MLP((), name='slow', dims='deter', **self.config.critic)
    self.updater = jaxutils.SlowUpdater(
        self.net, self.slow,
        self.config.slow_critic_fraction,
        self.config.slow_critic_update)
    self.optimizer = jaxutils.Optimizer(name='critic_opt', **self.config.critic_opt)

  def train(self, trajectory, actor):
    target = stop_gradient(self.score(trajectory)[1])
    mets, metrics = self.optimizer(self.net, self.loss, trajectory, target, has_aux=True)
    metrics.update(mets)
    self.updater()
    return metrics

  def loss(self, trajectory, target):
    metrics = {}
    trajectory = {k: v[:-1] for k, v in trajectory.items()}
    dist = self.net(trajectory)
    loss = -dist.log_prob(stop_gradient(target))
    if self.config.critic_slowreg == 'logprob':
      reg = -dist.log_prob(stop_gradient(self.slow(trajectory).mean()))
    elif self.config.critic_slowreg == 'xent':
      reg = -jnp.einsum(
          '...i,...i->...',
          stop_gradient(self.slow(trajectory).probs),
          jnp.log(dist.probs))
    else:
      raise NotImplementedError(self.config.critic_slowreg)
    loss += self.config.loss_scales.slowreg * reg
    loss = (loss * stop_gradient(trajectory['weight'])).mean()
    loss *= self.config.loss_scales.critic
    metrics = jaxutils.tensorstats(dist.mean())
    return loss, metrics

  def score(self, traj, actor=None):
    rewards = self.reward_function(traj)
    assert len(rewards) == len(traj['action']) - 1, (
        'reward function should provide rewards for all but last action')
    discount = 1 - 1 / self.config.horizon
    disc = traj['cont'][1:] * discount
    value = self.net(traj).mean()
    vals = [value[-1]]
    intermediate = rewards + disc * value[1:] * (1 - self.config.return_lambda)
    for t in reversed(range(len(disc))):
      vals.append(intermediate[t] + disc[t] * self.config.return_lambda * vals[-1])
    ret = jnp.stack(list(reversed(vals))[:-1])
    return rewards, ret, value[:-1]
