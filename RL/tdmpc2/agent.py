import jax
import jax.numpy as jnp
from .. import jaxagent, jaxutils, nets
from .. import ninjax as nj

tree_map = jax.tree_util.tree_map
sg = lambda x: tree_map(jax.lax.stop_gradient, x)

@jaxagent.Wrapper
class TDMPC2Agent(nj.Module):
    
    def __init__(self, obs_space, act_space, step, config):
        self.config = config
        self.obs_space = obs_space
        self.act_space = act_space['action']
        self.step = step
        self.wm = TOLD(obs_space, act_space, config, name='told')
        self.opt = jaxutils.Optimizer(name='model_opt', **config.model_opt)

    def policy_initial(self, batch_size):
        return (
            jnp.zeros((batch_size, self.config.horizon, self.act_space.shape[0]), jnp.float32), 
            jnp.ones((batch_size, self.config.horizon, self.act_space.shape[0]), jnp.float32) * self.config.rho, 
        )

    def train_initial(self, batch_size):
        return self.wm.initial(batch_size)

    def policy(self, obs, state, mode='train'):
        obs = self.preprocess(obs)
        prev_mean, prev_std = state
        
        # Ensure planning state is float32
        prev_mean = prev_mean.astype(jnp.float32)
        prev_std = prev_std.astype(jnp.float32)
        
        z = self.wm.encode(obs).astype(jnp.float32)
        batch_size = z.shape[0]
        
        # Shift previous solution for warm-start
        prev_mean = jnp.concatenate([prev_mean[:, 1:], jnp.zeros_like(prev_mean[:, :1])], axis=1)
        prev_std = jnp.ones_like(prev_mean) * self.config.rho
        
        def mppi_iteration(carry, _):
            mean, std = carry 
            rng = nj.rng()
            eps = jax.random.normal(rng, (batch_size, self.config.num_samples, self.config.horizon, self.act_space.shape[0]))
            actions = mean[:, None] + std[:, None] * eps
            actions = jnp.clip(actions, -1.0, 1.0)
            z_init = jnp.repeat(z[:, None], self.config.num_samples, axis=1)
            
            def mppi_step(c, a):
                curr_z = c
                next_z = self.wm.dynamics(curr_z, a).mode().astype(jnp.float32)
                rew = self.wm.reward(curr_z, a).mean().astype(jnp.float32)
                return next_z, rew

            actions_scan = jnp.moveaxis(actions, 2, 0)
            # Use nj.scan for inner rollout
            last_z, rewards = nj.scan(mppi_step, z_init, actions_scan)
            
            pi_action = self.wm.policy(last_z).mode()
            q1 = self.wm.target_q1(last_z, pi_action).mode()
            q2 = self.wm.target_q2(last_z, pi_action).mode()
            q_val = jnp.minimum(q1, q2).astype(jnp.float32)
            
            q_val = q_val.reshape(batch_size, self.config.num_samples)

            discount = float(self.config.discount)
            discounts = jnp.power(discount, jnp.arange(self.config.horizon)) 
            returns = (rewards * discounts[:, None, None]).sum(axis=0) + (discount ** self.config.horizon) * q_val
            
            score = returns
            score = score - score.max(axis=1, keepdims=True)
            weights = jnp.exp(score / float(self.config.mppi_temp)) 
            weights = weights / (weights.sum(axis=1, keepdims=True) + 1e-8)
            
            opt_mean = (actions * weights[:, :, None, None]).sum(axis=1)
            # nj.scan expects (new_carry, output)
            return (opt_mean, std), None

        # Use nj.scan for outer loop
        final_carry, _ = nj.scan(mppi_iteration, (prev_mean, prev_std), jnp.arange(self.config.iterations))
        opt_mean, opt_std = final_carry
        
        action = opt_mean[:, 0]
        if mode == 'explore':
             action = action + jax.random.normal(nj.rng(), action.shape) * float(self.config.expl_noise)
             action = jnp.clip(action, -1.0, 1.0)
        
        return {'action': action}, (opt_mean, opt_std)

    def train(self, data, state):
        data = self.preprocess(data)
        # Trainable modules
        modules = [
            self.wm.encoder, self.wm.projector, 
            self.wm.dynamics_net, self.wm.reward_net, 
            self.wm.q1_net, self.wm.q2_net, self.wm.policy_net
        ]
        mets, (state, outs, metrics) = self.opt(
            modules, self.wm.loss, data, state, has_aux=True
        )
        metrics.update(mets)
        
        # Update Target Networks (EMA)
        self.wm.update_targets()
        
        return outs, state, metrics

    def report(self, data):
        data = self.preprocess(data)
        # Compute Q-values for the batch
        embeds = self.wm.encode(data)
        q1 = self.wm.q1_head(embeds, data['action']).mode()
        q2 = self.wm.q2_head(embeds, data['action']).mode()
        
        report = {
            'action_hist': data['action'],
            'q_mean': jnp.minimum(q1, q2).mean(),
            'q_std': jnp.minimum(q1, q2).std(),
            'reward_mean': data['reward'].mean(),
        }

        # GIF logging
        for key, value in data.items():
            if len(value.shape) == 5: # [B, T, H, W, C]
                # Log first 6 sequences in a grid, convert to uint8 [0, 255]
                images = value[:6]
                if jnp.issubdtype(images.dtype, jnp.floating):
                    images = (images * 255.0).clip(0, 255).astype(jnp.uint8)
                report[f'policy_{key}'] = jaxutils.video_grid(images)
        
        return report

    def preprocess(self, obs):
        obs = obs.copy()
        for key, value in obs.items():
            if key.startswith("log_") or key in ("key", "env_step"):
                continue
            if len(value.shape) > 3 and value.dtype == jnp.uint8:
                value = jaxutils.cast_to_compute(value) / 255.0
            else:
                value = value.astype(jnp.float32)
            obs[key] = value
        obs["cont"] = 1.0 - obs["is_terminal"].astype(jnp.float32)
        return obs


class TOLD(nj.Module):
    def __init__(self, obs_space, act_space, config):
        self.obs_space = obs_space
        self.act_space = act_space['action']
        self.config = config
        
        shapes = {k: tuple(v.shape) for k, v in obs_space.items()}
        # Learning Networks
        self.encoder = nets.MultiEncoder(shapes, **config.encoder, name='enc')
        self.projector = nets.Linear(config.latent_dim, name='proj')
        
        dyn_kw = config.dynamics.copy()
        self.dynamics_net = nets.MLP(config.latent_dim, dyn_kw.pop('layers'), dyn_kw.pop('units'), **dyn_kw, name='dyn')
        
        rew_kw = config.reward_head.copy()
        self.reward_net = nets.MLP((), rew_kw.pop('layers'), rew_kw.pop('units'), **rew_kw, name='rew')
        
        # Ensemble Q
        q1_kw = config.q_head.copy()
        self.q1_net = nets.MLP((), q1_kw.pop('layers'), q1_kw.pop('units'), **q1_kw, name='q1')
        q2_kw = config.q_head.copy()
        self.q2_net = nets.MLP((), q2_kw.pop('layers'), q2_kw.pop('units'), **q2_kw, name='q2')
        
        pi_kw = config.actor.copy()
        self.policy_net = nets.MLP(self.act_space.shape, pi_kw.pop('layers'), pi_kw.pop('units'), **pi_kw, name='pi')
        
        # Target Networks (EMA)
        self.target_encoder = nets.MultiEncoder(shapes, **config.encoder, name='target_enc')
        self.target_projector = nets.Linear(config.latent_dim, name='target_proj')
        
        tq1_kw = config.q_head.copy()
        self.target_q1_net = nets.MLP((), tq1_kw.pop('layers'), tq1_kw.pop('units'), **tq1_kw, name='target_q1')
        tq2_kw = config.q_head.copy()
        self.target_q2_net = nets.MLP((), tq2_kw.pop('layers'), tq2_kw.pop('units'), **tq2_kw, name='target_q2')
        
        # EMA Updaters
        self.updaters = [
            jaxutils.SlowUpdater(self.encoder, self.target_encoder, config.target_ema_decay, config.target_ema_period),
            jaxutils.SlowUpdater(self.projector, self.target_projector, config.target_ema_decay, config.target_ema_period),
            jaxutils.SlowUpdater(self.q1_net, self.target_q1_net, config.target_ema_decay, config.target_ema_period),
            jaxutils.SlowUpdater(self.q2_net, self.target_q2_net, config.target_ema_decay, config.target_ema_period),
        ]
        
    def initial(self, batch_size):
        return ()

    def encode(self, data):
        return self.projector(self.encoder(data))

    def target_encode(self, data):
        return self.target_projector(self.target_encoder(data))

    def dynamics(self, z, a):
        return self.dynamics_net(jnp.concatenate([z, a], -1))

    def reward(self, z, a):
        return self.reward_net(jnp.concatenate([z, a], -1))
        
    def q1_head(self, z, a):
        return self.q1_net(jnp.concatenate([z, a], -1))

    def q2_head(self, z, a):
        return self.q2_net(jnp.concatenate([z, a], -1))
        
    def target_q1(self, z, a):
        return self.target_q1_net(jnp.concatenate([z, a], -1))

    def target_q2(self, z, a):
        return self.target_q2_net(jnp.concatenate([z, a], -1))

    def policy(self, z):
        return self.policy_net(z)

    def update_targets(self):
        for updater in self.updaters:
            updater()

    def loss(self, data, state):
        # 1. Encode sequence (and force float32 for scan consistency)
        embeds = self.encode(data).astype(jnp.float32)
        # 2. Get EMA targets for the whole sequence
        targets_z = sg(self.target_encode(data)).astype(jnp.float32)
        
        actions = data['action'][:, :-1].astype(jnp.float32)
        rewards = data['reward'][:, :-1].astype(jnp.float32)
        
        def step(carry, inp):
            prev_z = carry['z']
            a, target_z, r_target = inp
            
            # Dynamics & Consistency
            pred_z = self.dynamics(prev_z, a).mode().astype(jnp.float32)
            c_loss = jnp.square(pred_z - target_z).mean()
            
            # Reward prediction (Symlog Regression)
            r_dist = self.reward(prev_z, a)
            r_loss = -r_dist.log_prob(r_target).mean()
            
            # Double Q-Learning
            q1_dist = self.q1_head(prev_z, a)
            q2_dist = self.q2_head(prev_z, a)
            
            # Target Q = r + gamma * min(Q1_target, Q2_target)
            next_a = self.policy(sg(target_z)).mode()
            tq1 = self.target_q1(sg(target_z), next_a).mode()
            tq2 = self.target_q2(sg(target_z), next_a).mode()
            target_q = r_target + self.config.discount * jnp.minimum(tq1, tq2).reshape(r_target.shape)
            
            q1_loss = -q1_dist.log_prob(sg(target_q)).mean()
            q2_loss = -q2_dist.log_prob(sg(target_q)).mean()
            
            # Policy Prior (Maximize Q)
            pi_a = self.policy(sg(prev_z)).mode()
            q_pi = jnp.minimum(self.q1_head(sg(prev_z), pi_a).mode(), 
                               self.q2_head(sg(prev_z), pi_a).mode())
            p_loss = -q_pi.mean()
            
            return {
                'z': pred_z,
                'c_loss': c_loss,
                'r_loss': r_loss,
                'q_loss': q1_loss + q2_loss,
                'p_loss': p_loss,
            }

        scan_inputs = (
            jnp.swapaxes(actions, 0, 1),
            jnp.swapaxes(targets_z[:, 1:], 0, 1), # Targets start from t=1
            jnp.swapaxes(rewards, 0, 1)
        )
        
        init_carry = {
            'z': embeds[:, 0],
            'c_loss': jnp.array(0.0, jnp.float32),
            'r_loss': jnp.array(0.0, jnp.float32),
            'q_loss': jnp.array(0.0, jnp.float32),
            'p_loss': jnp.array(0.0, jnp.float32),
        }
        
        # Use nj.scan for loss as well to be safe with state management
        def loss_scan_fn(c, i):
            o = step(c, i)
            return o, o # carry, output
            
        _, out = nj.scan(loss_scan_fn, init_carry, scan_inputs)
        
        # Smoothness Loss
        s_loss = jnp.mean(jnp.square(actions[:, 1:] - actions[:, :-1]))
        
        # Weighted Total Loss
        total_loss = (
            self.config.consistency_loss_scale * jnp.mean(out['c_loss']) +
            self.config.reward_loss_scale * jnp.mean(out['r_loss']) +
            self.config.q_loss_scale * jnp.mean(out['q_loss']) +
            self.config.p_loss_scale * jnp.mean(out['p_loss']) +
            self.config.s_loss_scale * s_loss
        )
        
        metrics = {
            'model_loss': total_loss,
            'cons_loss': jnp.mean(out['c_loss']),
            'rew_loss': jnp.mean(out['r_loss']),
            'q_loss': jnp.mean(out['q_loss']),
            'p_loss': jnp.mean(out['p_loss']),
            's_loss': s_loss,
        }
        
        return total_loss, (state, {}, metrics)