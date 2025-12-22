import jax
import jax.numpy as jnp
import numpy as np

tree_map = jax.tree_util.tree_map
sg = lambda x: tree_map(jax.lax.stop_gradient, x)

from car_dreamer.toolkit.utils import get_logger

logger = get_logger(log_dir=".", job_name="dreamerv3")

from . import behaviors, jaxagent, jaxutils, nets
from . import ninjax as nj


# Main Agent Class (Controller)
@jaxagent.Wrapper
class Agent(nj.Module):
    def __init__(self, obs_space, act_space, step, config):

        # Dynamically adjust grad_heads based on world model type
        if config.world_model == "jepa":
            # In JEPA mode, decoder is not used for loss, so remove it from grad_heads
            new_grad_heads = [h for h in config.grad_heads if h != "decoder"]
            config = config.update({"grad_heads": new_grad_heads})

        self.config = config
        self.obs_space = obs_space
        self.act_space = act_space["action"]
        self.step = step

        # CONDITIONAL WORLD MODEL INSTANTIATION
        if config.world_model == "jepa":
            logger.info("Using JEPA World Model")
            self.wm = JEPAWorldModel(obs_space, act_space, config, name="jepa_wm")
        else:  # 'reconstruction'
            logger.info("Using Reconstruction World Model")
            self.wm = WorldModel(obs_space, act_space, config, name="wm")
        self.task_behavior = getattr(behaviors, config.task_behavior)(
            self.wm, self.act_space, self.config, name="task_behavior"
        )
        if config.expl_behavior == "None":
            self.expl_behavior = self.task_behavior
        else:
            self.expl_behavior = getattr(behaviors, config.expl_behavior)(
                self.wm, self.act_space, self.config, name="expl_behavior"
            )

    def policy_initial(self, batch_size):
        return (
            self.wm.initial(batch_size),
            self.task_behavior.initial(batch_size),
            self.expl_behavior.initial(batch_size),
        )

    def train_initial(self, batch_size):
        return self.wm.initial(batch_size)

    def policy(self, obs, state, mode="train"):
        self.config.jax.jit and logger.info("Tracing policy function.")
        obs = self.preprocess(obs)
        (prev_latent, prev_action), task_state, expl_state = state
        embed = self.wm.encoder(obs)
        latent, _ = self.wm.rssm.obs_step(
            prev_latent, prev_action, embed, obs["is_first"]
        )
        self.expl_behavior.policy(latent, expl_state)
        task_outs, task_state = self.task_behavior.policy(latent, task_state)
        expl_outs, expl_state = self.expl_behavior.policy(latent, expl_state)
        if mode == "eval":
            outs = task_outs
            outs["action"] = outs["action"].sample(seed=nj.rng())
            outs["log_entropy"] = jnp.zeros(outs["action"].shape[:1])
        elif mode == "explore":
            outs = expl_outs
            outs["log_entropy"] = outs["action"].entropy()
            outs["action"] = outs["action"].sample(seed=nj.rng())
        elif mode == "train":
            outs = task_outs
            outs["log_entropy"] = outs["action"].entropy()
            outs["action"] = outs["action"].sample(seed=nj.rng())
        state = ((latent, outs["action"]), task_state, expl_state)
        return outs, state

    def train(self, data, state):
        self.config.jax.jit and logger.info("Tracing train function.")
        metrics = {}
        data = self.preprocess(data)

        # Call the appropriate train function
        state, wm_outs, mets = self.wm.train(data, state)
        metrics.update(mets)

        # Call EMA update for JEPA target networks
        if self.config.world_model == "jepa":
            self.wm.update_target()

        context = {**data, **wm_outs["post"]}
        start = tree_map(lambda x: x.reshape([-1] + list(x.shape[2:])), context)
        _, mets = self.task_behavior.train(self.wm.imagine, start, context)
        metrics.update(mets)
        if self.config.expl_behavior != "None":
            _, mets = self.expl_behavior.train(self.wm.imagine, start, context)
            metrics.update({"expl_" + key: value for key, value in mets.items()})

        # Handle metrics that might not exist in JEPA mode
        if "model_loss_raw" in metrics:
            metrics.update({"model_loss_raw": metrics["model_loss_raw"].mean()})
        if "td_error" in metrics:
            metrics.update({"td_error": metrics["td_error"].mean()})
        if "expl_td_error" in metrics:
            metrics.update({"expl_td_error": metrics["expl_td_error"].mean()})

        return {}, state, metrics

    def report(self, data):
        self.config.jax.jit and logger.info("Tracing report function.")
        data = self.preprocess(data)
        report = {}
        report.update(self.wm.report(data))
        mets = self.task_behavior.report(data)
        report.update({f"task_{k}": v for k, v in mets.items()})
        if self.expl_behavior is not self.task_behavior:
            mets = self.expl_behavior.report(data)
            report.update({f"expl_{k}": v for k, v in mets.items()})
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


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ++ WORLD MODEL DEFINITIONS
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


# -----------------------------------------------------------------------------
# 1. Original Reconstruction-based World Model
# -----------------------------------------------------------------------------
class WorldModel(nj.Module):
    def __init__(self, obs_space, act_space, config):
        self.obs_space = obs_space
        self.act_space = act_space["action"]
        self.config = config
        shapes = {k: tuple(v.shape) for k, v in obs_space.items()}
        shapes = {k: v for k, v in shapes.items() if not k.startswith("log_")}
        self.encoder = nets.MultiEncoder(shapes, **config.encoder, name="enc")
        self.rssm = nets.RSSM(**config.rssm, name="rssm")
        self.heads = {
            "decoder": nets.MultiDecoder(shapes, **config.decoder, name="dec"),
            "reward": nets.MLP((), **config.reward_head, name="rew"),
            "cont": nets.MLP((), **config.cont_head, name="cont"),
        }
        self.opt = jaxutils.Optimizer(name="model_opt", **config.model_opt)
        scales = self.config.loss_scales.copy()
        image, vector = scales.pop("image"), scales.pop("vector")
        scales.update({k: image for k in self.heads["decoder"].cnn_shapes})
        scales.update({k: vector for k in self.heads["decoder"].mlp_shapes})
        self.scales = scales

    def initial(self, batch_size):
        prev_latent = self.rssm.initial(batch_size)
        prev_action = jnp.zeros((batch_size, *self.act_space.shape))
        return prev_latent, prev_action

    def train(self, data, state):
        modules = [self.encoder, self.rssm, *self.heads.values()]
        mets, (state, outs, metrics) = self.opt(
            modules, self.loss, data, state, has_aux=True
        )
        metrics.update(mets)
        return state, outs, metrics

    def loss(self, data, state):
        embed = self.encoder(data)
        prev_latent, prev_action = state
        prev_actions = jnp.concatenate(
            [prev_action[:, None], data["action"][:, :-1]], 1
        )
        post, prior = self.rssm.observe(
            embed, prev_actions, data["is_first"], prev_latent
        )
        dists = {}
        feats = {**post, "embed": embed}
        for name, head in self.heads.items():
            out = head(feats if name in self.config.grad_heads else sg(feats))
            out = out if isinstance(out, dict) else {name: out}
            dists.update(out)
        losses = {}
        losses["dyn"] = self.rssm.dyn_loss(post, prior, **self.config.dyn_loss)
        losses["rep"] = self.rssm.rep_loss(post, prior, **self.config.rep_loss)
        for key, dist in dists.items():
            loss = -dist.log_prob(data[key].astype(jnp.float32))
            assert loss.shape == embed.shape[:2], (key, loss.shape)
            losses[key] = loss
        scaled = {k: v * self.scales[k] for k, v in losses.items()}
        model_loss = sum(scaled.values())
        out = {"embed": embed, "post": post, "prior": prior}
        out.update({f"{k}_loss": v for k, v in losses.items()})
        last_latent = {k: v[:, -1] for k, v in post.items()}
        last_action = data["action"][:, -1]
        state = last_latent, last_action
        metrics = self._metrics(data, dists, post, prior, losses, model_loss)
        metrics["model_loss_raw"] = model_loss
        return model_loss.mean(), (state, out, metrics)

    def imagine(self, policy, start, horizon):
        first_cont = (1.0 - start["is_terminal"]).astype(jnp.float32)
        keys = list(self.rssm.initial(1).keys())
        start = {k: v for k, v in start.items() if k in keys}
        start["action"] = policy(start)

        def step(prev, _):
            prev = prev.copy()
            state = self.rssm.img_step(prev, prev.pop("action"))
            return {**state, "action": policy(state)}

        traj = jaxutils.scan(step, jnp.arange(horizon), start, self.config.imag_unroll)
        traj = {k: jnp.concatenate([start[k][None], v], 0) for k, v in traj.items()}
        cont = self.heads["cont"](traj).mode()
        traj["cont"] = jnp.concatenate([first_cont[None], cont[1:]], 0)
        discount = 1 - 1 / self.config.horizon
        traj["weight"] = jnp.cumprod(discount * traj["cont"], 0) / discount
        return traj

    def report(self, data):
        state = self.initial(len(data["is_first"]))
        report = {}
        report.update(self.loss(data, state)[-1][-1])
        context, _ = self.rssm.observe(
            self.encoder(data)[:6, :5], data["action"][:6, :5], data["is_first"][:6, :5]
        )
        start = {k: v[:, -1] for k, v in context.items()}
        recon = self.heads["decoder"](context)
        openl = self.heads["decoder"](self.rssm.imagine(data["action"][:6, 5:], start))
        for key in self.heads["decoder"].cnn_shapes.keys():
            truth = data[key][:6].astype(jnp.float32)
            model = jnp.concatenate([recon[key].mode()[:, :5], openl[key].mode()], 1)
            error = (model - truth + 1) / 2
            video = jnp.concatenate([truth, model, error], 2)
            report[f"openl_{key}"] = jaxutils.video_grid(video)
        return report

    def _metrics(self, data, dists, post, prior, losses, model_loss):
        entropy = lambda feat: self.rssm.get_dist(feat).entropy()
        metrics = {}
        metrics.update(jaxutils.tensorstats(entropy(prior), "prior_ent"))
        metrics.update(jaxutils.tensorstats(entropy(post), "post_ent"))
        metrics.update({f"{k}_loss": v.mean() for k, v in losses.items()})
        metrics.update({f"{k}_loss_std": v.std() for k, v in losses.items()})
        metrics["model_loss"] = model_loss.mean()
        metrics["model_loss_std"] = model_loss.std()
        metrics["reward_max_data"] = jnp.abs(data["reward"]).max()
        metrics["reward_max_pred"] = jnp.abs(dists["reward"].mean()).max()
        if "reward" in dists and not self.config.jax.debug_nans:
            stats = jaxutils.balance_stats(dists["reward"], data["reward"], 0.1)
            metrics.update({f"reward_{k}": v for k, v in stats.items()})
        if "cont" in dists and not self.config.jax.debug_nans:
            stats = jaxutils.balance_stats(dists["cont"], data["cont"], 0.5)
            metrics.update({f"cont_{k}": v for k, v in stats.items()})
        return metrics


class JEPAWorldModel(nj.Module):
    def __init__(self, obs_space, act_space, config):
        self.config = config
        self.obs_space = obs_space
        self.act_space = act_space["action"]
        self.jepa_cfg = config.jepa_wm

        shapes = {
            k: tuple(v.shape) for k, v in obs_space.items() if not k.startswith("log_")
        }

        self.encoder = nets.MultiEncoder(shapes, **config.encoder, name="enc")
        self.target_encoder = nets.MultiEncoder(
            shapes, **config.encoder, name="target_enc"
        )
        self.rssm = nets.RSSM(**config.rssm, name="rssm")

        # Gom các heads vào dictionary để tương thích với behaviors.py
        self.heads = {
            "reward": nets.MLP((), **config.reward_head, name="rew"),
            "cont": nets.MLP((), **config.cont_head, name="cont"),
        }

        # Projectors và Predictor
        self.online_projector = nets.JEPAProjector(
            output_dim=self.jepa_cfg.proj_dim,
            hidden_dim=self.jepa_cfg.pred_hidden,
            act=self.jepa_cfg.act,
            norm=self.jepa_cfg.norm,
            name="online_proj",
        )
        self.target_projector = nets.JEPAProjector(
            output_dim=self.jepa_cfg.proj_dim,
            hidden_dim=self.jepa_cfg.pred_hidden,
            act=self.jepa_cfg.act,
            norm=self.jepa_cfg.norm,
            name="target_proj",
        )
        self.predictor = nets.JEPAPredictor(
            output_dim=self.jepa_cfg.proj_dim,
            hidden_dim=self.jepa_cfg.pred_hidden,
            layers=self.jepa_cfg.pred_layers,
            act=self.jepa_cfg.act,
            norm=self.jepa_cfg.norm,
            name="predictor",
        )

        self.opt = jaxutils.Optimizer(name="model_opt", **config.model_opt)

        if self.jepa_cfg.jepa_loss_type == "vicreg":
            self._jepa_loss_fn = jaxutils.VICRegLoss()
        else:
            self._jepa_loss_fn = jaxutils.JEPALoss()

    def initial(self, batch_size):
        prev_latent = self.rssm.initial(batch_size)
        prev_action = jnp.zeros((batch_size, *self.act_space.shape))
        return prev_latent, prev_action

    def train(self, data, state):
        # Thêm các head từ dictionary vào danh sách tối ưu hóa
        modules = [
            self.encoder,
            self.rssm,
            self.predictor,
            self.online_projector,
            *self.heads.values(),
        ]
        mets, (state, outs, metrics) = self.opt(
            modules, self.loss, data, state, has_aux=True
        )
        metrics.update(mets)
        return state, outs, metrics

    def loss(self, data, state):
        online_embed = self.encoder(data)
        target_embed = sg(self.target_encoder(data))

        prev_latent, prev_action = state
        prev_actions = jnp.concatenate(
            [prev_action[:, None], data["action"][:, :-1]], 1
        )
        post, prior = self.rssm.observe(
            online_embed, prev_actions, data["is_first"], prev_latent
        )

        # JEPA Loss Logic
        embed_t = online_embed[:, :-1]
        action_t = data["action"][:, :-1]
        target_embed_t1 = target_embed[:, 1:]
        B, T_minus_1 = embed_t.shape[:2]

        z_t = self.online_projector(embed_t.reshape(B * T_minus_1, -1))
        z_target_t1 = sg(
            self.target_projector(target_embed_t1.reshape(B * T_minus_1, -1))
        )
        pred_z_t1 = self.predictor(z_t, action_t.reshape(B * T_minus_1, -1))

        jepa_loss, jepa_metrics = self._jepa_loss_fn(pred_z_t1, z_target_t1)

        # Dynamics losses
        dyn_loss = self.rssm.dyn_loss(post, prior, **self.config.dyn_loss).mean()
        rep_loss = self.rssm.rep_loss(post, prior, **self.config.rep_loss).mean()

        # Reward & Cont losses sử dụng dictionary self.heads
        feats = {**post, "embed": online_embed}
        dists = {}
        for name, head in self.heads.items():
            out = head(feats if name in self.config.grad_heads else sg(feats))
            dists[name] = out

        reward_loss = (
            -dists["reward"].log_prob(data["reward"].astype(jnp.float32)).mean()
        )
        cont_loss = -dists["cont"].log_prob(data["cont"].astype(jnp.float32)).mean()

        total_loss = (
            self.jepa_cfg.jepa_scale * jepa_loss
            + self.config.loss_scales.dyn * dyn_loss
            + self.config.loss_scales.rep * rep_loss
            + self.config.loss_scales.reward * reward_loss
            + self.config.loss_scales.cont * cont_loss
        )

        metrics = {
            **jepa_metrics,
            "dyn_loss": dyn_loss,
            "rep_loss": rep_loss,
            "reward_loss": reward_loss,
            "cont_loss": cont_loss,
            "model_loss": total_loss,
        }

        out = {"embed": online_embed, "post": post, "prior": prior}
        last_latent = {k: v[:, -1] for k, v in post.items()}
        last_action = data["action"][:, -1]
        state = last_latent, last_action

        return total_loss, (state, out, metrics)

    def update_target(self):
        decay = self.jepa_cfg.ema_decay

        # Safe EMA cho Encoder
        online_enc = self.encoder.getm()
        target_enc = self.target_encoder.getm()
        if online_enc and target_enc:
            online_renamed = {
                k.replace("/enc/", "/target_enc/"): v for k, v in online_enc.items()
            }
            new_enc = jaxutils.ema_update(online_renamed, target_enc, decay)
            self.target_encoder.putm(new_enc)

        # Safe EMA cho Projector
        online_proj = self.online_projector.getm()
        target_proj = self.target_projector.getm()
        if online_proj and target_proj:
            online_proj_renamed = {
                k.replace("/online_proj/", "/target_proj/"): v
                for k, v in online_proj.items()
            }
            new_proj = jaxutils.ema_update(online_proj_renamed, target_proj, decay)
            self.target_projector.putm(new_proj)

    def imagine(self, policy, start, horizon):
        first_cont = (1.0 - start["is_terminal"]).astype(jnp.float32)
        keys = list(self.rssm.initial(1).keys())
        start = {k: v for k, v in start.items() if k in keys}
        start["action"] = policy(start)

        def step(prev, _):
            prev = prev.copy()
            state = self.rssm.img_step(prev, prev.pop("action"))
            return {**state, "action": policy(state)}

        traj = jaxutils.scan(step, jnp.arange(horizon), start, self.config.imag_unroll)
        traj = {k: jnp.concatenate([start[k][None], v], 0) for k, v in traj.items()}
        # Sử dụng self.heads['cont'] thay vì self.cont_head
        cont = self.heads["cont"](traj).mode()
        traj["cont"] = jnp.concatenate([first_cont[None], cont[1:]], 0)
        discount = 1 - 1 / self.config.horizon
        traj["weight"] = jnp.cumprod(discount * traj["cont"], 0) / discount
        return traj

    def report(self, data):
        report = {}
        state = self.initial(len(data["is_first"]))
        loss, (state, out, metrics) = self.loss(data, state)
        report.update(metrics)
        return report


# -----------------------------------------------------------------------------
# 3. Original Actor-Critic and V-Function
# -----------------------------------------------------------------------------


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
            name="actor",
            dims="deter",
            shape=act_space.shape,
            **config.actor,
            dist=config.actor_dist_disc if disc else config.actor_dist_cont,
        )
        self.retnorms = {
            k: jaxutils.Moments(**config.retnorm, name=f"retnorm_{k}") for k in critics
        }
        self.opt = jaxutils.Optimizer(name="actor_opt", **config.actor_opt)

    def initial(self, batch_size):
        return {}

    def policy(self, state, carry):
        return {"action": self.actor(state)}, carry

    def train(self, imagine, start, context):
        def loss(start):
            policy = lambda s: self.actor(sg(s)).sample(seed=nj.rng())
            traj = imagine(policy, start, self.config.imag_horizon)
            loss, metrics = self.loss(traj)
            return loss, (traj, metrics)

        mets, (traj, metrics) = self.opt(self.actor, loss, start, has_aux=True)
        metrics.update(mets)
        for key, critic in self.critics.items():
            mets = critic.train(traj, self.actor)
            metrics.update({f"{key}_critic_{k}": v for k, v in mets.items()})
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
            metrics.update(jaxutils.tensorstats(rew, f"{key}_reward"))
            metrics.update(jaxutils.tensorstats(ret, f"{key}_return_raw"))
            metrics.update(jaxutils.tensorstats(normed_ret, f"{key}_return_normed"))
            metrics[f"{key}_return_rate"] = (jnp.abs(ret) >= 0.5).mean()

        r = (
            jnp.reshape(rew[0], (self.config.batch_size, self.config.batch_length))
            if "batch_size" in self.config
            else rew[0]
        )
        v = (
            jnp.reshape(base[0], (self.config.batch_size, self.config.batch_length))
            if "batch_size" in self.config
            else base[0]
        )
        disc = (
            jnp.reshape(
                traj["cont"][0], (self.config.batch_size, self.config.batch_length)
            )
            * (1 - 1 / self.config.horizon)
            if "batch_size" in self.config
            else traj["cont"][0]
        )
        td_error = (
            r[:, :-1] + disc[:, 1:] * v[:, 1:] - v[:, :-1]
            if "batch_size" in self.config
            else 0
        )
        metrics["td_error"] = td_error

        adv = jnp.stack(advs).sum(0)
        policy = self.actor(sg(traj))
        logpi = policy.log_prob(sg(traj["action"]))[:-1]
        loss = {"backprop": -adv, "reinforce": -logpi * sg(adv)}[self.grad]
        ent = policy.entropy()[:-1]
        loss -= self.config.actent * ent
        loss *= sg(traj["weight"])[:-1]
        loss *= self.config.loss_scales.actor
        metrics.update(self._metrics(traj, policy, logpi, ent, adv))
        return loss.mean(), metrics

    def _metrics(self, traj, policy, logpi, ent, adv):
        metrics = {}
        ent = policy.entropy()[:-1]
        rand = (ent - policy.minent) / (policy.maxent - policy.minent)
        rand = rand.mean(range(2, len(rand.shape)))
        act = traj["action"]
        act = jnp.argmax(act, -1) if self.act_space.discrete else act
        metrics.update(jaxutils.tensorstats(act, "action"))
        metrics.update(jaxutils.tensorstats(rand, "policy_randomness"))
        metrics.update(jaxutils.tensorstats(ent, "policy_entropy"))
        metrics.update(jaxutils.tensorstats(logpi, "policy_logprob"))
        metrics.update(jaxutils.tensorstats(adv, "adv"))
        metrics["imag_weight_dist"] = jaxutils.subsample(traj["weight"])
        return metrics


class VFunction(nj.Module):
    def __init__(self, rewfn, config):
        self.rewfn = rewfn
        self.config = config
        self.net = nets.MLP((), name="net", dims="deter", **self.config.critic)
        self.slow = nets.MLP((), name="slow", dims="deter", **self.config.critic)
        self.updater = jaxutils.SlowUpdater(
            self.net,
            self.slow,
            self.config.slow_critic_fraction,
            self.config.slow_critic_update,
        )
        self.opt = jaxutils.Optimizer(name="critic_opt", **self.config.critic_opt)

    def train(self, traj, actor):
        target = sg(self.score(traj)[1])
        mets, metrics = self.opt(self.net, self.loss, traj, target, has_aux=True)
        metrics.update(mets)
        self.updater()
        return metrics

    def loss(self, traj, target):
        metrics = {}
        traj = {k: v[:-1] for k, v in traj.items()}
        dist = self.net(traj)
        loss = -dist.log_prob(sg(target))
        if self.config.critic_slowreg == "logprob":
            reg = -dist.log_prob(sg(self.slow(traj).mean()))
        elif self.config.critic_slowreg == "xent":
            reg = -jnp.einsum(
                "...i,...i->...", sg(self.slow(traj).probs), jnp.log(dist.probs)
            )
        else:
            raise NotImplementedError(self.config.critic_slowreg)
        loss += self.config.loss_scales.slowreg * reg
        loss = (loss * sg(traj["weight"])).mean()
        loss *= self.config.loss_scales.critic
        metrics = jaxutils.tensorstats(dist.mean())
        return loss, metrics

    def score(self, traj, actor=None):
        rew = self.rewfn(traj)
        assert (
            len(rew) == len(traj["action"]) - 1
        ), "should provide rewards for all but last action"
        discount = 1 - 1 / self.config.horizon
        disc = traj["cont"][1:] * discount
        value = self.net(traj).mean()
        vals = [value[-1]]
        interm = rew + disc * value[1:] * (1 - self.config.return_lambda)
        for t in reversed(range(len(disc))):
            vals.append(interm[t] + disc[t] * self.config.return_lambda * vals[-1])
        ret = jnp.stack(list(reversed(vals))[:-1])
        return rew, ret, value[:-1]
