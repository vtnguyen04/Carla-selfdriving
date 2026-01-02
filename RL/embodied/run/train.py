import re
import embodied
import jax
import numpy as np
from tqdm import tqdm
from car_dreamer.toolkit.utils import get_logger

def train(agent, env, replay, logger, args):
    logdir = embodied.Path(args.logdir)
    logdir.mkdirs()
    log = get_logger(log_dir=str(logdir), job_name="train")
    log.info(f"Logdir: {logdir}")
    should_expl = embodied.when.Until(args.expl_until)
    should_train = embodied.when.Ratio(args.train_ratio / args.batch_steps)
    should_log = embodied.when.Clock(args.log_every)
    should_save = embodied.when.Clock(args.save_every)
    should_sync = embodied.when.Every(args.sync_every)
    step = logger.step
    updates = embodied.Counter()
    metrics = embodied.Metrics()
    log.info(f"Observation space:\n{embodied.format(env.obs_space)}")
    log.info(f"Action space:\n{embodied.format(env.act_space)}")

    timer = embodied.Timer()
    timer.wrap("agent", agent, ["policy", "train", "report", "save"])
    timer.wrap("env", env, ["step"])
    timer.wrap("replay", replay, ["add", "save"])
    timer.wrap("logger", logger, ["write"])

    nonzeros = set()

    def per_episode(ep):
        length = len(ep["reward"]) - 1
        score = float(ep["reward"].astype(np.float64).sum())
        sum_abs_reward = float(np.abs(ep["reward"]).astype(np.float64).sum())
        logger.add(
            {
                "length": length,
                "score": score,
                "sum_abs_reward": sum_abs_reward,
                "reward_rate": (np.abs(ep["reward"]) >= 0.5).mean(),
            },
            prefix="episode",
        )
        log.info(f"Episode has {length} steps and return {score:.1f}.")
        stats = {}
        for key in args.log_keys_video:
            if key in ep:
                stats[f"policy_{key}"] = ep[key]
        for key, value in ep.items():
            if not args.log_zeros and key not in nonzeros and (value == 0).all():
                continue
            nonzeros.add(key)
            if re.match(args.log_keys_sum, key):
                stats[f"sum_{key}"] = ep[key].sum()
            if re.match(args.log_keys_mean, key):
                stats[f"mean_{key}"] = ep[key].mean()
            if re.match(args.log_keys_max, key):
                stats[f"max_{key}"] = ep[key].max(0).mean()
        metrics.add(stats, prefix="stats")

    driver = embodied.Driver(env)
    driver.on_episode(lambda ep, ep_info, worker: per_episode(ep))
    driver.on_step(lambda _, __, ___: step.increment())
    driver.on_step(lambda tran, _, worker: replay.add(tran, worker))

    log.info("Prefill train dataset.")
    random_agent = embodied.RandomAgent(env.act_space, args.actor_dist_disc)
    while len(replay) < max(args.batch_steps, args.train_fill):
        driver(random_agent.policy, steps=100)
    logger.add(metrics.result())
    logger.write()

    dataset = agent.dataset(replay.dataset)
    state = [None]  # To be writable from train step function below.
    batch = [None]

    def train_step(_, __, ___):
        for _ in range(should_train(step)):
            with timer.scope("dataset"):
                batch[0] = next(dataset)
            outs, state[0], mets = agent.train(batch[0], state[0])
            metrics.add(mets, prefix="train")

            if getattr(replay, "update_visit_count", False):
                replay.update_visit_count(jax.device_get(batch[0]["env_step"]))

            if "key" in outs:
                replay.prioritize(outs["key"], outs["env_step"], outs["model_loss"], outs["td_error"])

            updates.increment()
        
        # DEBUGGING BLOCK START (HEARTBEAT)
        if int(step) > 0 and int(step) % 500 == 0:
            log.info(f"--- DEBUG HEARTBEAT @ STEP {int(step)} ---")
        # DEBUGGING BLOCK END

        if should_sync(updates):
            agent.sync()
        # DEBUGGING: FORCING LOGGING MORE FREQUENTLY
        if int(step) > 0 and int(step) % 500 == 0: # Force logging every 500 steps
            agg = metrics.result()
            # The batch[0] might be None if no training happened in this step
            # We need to guard this.
            report = {}
            if batch[0] is not None:
                report = agent.report(batch[0])
                report = {k: v for k, v in report.items() if "train/" + k not in agg}
            
            logger.add(agg)
            logger.add(report, prefix="report")
            logger.add(replay.stats, prefix="replay")
            logger.add(timer.stats(), prefix="timer")
            logger.write(fps=True)

    driver.on_step(train_step)

    checkpoint = embodied.Checkpoint(logdir / "checkpoint.ckpt")
    timer.wrap("checkpoint", checkpoint, ["save", "load"])
    checkpoint.step = step
    checkpoint.agent = agent
    # checkpoint.replay = replay # Disabled to save I/O time. Replay buffer will be refilled on resume.
    if args.from_checkpoint:
        checkpoint.load(args.from_checkpoint)
    checkpoint.load_or_save()
    should_save(step)  # Register that we jused saved.

    log.info("Start training loop.")
    driver._state = None
    policy = lambda *args: agent.policy(*args, mode="explore" if should_expl(step) else "train")
    with tqdm(total=args.steps, initial=step.value) as pbar:
        while step < args.steps:
            driver(policy, steps=100)
            pbar.update(100)
            if should_save(step):
                checkpoint.save()
    log.info("Finished training loop. Saving final checkpoint.")
    checkpoint.save_sync()
    logger.write()
