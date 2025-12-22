import datetime
import warnings
import embodied
import ruamel.yaml as yaml
from tqdm import tqdm
import numpy as np
import pickle # Import pickle

import car_dreamer
import dreamerv3
from car_dreamer.toolkit.utils import get_logger

log = get_logger(log_dir=".", job_name="train_offline")

warnings.filterwarnings("ignore", ".*truncated to dtype int32.*")


def train_offline(agent, replay, logger, args):
    logdir = embodied.Path(args.logdir)
    log.info(f"Logdir: {logdir}")

    should_train = embodied.when.Ratio(args.train_ratio / args.batch_steps)
    should_log = embodied.when.Clock(args.log_every)
    should_save = embodied.when.Clock(args.save_every)
    should_sync = embodied.when.Every(args.sync_every)
    step = logger.step
    updates = embodied.Counter()
    metrics = embodied.Metrics()
    
    timer = embodied.Timer()
    timer.wrap("agent", agent, ["train", "report", "save"])
    timer.wrap("replay", replay, ["add", "save"]) # Replay add not used here, but for consistency
    timer.wrap("logger", logger, ["write"])

    log.info("Loading replay buffer.")
    try:
        replay.load(logdir / "replay", continue_obs=False)
        log.info(f"Loaded existing replay buffer with {len(replay)} steps.")
    except Exception as e:
        log.error(f"Could not load replay buffer from {logdir / 'replay'}: {e}")
        return # Exit if replay buffer cannot be loaded

    dataset = agent.dataset(replay.dataset)
    state = [None]
    batch = [None]

    checkpoint = embodied.Checkpoint(logdir / "checkpoint.ckpt")
    checkpoint.step = step
    checkpoint.agent = agent
    checkpoint.replay = replay # To save replay stats etc, not the actual buffer for training.
    if args.from_checkpoint and args.from_checkpoint != 'none':
        try:
            checkpoint.load(args.from_checkpoint, keys=['agent', 'step'])
            log.info(f"Loaded agent checkpoint from {args.from_checkpoint}")
        except Exception as e:
            log.error(f"Could not load agent checkpoint from {args.from_checkpoint}: {e}")
            log.info("Starting training from scratch.")

    checkpoint.load_or_save() # Loads step, agent, replay if checkpoint exists, otherwise saves initial state.
    should_save(step) # Register that we just saved (or loaded).

    log.info("Start training loop.")
    with tqdm(total=args.steps, initial=int(step.value)) as pbar:
        while step < args.steps:
            # We are not interacting with env, so we directly increment step by batch_steps
            # for each training iteration.
            num_updates = should_train(step)
            if num_updates == 0:
                step.increment(args.batch_steps) # Increment to avoid infinite loop
                pbar.update(args.batch_steps)
                continue

            for _ in range(num_updates):
                with timer.scope("dataset"):
                    batch[0] = next(dataset)
                outs, state[0], mets = agent.train(batch[0], state[0])
                metrics.add(mets, prefix="train")
                updates.increment()
            
            # The original embodied.run.train increments step for each environment interaction.
            # Here, we increment step for each batch of training.
            step.increment(args.batch_steps * num_updates) 
            pbar.update(args.batch_steps * num_updates)


            if should_sync(updates):
                agent.sync()
            
            if should_log(step):
                agg = metrics.result()
                report = {}
                if batch[0] is not None: # Ensure batch is not None before reporting
                    report = agent.report(batch[0])
                    report = {k: v for k, v in report.items() if "train/" + k not in agg}
                logger.add(agg)
                logger.add(report, prefix="report")
                logger.add(replay.stats, prefix="replay") # replay.stats should contain info even if not saving the buffer itself
                logger.add(timer.stats(), prefix="timer")
                logger.write(fps=True)

            if should_save(step):
                checkpoint.save()

    logger.write()

def main(argv=None):
    model_configs = yaml.YAML(typ="safe").load(
        (embodied.Path(__file__).parent / "dreamerv3.yaml").read()
    )
    temp_parsed, temp_other = embodied.Flags(model_size="small").parse_known(argv)
    model_size = temp_parsed.model_size
    if model_size not in model_configs:
        raise ValueError(f"Unknown model_size: {model_size}. Available: {list(model_configs.keys())}")

    config = embodied.Config({"dreamerv3": model_configs["defaults"]})
    config = config.update({"dreamerv3": model_configs[model_size]})
    
    # Load spaces from file
    logdir = embodied.Path(config.dreamerv3.logdir)
    spaces_path = logdir / 'spaces.pkl'
    try:
        with open(spaces_path, 'rb') as f:
            spaces = pickle.load(f)
        obs_space = spaces['obs_space']
        act_space = spaces['act_space']
        log.info(f"Loaded observation and action spaces from {spaces_path}")
    except FileNotFoundError:
        log.error(f"Spaces file not found at {spaces_path}. Cannot create agent.")
        log.error("Please run the `collect.py` script first to generate the spaces.pkl file.")
        return # Exit if spaces cannot be loaded
    except Exception as e:
        log.error(f"Error loading spaces from {spaces_path}: {e}")
        return

    # Add a flag for from_checkpoint
    config = config.update({"dreamerv3.from_checkpoint": "none"})
    config = embodied.Flags(config).parse(other) # Use 'other' which contains remaining args
    log.info(config)

    step = embodied.Counter()
    logger = embodied.Logger(
        step,
        [
            embodied.logger.TerminalOutput(pattern=r".*reward.*|.*return.*|.*loss.*"),
            embodied.logger.JSONLOutput(logdir, "metrics.jsonl"),
            embodied.logger.TensorBoardOutput(logdir),
        ],
    )

    dreamerv3_config = config.dreamerv3

    agent = dreamerv3.Agent(obs_space, act_space, step, dreamerv3_config)
    replay = embodied.replay.Uniform(
        dreamerv3_config.batch_length, dreamerv3_config.replay_size, logdir / "replay"
    )
    
    args = embodied.Config(
        **dreamerv3_config.run,
        logdir=dreamerv3_config.logdir,
        batch_steps=dreamerv3_config.batch_size * dreamerv3_config.batch_length,
        from_checkpoint=dreamerv3_config.from_checkpoint,
        actor_dist_disc=dreamerv3_config.actor_dist_disc,
    )
    train_offline(agent, replay, logger, args)

if __name__ == "__main__":
    main()
