import re
import sys
import warnings
import embodied
import numpy as np
import ruamel.yaml as yaml
from tqdm import tqdm
import car_dreamer
import RL
from car_dreamer.toolkit.utils import get_logger

warnings.filterwarnings("ignore", ".*truncated to dtype int32.*")


def wrap_env(env, config):
    args = config.wrapper
    env = embodied.wrappers.InfoWrapper(env)
    for name, space in env.act_space.items():
        if name == "reset":
            continue
        elif space.discrete:
            env = embodied.wrappers.OneHotAction(env, name)
        elif args.discretize:
            env = embodied.wrappers.DiscretizeAction(env, name, args.discretize)
        else:
            env = embodied.wrappers.NormalizeAction(env, name)
    env = embodied.wrappers.ExpandScalars(env)
    if args.length:
        env = embodied.wrappers.TimeLimit(env, args.length, args.reset)
    if args.checks:
        env = embodied.wrappers.CheckSpaces(env)
    for name, space in env.act_space.items():
        if not space.discrete:
            env = embodied.wrappers.ClipAction(env, name)
    return env


def eval_only(agent, env, logger, args):
    logdir = embodied.Path(args.logdir)
    logdir.mkdirs()
    log = get_logger(log_dir=str(logdir), job_name="eval")
    log.info("Start evaluation.")
    log.info(f"Args: {args}")
    log.info(f"Logdir: {logdir}")
    step = logger.step
    metrics = embodied.Metrics()
    log.info(f"Observation space: {env.obs_space}")
    log.info(f"Action space: {env.act_space}")

    timer = embodied.Timer()
    timer.wrap("agent", agent, ["policy"])
    timer.wrap("env", env, ["step"])
    timer.wrap("logger", logger, ["write"])

    nonzeros = set()

    def per_episode(ep, ep_info):
        length = len(ep["reward"]) - 1
        score = float(ep["reward"].astype(np.float64).sum())
        logger.add({"length": length, "score": score}, prefix="episode")
        log.info(f"Episode has {length} steps and return {score:.1f}.")
        stats = {}
        for key in args.log_keys_video:
            if key in ep:
                stats[f"policy_{key}"] = ep[key]

        def log_stats(key, value):
            if re.match(args.log_keys_sum, key):
                stats[f"sum_{key}"] = value.sum()
            if re.match(args.log_keys_mean, key):
                stats[f"mean_{key}"] = value.mean()
            if re.match(args.log_keys_max, key):
                stats[f"max_{key}"] = value.max(0).mean()

        for key, value in ep.items():
            if not args.log_zeros and key not in nonzeros and (value == 0).all():
                continue
            nonzeros.add(key)
            log_stats(key, value)
        for key, value in ep_info.items():
            log_stats(key, value)

        logger.add(metrics.result())
        logger.add(timer.stats(), prefix="timer")
        logger.write(fps=True)

        metrics.add(stats, prefix="stats")

    def per_step(tran):
        step.increment()

    driver = embodied.Driver(env)
    driver.on_episode(lambda ep, ep_info, worker: per_episode(ep, ep_info))
    driver.on_step(lambda tran, info, _: per_step(step))

    episode_count = embodied.Counter()  # Thêm bộ đếm episode
    driver.on_episode(
        lambda *args: episode_count.increment()
    )  # Tăng bộ đếm mỗi khi kết thúc episode

    checkpoint = embodied.Checkpoint()
    checkpoint.agent = agent
    if args.from_checkpoint:
        checkpoint.load(args.from_checkpoint, keys=["agent"])
    else:
        raise ValueError("No checkpoint specified.")

    log.info("Start evaluation loop.")
    policy = lambda *args: agent.policy(*args, mode="eval")
    with tqdm(
        total=args.steps if args.eval_episodes < 0 else args.eval_episodes
    ) as pbar:
        while step < args.steps and (
            args.eval_episodes < 0 or episode_count.value < args.eval_episodes
        ):
            driver(policy, steps=100)
            pbar.update(100 if args.eval_episodes < 0 else 1)
    logger.write()


def main(argv=None):
    # BOILERPLATE START
    log = get_logger(log_dir=".", job_name="main_eval")
    model_configs = yaml.YAML(typ="safe").load(
        (embodied.Path(__file__).parent / "dreamerv3.yaml").read()
    )
    if argv is None:
        argv = sys.argv[1:]
    # BOILERPLATE END

    # 1. Parse flags that determine config files (model_size, task)
    pre_parsed, other_argv = embodied.Flags(
        model_size="small", task=["carla_navigation"]
    ).parse_known(argv)

    # 2. Build base config from dreamerv3.yaml (defaults + model_size)
    model_size = pre_parsed.model_size
    if model_size not in model_configs:
        raise ValueError(
            f"Unknown model_size: {model_size}. Available: {list(model_configs.keys())}"
        )
    config = embodied.Config({"dreamerv3": model_configs["defaults"]})
    config = config.update({"dreamerv3": model_configs[model_size]})

    # 3. Load and merge task-specific configs
    task_name = pre_parsed.task[0]  # Assuming one task for eval
    log.info(f"Using task: {task_name}")
    env_config = car_dreamer.load_task_configs(task_name)
    config = config.update(env_config)

    # 4. Load and merge hires config if requested
    if "--hires" in other_argv:
        log.info("High-resolution mode requested. Loading eval_hires.yaml.")
        hires_config = yaml.YAML(typ="safe").load(
            (embodied.Path(__file__).parent / "eval_hires.yaml").read()
        )
        config = config.update(hires_config)
        # Remove the flag so the final parser doesn't see it
        other_argv = [arg for arg in other_argv if arg != "--hires"]

    # 5. Final parse of all remaining command-line flags
    # This uses the fully merged config to parse remaining overrides.
    config = embodied.Flags(config).parse(other_argv)

    # 6. Create logger
    logdir = embodied.Path(config.dreamerv3.logdir)
    step = embodied.Counter()
    logger = embodied.Logger(
        step,
        [
            embodied.logger.TerminalOutput(pattern=r".*reward.*|.*return.*|.*loss.*"),
            embodied.logger.JSONLOutput(logdir, "metrics.jsonl"),
            embodied.logger.TensorBoardOutput(logdir),
        ],
    )

    # 7. Create and wrap the environment *with the final config*
    import gym
    from embodied.envs import from_gym

    env = gym.make(config.env.name, config=config.env)
    dreamerv3_config = config.dreamerv3  # for wrappers and agent
    env = from_gym.FromGym(env)
    env = wrap_env(env, dreamerv3_config)
    env = embodied.BatchEnv([env], parallel=False)

    # 8. Update config with runtime values
    dreamerv3_config = dreamerv3_config.update(
        {
            "run.log_keys_sum": "(travel_distance|destination_reached|out_of_lane|time_exceeded|is_collision|timesteps)",
            "run.log_keys_mean": "(travel_distance|ttc|speed_norm|wpt_dis)",
            "run.log_keys_max": "(travel_distance|ttc|speed_norm|wpt_dis)",
            "run.steps": 5e4,
            "run.eval_episodes": -1,
        }
    )

    # 9. Create agent and run evaluation
    agent = RL.Agent(env.obs_space, env.act_space, step, dreamerv3_config)
    args = embodied.Config(
        **dreamerv3_config.run,
        logdir=dreamerv3_config.logdir,
        batch_steps=dreamerv3_config.batch_size * dreamerv3_config.batch_length,
    )
    eval_only(agent, env, logger, args)


if __name__ == "__main__":
    main()
