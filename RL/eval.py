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
    # Add project root to path
    import pathlib
    import sys
    root = pathlib.Path(__file__).parent.parent
    sys.path.append(str(root))

    # Initial argument parsing
    if argv is None:
        argv = sys.argv[1:]
    
    temp_flags = embodied.Flags(method="dreamerv3", model_size="small")
    temp_parsed, temp_other = temp_flags.parse_known(argv)
    method = temp_parsed.method
    model_size = temp_parsed.model_size

    log = get_logger(log_dir=".", job_name="main_eval")
    
    # Load configuration based on method
    if method == "dreamerv3":
        config_path = embodied.Path(__file__).parent / "dreamerv3/dreamerv3.yaml"
    elif method == "tdmpc2":
        config_path = embodied.Path(__file__).parent / "tdmpc2/tdmpc2.yaml"
    else:
        raise ValueError(f"Unknown method: {method}")

    model_configs = yaml.YAML(typ="safe").load(config_path.read())
    
    if model_size not in model_configs:
        raise ValueError(f"Unknown model_size: {model_size}. Available: {list(model_configs.keys())}")

    # Initialize Config
    config = embodied.Config({method: model_configs["defaults"]})
    config = config.update({method: model_configs[model_size]})

    # Parse task early to load environment configs
    pre_parsed, other_argv = embodied.Flags(
        task=["carla_navigation"]
    ).parse_known(temp_other)

    task_name = pre_parsed.task[0]
    log.info(f"Using task: {task_name}")
    env_config = car_dreamer.load_task_configs(task_name)
    config = config.update(env_config)

    # Load and merge hires config if requested
    if "--hires" in other_argv:
        log.info("High-resolution mode requested. Loading eval_hires.yaml.")
        hires_path = embodied.Path(__file__).parent / "eval_hires.yaml"
        hires_config = yaml.YAML(typ="safe").load(hires_path.read())
        config = config.update(hires_config)
        other_argv = [arg for arg in other_argv if arg != "--hires"]

    # Filter arguments to prevent mismatch errors
    if method == "tdmpc2":
        other_argv = [arg for arg in other_argv if not arg.startswith("--dreamerv3.")]
    elif method == "dreamerv3":
        other_argv = [arg for arg in other_argv if not arg.startswith("--tdmpc2.")]

    # Final parse of all remaining command-line flags
    config = embodied.Flags(config).parse(other_argv)

    # Logdir setup
    method_config = config[method]
    logdir = embodied.Path(method_config.logdir)
    step = embodied.Counter()
    logger = embodied.Logger(
        step,
        [
            embodied.logger.TerminalOutput(pattern=r".*reward.*|.*return.*|.*loss.*"),
            embodied.logger.JSONLOutput(logdir, "metrics.jsonl"),
            embodied.logger.TensorBoardOutput(logdir),
        ],
    )

    # Create and wrap the environment
    import gym
    from embodied.envs import from_gym

    env = gym.make(config.env.name, config=config.env)
    env = from_gym.FromGym(env)
    env = wrap_env(env, method_config)
    env = embodied.BatchEnv([env], parallel=False)

    # Update config with runtime values
    method_config = method_config.update(
        {
            "run.log_keys_sum": "(travel_distance|destination_reached|out_of_lane|time_exceeded|is_collision|timesteps)",
            "run.log_keys_mean": "(travel_distance|ttc|speed_norm|wpt_dis)",
            "run.log_keys_max": "(travel_distance|ttc|speed_norm|wpt_dis)",
            "run.steps": 5e4,
            "run.eval_episodes": -1,
        }
    )

    # Instantiate Agent
    if method == "dreamerv3":
        agent = RL.Agent(env.obs_space, env.act_space, step, method_config)
    elif method == "tdmpc2":
        agent = RL.TDMPC2Agent(env.obs_space, env.act_space, step, method_config)

    args = embodied.Config(
        **method_config.run,
        logdir=method_config.logdir,
        batch_steps=method_config.batch_size * method_config.batch_length,
    )
    eval_only(agent, env, logger, args)


if __name__ == "__main__":
    main()
