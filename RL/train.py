import datetime
import pathlib
import sys
import warnings

# Add project root to path
root = pathlib.Path(__file__).parent.parent
sys.path.append(str(root))

import embodied
import ruamel.yaml as yaml

import car_dreamer
import RL
from car_dreamer.toolkit.utils import get_logger

log = get_logger(log_dir=".", job_name="train")

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


def main(argv=None):
    # Initial argument parsing to determine method and model size
    temp_flags = embodied.Flags(method="dreamerv3", model_size="small")
    temp_parsed, temp_other = temp_flags.parse_known(argv)
    method = temp_parsed.method
    model_size = temp_parsed.model_size

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
    # DreamerV3 config structure uses a root key (e.g., 'dreamerv3')
    # TD-MPC2 config might be flat or use a different key. 
    # Assuming standard structure: defaults updated by size.
    config = embodied.Config({method: model_configs["defaults"]})
    config = config.update({method: model_configs[model_size]})

    parsed, other = embodied.Flags(task=["carla_navigation"]).parse_known(temp_other)
    for name in parsed.task:
        log.info(f"Using task: {name}")
        env, env_config = car_dreamer.create_task(name, other)
        config = config.update(env_config)
    
    # Filter arguments to prevent mismatch errors
    if method == "tdmpc2":
        other = [arg for arg in other if not arg.startswith("--dreamerv3.")]
    elif method == "dreamerv3":
        other = [arg for arg in other if not arg.startswith("--tdmpc2.")]

    config = embodied.Flags(config).parse(other)
    log.info(config)

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

    from embodied.envs import from_gym

    env = from_gym.FromGym(env)
    env = wrap_env(env, method_config)
    env = embodied.BatchEnv([env], parallel=False)

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    config_filename = f"config_{timestamp}.yaml"
    config.save(str(logdir / config_filename))
    log.info(f"[Train] Config saved to {logdir / config_filename}")

    # Instantiate Agent
    if method == "dreamerv3":
        agent = RL.Agent(env.obs_space, env.act_space, step, method_config)
    elif method == "tdmpc2":
        agent = RL.TDMPC2Agent(env.obs_space, env.act_space, step, method_config)

    replay = embodied.replay.Uniform(
        method_config.batch_length, method_config.replay_size, logdir / "replay"
    )
    args = embodied.Config(
        **method_config.run,
        logdir=method_config.logdir,
        batch_steps=method_config.batch_size * method_config.batch_length,
        actor_dist_disc=method_config.get("actor_dist_disc", "onehot"), # Handle generic keys
    )
    embodied.run.train(agent, env, replay, logger, args)


if __name__ == "__main__":
    main()
