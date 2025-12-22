import datetime
import warnings
import embodied
import numpy as np
import ruamel.yaml as yaml
import cv2
import time
from tqdm import tqdm

import car_dreamer
import RL
from car_dreamer.toolkit.utils import get_logger
from car_dreamer.toolkit.deploy_env import VideoEnv
from car_dreamer.toolkit.monitor.monitor import EnvMonitorLocalCV

warnings.filterwarnings("ignore", ".*truncated to dtype int32.*")

def deploy_agent(agent, env, monitor, args):
    log = get_logger(log_dir=str(args.logdir), job_name="deploy")
    log.info("Starting deployment with video input.")
    log.info(f"Args: {args}")

    step = embodied.Counter()
    timer = embodied.Timer()
    timer.wrap("agent", agent, ["policy"])
    timer.wrap("env", env, ["step"])

    # This is a simplified loop for video deployment.
    # We will iterate through the video frames and let the agent predict actions.
    # The actions won't directly affect the video playback, as it's pre-recorded.

    checkpoint = embodied.Checkpoint()
    checkpoint.agent = agent
    if args.from_checkpoint:
        checkpoint.load(args.from_checkpoint, keys=["agent"])
    else:
        raise ValueError("No checkpoint specified for deployment.")

    log.info("Starting deployment loop.")
    policy = lambda *args: agent.policy(*args, mode="eval") # Use eval mode for deployment

    obs = env.reset()
    agent_state = agent.policy_initial(1) # Initial agent state for a single environment

    # Initialize monitor for visualization
    monitor_config = embodied.Config({'display': {'enable': True, 'render_keys': ['camera', 'birdeye_wpt']}})
    env_monitor = EnvMonitorLocalCV(monitor_config)

    with tqdm(total=env._max_frames) as pbar:
        while True:
            # Agent's turn to act
            agent_output, agent_state = policy(obs, agent_state)
            action = agent_output["action"].sample() # Sample action from agent's policy

            # Step the video environment with a dummy action (as it's pre-recorded)
            # The reward and terminal status will be determined by the VideoEnv
            obs, reward, done, info = env.step(action)
            step.increment()

            # Update monitor with current observations and agent's actions
            monitor_info = {
                'action_steer': action[1].item() if len(action) > 1 else 0.0,
                'action_throttle': action[0].item() if len(action) > 0 else 0.0,
                'current_frame': env._current_frame,
                'max_frames': env._max_frames,
                **info, # Include any info from VideoEnv
            }
            # The monitor expects 'speed_norm', 'throttle', 'steer', 'brake'
            # We will map agent's action to these for display purposes
            monitor_info['throttle'] = np.clip(action[0].item(), 0, 1) if len(action) > 0 else 0.0
            monitor_info['steer'] = np.clip(action[1].item(), -1, 1) if len(action) > 1 else 0.0
            monitor_info['brake'] = np.clip(-action[0].item(), 0, 1) if len(action) > 0 and action[0].item() < 0 else 0.0
            monitor_info['speed_norm'] = 0.0 # VideoEnv doesn't provide speed directly

            env_monitor.render(obs, monitor_info)

            pbar.update(1)

            if done:
                log.info(f"Video deployment finished after {env._current_frame} frames.")
                break

            # Introduce a small delay to control playback speed
            # If target_fps is available, calculate delay needed
            if env._target_fps > 0:
                time.sleep(1.0 / env._target_fps)

    env_monitor.close()
    env.close()

def main(argv=None):
    log = get_logger(log_dir=".", job_name="main_deploy")
    model_configs = yaml.YAML(typ="safe").load((embodied.Path(__file__).parent / "dreamerv3.yaml").read())
    config = embodied.Config({"dreamerv3": model_configs["defaults"]})
    config = config.update({"dreamerv3": model_configs["small"]})

    parsed, other = embodied.Flags(
        task=["carla_navigation"], # Dummy task, will be overridden by video paths
        deploy_camera_video_path="",
        deploy_birdeye_video_path="",
        deploy_target_fps=30,
    ).parse_known(argv)

    config = embodied.Flags(config).parse(other)

    logdir = embodied.Path(config.dreamerv3.logdir)
    logdir.mkdirs() # Ensure logdir exists for the deploy script

    # --- Environment Creation ---
    if parsed.deploy_camera_video_path or parsed.deploy_birdeye_video_path:
        log.info("Deploying with video input.")
        deploy_env_config = embodied.Config({
            'camera_video_path': parsed.deploy_camera_video_path,
            'birdeye_video_path': parsed.deploy_birdeye_video_path,
            'target_fps': parsed.deploy_target_fps,
        })
        env = VideoEnv(deploy_env_config)
    else:
        # Fallback to CARLA environment if no video path is provided
        # This part is mostly copied from eval.py but will not be the primary use case here
        log.info(f"Deploying with CARLA environment (task: {parsed.task[0]}).")
        env, env_config = car_dreamer.create_task(parsed.task[0], argv)
        config = config.update(env_config)
        from embodied.envs import from_gym
        env = from_gym.FromGym(env)
        env = wrap_env(env, config.dreamerv3) # wrap with the same wrappers as eval/train
        env = embodied.BatchEnv([env], parallel=False)


    # --- Agent Creation ---
    dreamerv3_config = config.dreamerv3
    # Ensure action space from VideoEnv is compatible. If VideoEnv has a default, it will be used.
    # Otherwise, agent's action space will adapt to actual env.
    agent = RL.Agent(env.obs_space, env.act_space, embodied.Counter(), dreamerv3_config)

    # --- Args for deployment ---
    args = embodied.Config(
        **dreamerv3_config.run,
        logdir=logdir, # Use the logdir created for deploy script
        from_checkpoint=config.dreamerv3.run.from_checkpoint, # Checkpoint path from config
    )
    # Add deploy-specific args
    args = args.update({
        'deploy_camera_video_path': parsed.deploy_camera_video_path,
        'deploy_birdeye_video_path': parsed.deploy_birdeye_video_path,
        'deploy_target_fps': parsed.deploy_target_fps,
    })


    deploy_agent(agent, env, EnvMonitorLocalCV, args) # Pass EnvMonitorLocalCV class directly


if __name__ == "__main__":
    main()
