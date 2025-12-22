import gym
import numpy as np
import cv2
from gym import spaces
from typing import Dict, Tuple

from car_dreamer.toolkit.utils import get_logger

log = get_logger(log_dir=".", job_name="video_env")

class VideoEnv(gym.Env):
    """
    A Gym environment that reads observations from video files instead of a simulator.
    This is used for deploying a trained agent on pre-recorded video data.
    """
    metadata = {'render_modes': ['human'], 'render_fps': 30} # Placeholder metadata

    def __init__(self, config):
        super().__init__()
        self._config = config
        self._camera_video_path = config.get('camera_video_path')
        self._birdeye_video_path = config.get('birdeye_video_path')
        self._target_fps = config.get('target_fps', 30) # Target FPS for video playback

        if not self._camera_video_path and not self._birdeye_video_path:
            raise ValueError("At least one video path (camera or birdeye) must be provided.")

        self._cap_camera = None
        self._cap_birdeye = None
        self._current_frame = 0
        self._max_frames = 0
        self._initial_obs_shape = {}

        # Initialize observation space based on the first frame of videos
        self._initialize_video_captures()
        self.observation_space = self._get_observation_space()
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32) # Placeholder action space, will be overridden by agent

        log.info(f"VideoEnv initialized. Camera: {self._camera_video_path}, Birdeye: {self._birdeye_video_path}")

    def _initialize_video_captures(self):
        if self._camera_video_path:
            self._cap_camera = cv2.VideoCapture(self._camera_video_path)
            if not self._cap_camera.isOpened():
                raise IOError(f"Cannot open camera video file: {self._camera_video_path}")
            width = int(self._cap_camera.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self._cap_camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = self._cap_camera.get(cv2.CAP_PROP_FPS)
            self._max_frames = max(self._max_frames, int(self._cap_camera.get(cv2.CAP_PROP_FRAME_COUNT)))
            self._initial_obs_shape['camera'] = (height, width, 3) # Assuming BGR, will convert to RGB later
            log.info(f"Camera video: {width}x{height} @ {fps} FPS, {self._max_frames} frames.")

        if self._birdeye_video_path:
            self._cap_birdeye = cv2.VideoCapture(self._birdeye_video_path)
            if not self._cap_birdeye.isOpened():
                raise IOError(f"Cannot open birdeye video file: {self._birdeye_video_path}")
            width = int(self._cap_birdeye.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self._cap_birdeye.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = self._cap_birdeye.get(cv2.CAP_PROP_FPS)
            self._max_frames = max(self._max_frames, int(self._cap_birdeye.get(cv2.CAP_PROP_FRAME_COUNT)))
            self._initial_obs_shape['birdeye_wpt'] = (height, width, 3) # Assuming BGR, will convert to RGB later
            log.info(f"Birdeye video: {width}x{height} @ {fps} FPS, {self._max_frames} frames.")

        if not self._max_frames:
            raise ValueError("No frames found in any video stream.")
        
        # Ensure target FPS is set if it exists, otherwise use video FPS
        if 'fps' in locals() and not self._target_fps:
            self._target_fps = fps


    def _get_observation_space(self) -> spaces.Dict:
        obs_spaces = {}
        for key, shape in self._initial_obs_shape.items():
            obs_spaces[key] = spaces.Box(low=0, high=255, shape=shape, dtype=np.uint8)
        # Add a dummy `is_first` observation, as it's expected by DreamerV3
        obs_spaces['is_first'] = spaces.Box(low=0, high=1, shape=(), dtype=np.bool_)
        # Add a dummy `is_terminal` observation, as it's expected by DreamerV3
        obs_spaces['is_terminal'] = spaces.Box(low=0, high=1, shape=(), dtype=np.bool_)
        # Add a dummy `is_last` observation, as it's expected by DreamerV3
        obs_spaces['is_last'] = spaces.Box(low=0, high=1, shape=(), dtype=np.bool_)
        
        # Additional observations expected by dreamerv3 preprocess
        obs_spaces['reward'] = spaces.Box(low=-np.inf, high=np.inf, shape=(), dtype=np.float32)
        
        return spaces.Dict(obs_spaces)

    def reset(self) -> Dict[str, np.ndarray]:
        log.info("VideoEnv reset.")
        if self._cap_camera:
            self._cap_camera.set(cv2.CAP_PROP_POS_FRAMES, 0)
        if self._cap_birdeye:
            self._cap_birdeye.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self._current_frame = 0
        return self._read_frames(is_first=True)

    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, Dict]:
        self._current_frame += 1
        is_last = self._current_frame >= self._max_frames
        obs = self._read_frames(is_first=False, is_terminal=is_last, is_last=is_last)
        reward = 0.0 # Dummy reward
        done = is_last
        info = {} # Empty info dict
        return obs, reward, done, info

    def _read_frames(self, is_first: bool = False, is_terminal: bool = False, is_last: bool = False) -> Dict[str, np.ndarray]:
        obs = {}
        ret_camera, frame_camera = (True, None)
        ret_birdeye, frame_birdeye = (True, None)

        if self._cap_camera:
            self._cap_camera.set(cv2.CAP_PROP_POS_FRAMES, self._current_frame)
            ret_camera, frame_camera = self._cap_camera.read()
            if ret_camera and frame_camera is not None:
                # Convert BGR to RGB (OpenCV reads BGR by default)
                obs['camera'] = cv2.cvtColor(frame_camera, cv2.COLOR_BGR2RGB)
            else:
                log.warning(f"Failed to read frame {self._current_frame} from camera video. Assuming end of stream.")
                is_terminal = True # End episode if video fails to read

        if self._cap_birdeye:
            self._cap_birdeye.set(cv2.CAP_PROP_POS_FRAMES, self._current_frame)
            ret_birdeye, frame_birdeye = self._cap_birdeye.read()
            if ret_birdeye and frame_birdeye is not None:
                # Convert BGR to RGB (OpenCV reads BGR by default)
                obs['birdeye_wpt'] = cv2.cvtColor(frame_birdeye, cv2.COLOR_BGR2RGB)
            else:
                log.warning(f"Failed to read frame {self._current_frame} from birdeye video. Assuming end of stream.")
                is_terminal = True # End episode if video fails to read
        
        # Add dummy required observations
        obs['is_first'] = np.array(is_first, dtype=np.bool_)
        obs['is_terminal'] = np.array(is_terminal, dtype=np.bool_)
        obs['is_last'] = np.array(is_last, dtype=np.bool_)
        obs['reward'] = np.array(0.0, dtype=np.float32)

        return obs

    def close(self):
        log.info("VideoEnv closed.")
        if self._cap_camera:
            self._cap_camera.release()
        if self._cap_birdeye:
            self._cap_birdeye.release()

    def render(self, mode='human'):
        # Rendering will be handled externally by the monitor
        pass

    def __del__(self):
        self.close()
