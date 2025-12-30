import atexit
import base64
import datetime
import json
import os
import queue
import threading

import cv2
import numpy as np
from flask import Flask, Response, render_template
from car_dreamer.toolkit.utils import get_logger

log = get_logger(log_dir=".", job_name="monitor")


def _generate_frame(obs, info, config):
    # --- Find main camera image and map image ---
    main_img = None
    map_img = None
    render_keys = config.display.render_keys
    
    # Prioritize 'camera_display' for the main view
    display_cam_key = next((key for key in render_keys if key == 'camera_display'), None)
    # Fallback to 'camera' if 'camera_display' is not available or not in render_keys
    model_cam_key = next((key for key in render_keys if key == 'camera'), None)
    birdeye_key = next((key for key in render_keys if 'birdeye' in key), None)

    if birdeye_key and birdeye_key in obs:
        map_img = obs[birdeye_key].copy()

    if display_cam_key and display_cam_key in obs:
        main_img = obs[display_cam_key].copy()
    elif model_cam_key and model_cam_key in obs: # Fallback to model's camera
        main_img = obs[model_cam_key].copy()
    # If no main image, fallback to first available image, otherwise do nothing
    if main_img is None:
        if render_keys and render_keys[0] in obs:
             main_img = obs[render_keys[0]].copy()
        else:
            return None

    # Convert to BGR for OpenCV (assuming input is RGB)
    if len(main_img.shape) == 3 and main_img.shape[2] == 3:
        main_img = cv2.cvtColor(main_img, cv2.COLOR_RGB2BGR)
    
    H, W, _ = main_img.shape

    # --- Overlay map image if it exists ---
    if map_img is not None:
        map_h_new = int(H * 0.3)  # 30% of main image height
        map_aspect_ratio = map_img.shape[1] / map_img.shape[0]
        map_w_new = int(map_h_new * map_aspect_ratio)
        
        map_resized = cv2.resize(map_img, (map_w_new, map_h_new))

        # Convert map to BGR for OpenCV
        if len(map_resized.shape) == 3 and map_resized.shape[2] == 3:
             map_resized = cv2.cvtColor(map_resized, cv2.COLOR_RGB2BGR)

        # Position on top-right with a margin
        margin = 10
        x_offset = W - map_w_new - margin
        y_offset = margin
        
        # Create ROI and blend
        roi = main_img[y_offset:y_offset+map_h_new, x_offset:x_offset+map_w_new]
        alpha = 0.8
        cv2.addWeighted(map_resized, alpha, roi, 1 - alpha, 0, roi)
        main_img[y_offset:y_offset+map_h_new, x_offset:x_offset+map_w_new] = roi

    # --- Create Dashboard Area ---
    dash_height = 80
    dashboard = np.zeros((dash_height, W, 3), dtype=np.uint8)

    # Get Data from Info Dict
    speed_kmh = info.get("speed_norm", 0.0) * 3.6
    throttle = info.get("throttle", 0.0)
    steer = info.get("steer", 0.0)
    brake = info.get("brake", 0.0)

    # Draw Dashboard Elements
    cv2.putText(dashboard, f"Speed: {speed_kmh:.1f} km/h", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    bar_width = 180
    bar_height = 15
    
    # Throttle Bar
    cv2.putText(dashboard, "Throttle", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.rectangle(dashboard, (100, 55), (100 + bar_width, 55 + bar_height), (50, 50, 50), -1)
    throttle_len = int(bar_width * throttle)
    cv2.rectangle(dashboard, (100, 55), (100 + throttle_len, 55 + bar_height), (0, 255, 0), -1)

    # Brake Bar
    cv2.putText(dashboard, "Brake", (320, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.rectangle(dashboard, (380, 55), (380 + bar_width, 55 + bar_height), (50, 50, 50), -1)
    brake_len = int(bar_width * brake)
    cv2.rectangle(dashboard, (380, 55), (380 + brake_len, 55 + bar_height), (255, 0, 0), -1)
    
    # Steer Indicator
    steer_center = W - bar_width // 2 - 20
    steer_pos = int(steer * (bar_width / 2))
    cv2.line(dashboard, (steer_center, 55), (steer_center, 55 + bar_height), (100, 100, 100), 2)
    cv2.line(dashboard, (steer_center, 62), (steer_center + steer_pos, 62), (0, 200, 255), 4)

    # --- Combine Image and Dashboard ---
    return np.vstack([main_img, dashboard])


class EnvMonitorBase:
    def __init__(self, config):
        self._config = config
        self._obs_queue = queue.Queue()
        self._info_queue = queue.Queue()
        self._thread = threading.Thread(target=self._run_server)
        self._thread.start()
        atexit.register(self.stop)

    def _run_server(self):
        app = Flask(__name__, template_folder="templates")

        @app.route("/")
        def index():
            return render_template("index.html")

        @app.route("/stream")
        def stream():
            def generate():
                while True:
                    obs = self._obs_queue.get()
                    info = self._info_queue.get()
                    frame = self._render(obs, info)
                    yield f"data: {json.dumps(frame)}\n\n"
                    self._obs_queue.task_done()
                    self._info_queue.task_done()

            return Response(generate(), mimetype="text/event-stream")

        app.run(
            host="0.0.0.0",
            port=self._config.world.carla_port + 7000,
            use_reloader=False,
        )

    def _render_info(self, info):
        rendered_info = {}
        for key, value in info.items():
            if isinstance(value, (float, int, bool)):
                rendered_info[key] = value
            elif isinstance(value, np.number):
                rendered_info[key] = value.item()
            elif isinstance(value, np.ndarray) and value.ndim == 1:
                rendered_info[key] = value.tolist()
            else:
                rendered_info[key] = str(value)
        return rendered_info

    def _render_images(self, obs):
        images = []
        display_config = self._config.display
        if display_config.enable and display_config.render_keys:
            for key in display_config.render_keys:
                if key in obs:
                    img = obs[key]
                    if len(img.shape) == 2:
                        img = np.repeat(img[:, :, np.newaxis], 3, axis=2)
                    else:
                        img = img[:, :, ::-1]
                    _, img_encoded = cv2.imencode(".webp", img)
                    img_base64 = base64.b64encode(img_encoded).decode("utf-8")
                    images.append({"key": key, "image": img_base64})
        return images

    def _render(self, obs, info):
        return {"images": self._render_images(obs), "info": self._render_info(info)}

    def stop(self):
        if self._thread and self._thread.is_alive():
            self._thread.join()

    def __del__(self):
        self.stop()


class EnvMonitorOpenCV(EnvMonitorBase):
    def render(self, obs, info):
        if not self._obs_queue.full():
            self._obs_queue.put(obs)
        if not self._info_queue.full():
            self._info_queue.put(info)

class EnvMonitorLocalCV:
    """
    A monitor that displays the environment in a local OpenCV window.
    It shows a picture-in-picture view with camera as the main feed and
    a birdeye map in the top-right corner, plus a dashboard.
    """

    def __init__(self, config):
        self._config = config
        self._window_name = "CARLA Agent Evaluation"
        cv2.namedWindow(self._window_name, cv2.WINDOW_AUTOSIZE)
        atexit.register(self.close)

    def render(self, obs, info):
        final_image = _generate_frame(obs, info, self._config)
        if final_image is not None:
            cv2.imshow(self._window_name, final_image)
            cv2.waitKey(1)

    def close(self):
        cv2.destroyAllWindows()


class EnvMonitorVideo:
    """
    A monitor that saves each evaluation episode to a separate video file.
    """

    def __init__(self, config):
        self._config = config
        self._video_writer = None
        self._task_name = config.get("task_name", "unknown_task")
        self._save_dir = os.path.join("evaluation_videos", self._task_name)
        os.makedirs(self._save_dir, exist_ok=True)
        self._episode_count = 0
        atexit.register(self.close)

    def _init_writer(self, frame_shape):
        self.close() # Close existing if any
        H, W, _ = frame_shape
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = os.path.join(self._save_dir, f"episode_{self._episode_count:03d}_{timestamp}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self._video_writer = cv2.VideoWriter(filename, fourcc, 20.0, (W, H))
        log.info(f"Started recording new episode to: {filename}")
        self._episode_count += 1

    def render(self, obs, info):
        # Detect start of a new episode
        # info contains terminal conditions and potentially 'is_first' from driver
        is_first = info.get("is_first", False)
        # Some envs wrap info differently, check if it's a new start
        if is_first and self._video_writer is not None:
            self.close()

        frame = _generate_frame(obs, info, self._config)
        if frame is not None:
            if self._video_writer is None:
                self._init_writer(frame.shape)
            self._video_writer.write(frame)
        
        # Detect end of episode to release early
        if info.get("is_last", False) or info.get("terminal", False):
            self.close()

    def close(self):
        if self._video_writer is not None:
            self._video_writer.release()
            self._video_writer = None