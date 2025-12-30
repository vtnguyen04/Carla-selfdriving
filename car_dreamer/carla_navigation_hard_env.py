import carla
from .carla_wpt_env import CarlaWptEnv
from .toolkit import FixedPathPlanner
from car_dreamer.toolkit.utils import get_logger

log = get_logger(log_dir=".", job_name="carla_navigation_hard_env")


class CarlaNavigationHardEnv(CarlaWptEnv):
    """
    In this task, the ego vehicle needs to navigate through a specific long route
    that integrates right turns, left turns, and a roundabout in Town03.
    """

    def on_reset(self) -> None:
        # Spawn ego vehicle at the specific start point
        self.ego_src = self._config.lane_start_point
        ego_transform = carla.Transform(
            carla.Location(*self.ego_src[:3]), 
            carla.Rotation(yaw=self.ego_src[3])
        )
        self.ego = self._world.spawn_actor(transform=ego_transform)
        
        # Spawn background traffic
        self._world.spawn_auto_actors(self._config.num_vehicles)
        
        # Use FixedPathPlanner to follow the defined ego_path
        self.ego_path = self._config.ego_path
        self.use_road_waypoints = self._config.use_road_waypoints
        self.ego_planner = FixedPathPlanner(
            vehicle=self.ego,
            vehicle_path=self.ego_path,
            use_road_waypoints=self.use_road_waypoints,
        )
        
        self.waypoints, self.planner_stats = self.ego_planner.run_step()
        self.num_completed = self.planner_stats["num_completed"]
        self.sum_travel_distance = self.planner_stats["travel_distance"]
        log.info(f"[Task Hard] Path initialized with {self.ego_planner.get_waypoint_num()} waypoints.")

    def on_step(self) -> None:
        super().on_step()
        self.sum_travel_distance += self.planner_stats["travel_distance"]
        if self._time_step % 100 == 0:
            log.info(f"[Task Hard] Remaining waypoints: {self.ego_planner.get_waypoint_num()}")

    def is_destination_reached(self):
        # In hard mode, we consider it a success if the vehicle reaches 
        # the end of the fixed path.
        reached = self.ego_planner.get_waypoint_num() == 0
        if reached:
            log.info("[Task Hard] SUCCESS: All waypoints reached!")
        return reached