import carla
import time
import argparse
from car_dreamer.toolkit.utils import get_logger

log = get_logger(log_dir=".", job_name="get_coords")

def main(port):
    client = carla.Client('localhost', port)
    client.set_timeout(10.0)
    world = client.get_world()
    spectator = world.get_spectator()

    log.info("---------------------------------------------------------------------------")
    log.info("Get Coordinates Script is running.")
    log.info(f"Connected to CARLA on port {port}.")
    log.info("Switch to the CARLA window and fly the spectator camera around.")
    log.info("The coordinates will be printed here every 2 seconds.")
    log.info("Use these coordinates for your 'ego_path' in tasks.yaml.")
    log.info("Press Ctrl+C to exit.")
    log.info("---------------------------------------------------------------------------")

    try:
        while True:
            transform = spectator.get_transform()
            location = transform.location
            rotation = transform.rotation
            
            # Formatting for easy copy-pasting into YAML
            yaml_format = f"- [{location.x:.1f}, {location.y:.1f}, {location.z:.1f}]"
            
            log.info(f"Current Location (X, Y, Z): ({location.x:.1f}, {location.y:.1f}, {location.z:.1f}) | Yaw: {rotation.yaw:.1f} | YAML format: {yaml_format}")
            
            time.sleep(2)
    except KeyboardInterrupt:
        log.info("\nExiting script.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get spectator coordinates from CARLA.')
    parser.add_argument('--port', type=int, default=2000, help='Port to connect to CARLA.')
    args = parser.parse_args()
    
    main(args.port)
