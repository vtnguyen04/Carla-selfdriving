import os
import sys
import pathlib
import time

# Thêm project root vào path để import
root = pathlib.Path(__file__).parent.parent
sys.path.append(str(root))

import carla
import gym
import numpy as np
import car_dreamer
from car_dreamer.toolkit.planner.agents.navigation.basic_agent import BasicAgent

def record_full_route():
    print("🚀 Đang khởi tạo lộ trình 'carla_navigation_hard'...")
    
    # 1. Cấu hình task 
    # QUAN TRỌNG: Phải có --env.display.enable True thì môi trường mới render để save video
    overrides = [
        '--env.display.enable', 'True',
        '--env.display.save_video', 'True',
        '--env.display.use_local_window', 'False',
        '--env.world.carla_port', '2000',
        '--env.world.town', 'Town03',
        '--env.action.discrete', 'False'
    ]
    
    try:
        env, config = car_dreamer.create_task('carla_navigation_hard', overrides)
        obs = env.reset()
        ego_vehicle = env.ego
        
        # Warm-up để kích hoạt hệ thống render và lấy dữ liệu
        env.step(np.array([0.0, 0.0], dtype=np.float32))
        
        # 2. Expert Agent (PID)
        agent = BasicAgent(ego_vehicle, target_speed=40) 
        
        # Lấy điểm cuối cùng làm đích đến
        final_destination = carla.Location(
            x=config.env.ego_path[-1][0], 
            y=config.env.ego_path[-1][1], 
            z=config.env.ego_path[-1][2]
        )
        agent.set_destination(final_destination)

        print(f"🎬 Bắt đầu ghi hình. Đang lái đến: {final_destination}")
        print("💡 Lưu ý: Video chỉ thực sự được lưu khi Episode kết thúc hoặc bạn nhấn Ctrl+C.")

        done = False
        step = 0
        while not done:
            control = agent.run_step()
            
            # Ánh xạ chuẩn sang hệ thống gia tốc của Env
            if control.brake > 0:
                acc = -control.brake * 3.0
            else:
                # PID throttle thường nhỏ, nhân 3 để xe chạy bốc hơn
                acc = control.throttle * 3.0
            
            action = np.array([acc, -control.steer], dtype=np.float32)
            
            # Gửi lệnh và Render
            obs, reward, done, info = env.step(action)
            
            step += 1
            if step % 50 == 0:
                v = ego_vehicle.get_velocity()
                speed = 3.6 * np.sqrt(v.x**2 + v.y**2 + v.z**2)
                print(f"📍 Step {step}: Tốc độ {speed:.1f} km/h | Waypoints: {env.ego_planner.get_waypoint_num()}")

            if env.is_destination_reached():
                print("🏁 Đã hoàn thành lộ trình!")
                break

    except KeyboardInterrupt:
        print("\n🛑 Đã dừng ghi hình thủ công.")
    except Exception as e:
        print(f"❌ Lỗi: {e}")
    finally:
        if 'env' in locals():
            env.close()
        print(f"✅ Đã đóng môi trường. Kiểm tra video trong thư mục 'evaluation_videos/carla_navigation_hard/'")

if __name__ == "__main__":
    record_full_route()