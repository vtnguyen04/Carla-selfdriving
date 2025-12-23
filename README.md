# Cài đặt mã nguồn
```bash
git clone https://github.com/vtnguyen04/Carla-selfdriving.git
cd Carla-selfdriving
```

```bash
export CARLA_ROOT="</path/to/carla>"
export PYTHONPATH="${CARLA_ROOT}/PythonAPI/carla":${PYTHONPATH}
```

Cài đặt gói bằng flit. Cờ `--symlink` được dùng để tạo liên kết tượng trưng (symlink) tới gói trong môi trường Python, nên các thay đổi trong gói sẽ có hiệu lực ngay mà không cần cài lại. (`--pth-file` cũng hoạt động như một lựa chọn thay thế cho `--symlink`.)

```bash
conda create python=3.10 --name self_driving
conda activate self_driving
pip install flit
flit install --symlink
```

```bash
cd RL
conda install -c "nvidia/label/cuda-12.8.0" cuda-toolkit
pip install -r requirements.txt
pip install "jax[cuda12_pip]==0.4.34" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

Thiết lập CARLA và biến môi trường:

```bash
export CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))
export CUSOLVER_PATH=$(dirname $(python -c "import nvidia.cusolver;print(nvidia.cusolver.__file__)"))
export LD_LIBRARY_PATH=$CUDNN_PATH/lib:$CUSOLVER_PATH/lib:$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
```

## Đào tạo

Chạy script đào tạo với các cấu hình mong muốn:
```bash
cd ..
chmode +x ./train_dm3

# Ví dụ 1: Dùng cài đặt mặc định để đào tạo tác nhân
./train_dm3.sh 2000 0 --task carla_four_lane --dreamerv3.logdir ./logdir/carla_four_lane
# Ví dụ 2: Ghi đè task và tham số mô hình
./train_dm3.sh 2000 0 --task carla_right_turn_simple \
    --dreamerv3.logdir ./logdir/carla_right_turn_simple \
    --dreamerv3.run.steps=5e6

./train_dm3.sh 2000 0 --task carla_navigation --dreamerv3.logdir ./logdir/carla_navigation1 --model_size xsmall --dreamerv3.world_model reconstruction

```

`2000` là số cổng (port) của server CARLA. Script sẽ tự động khởi động server nên bạn không cần khởi động thủ công.
`0` là số GPU.
`--task` là tên tác vụ và `--dreamerv3.logdir` là thư mục để lưu nhật ký (logs) đào tạo. Để xem danh sách đầy đủ các task và cấu hình của chúng, xem tài liệu tại [documentation](https://car-dreamer.readthedocs.io/en/latest/tasks.html).

## Trực quan hóa

Theo dõi dữ liệu trực tuyến có thể truy cập trên trang web tại `http://localhost:9000/`, cổng này nên được thay đổi thành `<carla-port> + 7000` nếu bạn không dùng cổng mặc định `2000` của CARLA server.

Ghi log dữ liệu ngoại tuyến có thể truy cập qua TensorBoard.

```bash
tensorboard --logdir ./logdir/carla_four_lane
```

Mở `http://localhost:6006/` trong trình duyệt để xem kết quả.

## Đánh giá

Chạy các lệnh sau để đánh giá mô hình đã huấn luyện, trong đó đối số thứ ba là đường dẫn tới checkpoint:

```bash
bash eval_dm3.sh 2000 0 ./logdir/carla_four_lane/checkpoint.ckpt --task carla_four_lane --dreamerv3.logdir ./logdir/eval_carla_four_lane
```
