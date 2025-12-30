# Cài đặt mã nguồn
```bash
git clone https://github.com/vtnguyen04/Carla-selfdriving.git
cd Carla-selfdriving

wget https://tiny.carla.org/carla-0-9-15-linux
mkdir carla_simulate
tar -xvzf carla-0-9-15-linux -C carla_simulate

export CARLA_ROOT=/workspace/Carla-selfdriving/carla_simulate
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
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
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
chmod +x ./train_dm3.sh

# Ví dụ 1: Dùng cài đặt mặc định để đào tạo tác nhân
./train_dm3.sh 2000 0 --task carla_four_lane --dreamerv3.logdir ./logdir/carla_four_lane
# tùy chọn kích thước mô hình và task mong muốn
./train_dm3.sh 2000 0 --task carla_navigation --dreamerv3.logdir ./logdir/carla_navigation --model_size xsmall
hoặc
./train_dm3.sh 2000 0 --task carla_right_turn_simple --dreamerv3.logdir ./logdir/carla_right_turn_simple --model_size small


./train_dm3.sh 2000 0 --task carla_right_turn_simple \
    --dreamerv3.logdir ./logdir/carla_right_turn_simple \
    --dreamerv3.run.steps=5e6

```

## Trực quan hóa

Theo dõi dữ liệu trực tuyến có thể truy cập trên trang web tại `http://localhost:9000/`

Ghi log dữ liệu ngoại tuyến có thể truy cập qua TensorBoard.

```bash
tensorboard --logdir ./logdir/tên-folder-log
```

Mở `http://localhost:6006/` trong trình duyệt để xem kết quả.

## Đánh giá

Chạy các lệnh sau để đánh giá mô hình đã huấn luyện, trong đó đối số thứ ba là đường dẫn tới checkpoint:

```bash
bash eval_dm3.sh 2000 0 ./logdir/carla_four_lane/checkpoint.ckpt --task carla_four_lane --dreamerv3.logdir ./logdir/carla_navigation
```
