
import jax
import sys
import subprocess

def check_cuda():
    print("\n--- JAX CUDA Check ---")
    try:
        devices = jax.devices()
        print(f"JAX devices: {devices}")
    except Exception as e:
        print(f"Error during JAX CUDA check: {e}")

    print("\n--- System CUDA Check ---")
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print("nvidia-smi output:")
            print(result.stdout)
        else:
            print("nvidia-smi failed to run. Is the NVIDIA driver installed correctly?")
            print(f"Stderr: {result.stderr}")
    except FileNotFoundError:
        print("nvidia-smi not found. Is the NVIDIA driver installed and in your PATH?")
    except Exception as e:
        print(f"An error occurred while running nvidia-smi: {e}")

if __name__ == "__main__":
    check_cuda()
