import torch
import time

def get_device():
    """Detects and returns the best available device for training."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using NVIDIA GPU (CUDA)")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon GPU (MPS)")
    else:
        device = torch.device("cpu")
        print("Using CPU - training might be slow. Consider using a GPU for better performance.")

    return device


if __name__ == '__main__':
    device = get_device()

    # Matrix multiplication on device with timing to benchmark performance
    size = 1000
    a = torch.randn(size, size).to(device)
    b = torch.randn(size, size).to(device)

    start = time.time()
    for _ in range(100):
        c = torch.mm(a, b)
    elapsed = time.time() - start

    print(f"Matrix multiply benchmark: {elapsed:.3f}s for 100 iterations")
    print(f"Throughput: {100/elapsed:.1f} iterations/second")
    print(f"Device: {device}")