import torch
import time


def get_device():
    """
    Detects the best available device for training.
    Priority: Apple MPS > NVIDIA CUDA > CPU
    """
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon GPU (MPS)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using NVIDIA GPU (CUDA)")
    else:
        device = torch.device("cpu")
        print("Using CPU — training will be slower")

    return device


if __name__ == '__main__':
    device = get_device()

    # Quick benchmark — matrix multiplication on device
    size = 1000
    a = torch.randn(size, size).to(device)
    b = torch.randn(size, size).to(device)

    start = time.time()
    for _ in range(100):
        c = torch.mm(a, b)
    elapsed = time.time() - start

    print(f"Matrix multiply benchmark: {elapsed:.3f}s for 100 iterations")
    print(f"Device: {device}")