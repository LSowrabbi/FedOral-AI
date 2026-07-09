import time
import torch
import numpy as np
import sys
sys.path.insert(0, '.')

from models.resnet_baseline import OralCancerResNet
from models.efficientnet_baseline import OralCancerEfficientNet
from models.mobilenet_baseline import OralCancerMobileNet
from utils.device import get_device


def measure_inference_time(model, device, n_runs=100, n_trials=5):
    model.eval()
    dummy = torch.randn(1, 3, 224, 224).to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(20):
            model(dummy)

    # Run multiple trials for stability
    trial_times = []
    for trial in range(n_trials):
        start = time.time()
        with torch.no_grad():
            for _ in range(n_runs):
                model(dummy)
        elapsed = time.time() - start
        trial_times.append((elapsed / n_runs) * 1000)

    avg_ms = np.mean(trial_times)
    std_ms = np.std(trial_times)
    throughput = 1000 / avg_ms

    return avg_ms, std_ms, throughput


if __name__ == '__main__':
    device = get_device()

    models_to_test = {
        'ResNet-18': OralCancerResNet(),
        'EfficientNet-B0': OralCancerEfficientNet(),
        'MobileNetV2': OralCancerMobileNet(),
    }

    print(f"\n{'='*65}")
    print(f"{'Model':<20}{'Params':<15}{'Latency (ms)':<20}{'Throughput (img/s)'}")
    print(f"{'-'*65}")

    for name, model in models_to_test.items():
        model = model.to(device)
        total_params = sum(p.numel() for p in model.parameters())
        avg_ms, std_ms, throughput = measure_inference_time(model, device)
        result_str = f"{avg_ms:.2f}±{std_ms:.2f}"
        print(f"{name:<20}{total_params:<15,}{result_str:<20}{throughput:.1f}")

    print(f"{'='*65}")