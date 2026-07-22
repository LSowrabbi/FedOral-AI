import os
import sys
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import OralCancerDataset, get_transforms
from models.resnet_baseline import OralCancerResNet
from utils.device import get_device
from torch.utils.data import DataLoader
import numpy as np
from collections import Counter

device = get_device()

# Load model
model = OralCancerResNet(num_classes=2, freeze_backbone=True).to(device)
model.load_state_dict(torch.load(
    'models/checkpoints/best_model.pth', map_location=device))
model.eval()

# Load test set
root_dirs = [
    'data/raw/orca-deduplicated',
    'data/raw/ashen-deduplicated',
]
test_ds = OralCancerDataset(root_dirs, 'test', get_transforms('test'))
test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

all_preds = []
all_labels = []
all_probs = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        preds = torch.argmax(outputs, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_probs.extend(probs.cpu().numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)
all_probs = np.array(all_probs)

print(f"\n{'='*50}")
print(f"DIAGNOSIS REPORT")
print(f"{'='*50}")
print(f"\nTrue label distribution:")
print(f"  NON CANCER (0): {(all_labels==0).sum()}")
print(f"  CANCER (1):     {(all_labels==1).sum()}")

print(f"\nPrediction distribution:")
print(f"  NON CANCER (0): {(all_preds==0).sum()}")
print(f"  CANCER (1):     {(all_preds==1).sum()}")

print(f"\nConfusion Matrix:")
tp = ((all_preds==1) & (all_labels==1)).sum()
tn = ((all_preds==0) & (all_labels==0)).sum()
fp = ((all_preds==1) & (all_labels==0)).sum()
fn = ((all_preds==0) & (all_labels==1)).sum()
print(f"  True Positive  (Cancer correctly detected):  {tp}")
print(f"  True Negative  (Healthy correctly cleared):  {tn}")
print(f"  False Positive (Healthy wrongly flagged):    {fp}")
print(f"  False Negative (Cancer missed!):             {fn}")

print(f"\nProbability distribution sample (first 10):")
for i in range(min(10, len(all_probs))):
    print(f"  Image {i+1}: NON CANCER={all_probs[i][0]:.3f} "
          f"CANCER={all_probs[i][1]:.3f} → "
          f"Predicted: {'CANCER' if all_preds[i]==1 else 'NON CANCER'} "
          f"| Actual: {'CANCER' if all_labels[i]==1 else 'NON CANCER'}")
print(f"{'='*50}")