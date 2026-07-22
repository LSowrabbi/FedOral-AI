import os
import sys
sys.path.insert(0, '.')

import torch
from models.efficientnet_baseline import OralCancerEfficientNet
from data.dataset import OralCancerDataset, get_transforms
from utils.device import get_device

device = get_device()

model = OralCancerEfficientNet(num_classes=2, freeze_backbone=True).to(device)
model.load_state_dict(torch.load(
    'models/checkpoints/best_model.pth', map_location=device))
model.eval()

root_dirs = [
    'data/raw/orca-deduplicated',
    'data/raw/ashen-deduplicated',
]
test_ds = OralCancerDataset(root_dirs, 'test', get_transforms('test'))

# Separate samples by modality
clinical_samples = [(p, l) for p, l in test_ds.samples if 'orca-deduplicated' in p]
histo_samples = [(p, l) for p, l in test_ds.samples if 'ashen-deduplicated' in p]

print(f"Clinical samples in test set: {len(clinical_samples)}")
print(f"Histopathology samples in test set: {len(histo_samples)}")

clinical_cancer = [s for s in clinical_samples if s[1] == 1][2:3]
clinical_normal = [s for s in clinical_samples if s[1] == 0][2:3]
histo_cancer = [s for s in histo_samples if s[1] == 1][2:3]
histo_normal = [s for s in histo_samples if s[1] == 0][2:3]

print(f"Selected: {len(clinical_cancer)} clinical-cancer, "
      f"{len(clinical_normal)} clinical-normal, "
      f"{len(histo_cancer)} histo-cancer, "
      f"{len(histo_normal)} histo-normal")

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

def denormalize(tensor):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    img = std * img + mean
    return np.clip(img, 0, 1).astype(np.float32)

# Combine all 6 selected samples in desired order
selected = (clinical_cancer + clinical_normal +
            histo_cancer + histo_normal)
labels_text = (['Clinical\nCANCER'] * len(clinical_cancer) +
               ['Clinical\nNON CANCER'] * len(clinical_normal) +
               ['Histo\nCANCER'] * len(histo_cancer) +
               ['Histo\nNON CANCER'] * len(histo_normal))

transform = get_transforms('test')
model_cpu = model.cpu()
model_cpu.eval()

# Enable gradients for Grad-CAM (required even in eval mode)
for param in model_cpu.parameters():
    param.requires_grad = True

target_layers = [model_cpu.model.features[-2]]

class_labels = ["NON CANCER", "CANCER"]
colors = {"CANCER": "#D85A30", "NON CANCER": "#0F6E56"}

fig, axes = plt.subplots(len(selected), 3, figsize=(12, 4 * len(selected)))

with GradCAM(model=model_cpu, target_layers=target_layers) as cam:
    for i, ((path, label), modality_label) in enumerate(zip(selected, labels_text)):
        img = Image.open(path).convert('RGB')
        img_tensor = transform(img)
        img_input = img_tensor.unsqueeze(0).cpu()

        targets = [ClassifierOutputTarget(label)]
        grayscale = cam(input_tensor=img_input, targets=targets)
        cam_map = grayscale[0]

        with torch.no_grad():
            output = model_cpu(img_input)
            probs = output.softmax(dim=1)[0].numpy()
            pred_idx = int(probs.argmax())
            pred_label = class_labels[pred_idx]
            true_label = class_labels[label]

        original = denormalize(img_tensor)
        overlaid = show_cam_on_image(original, cam_map, use_rgb=True)

        axes[i][0].imshow(original)
        axes[i][0].set_title(f"{modality_label}\nTrue: {true_label}",
                              fontsize=10, color=colors[true_label])
        axes[i][0].axis("off")

        axes[i][1].imshow(original)
        axes[i][1].imshow(cm.jet(cam_map), alpha=0.5)
        axes[i][1].set_title("Grad-CAM Heatmap", fontsize=10)
        axes[i][1].axis("off")

        correct = "Correct" if pred_idx == label else "Incorrect"
        axes[i][2].imshow(overlaid)
        axes[i][2].set_title(
            f"{correct}\nPred: {pred_label} ({probs[pred_idx]*100:.1f}%)",
            fontsize=10, color=colors[pred_label])
        axes[i][2].axis("off")

        print(f"{modality_label.replace(chr(10),' ')}: True={true_label} "
            f"Pred={pred_label} Conf={probs[pred_idx]*100:.1f}% "
            f"{correct}")

plt.suptitle("Grad-CAM Explainability — Clinical and Histopathology Modalities\n"
             "EfficientNet-B0 (Clean Dataset)",
             fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()

os.makedirs("results/gradcam_paper", exist_ok=True)
save_path = "results/gradcam_paper/gradcam_figure1.png"
plt.savefig(save_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSaved: {save_path}")