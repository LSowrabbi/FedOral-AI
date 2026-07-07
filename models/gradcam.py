import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import random as rnd
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import OralCancerDataset, get_transforms
from models.efficientnet_baseline import OralCancerEfficientNet
from utils.device import get_device

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


def denormalize(tensor):
    """Convert normalized tensor back to displayable image."""
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img  = tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    img  = std * img + mean
    return np.clip(img, 0, 1).astype(np.float32)


def generate_gradcam_figure(
        model, dataset, device,
        num_samples=20, class_name="CANCER",
        save_path="results/gradcam/"):

    os.makedirs(save_path, exist_ok=True)
    label_idx  = 1 if class_name == "CANCER" else 0
    class_labels = ["NON CANCER", "CANCER"]
    colors       = {"CANCER": "#D85A30", "NON CANCER": "#0F6E56"}

    # Target layer — last conv block of EfficientNet-B0
    target_layers = [model.model.features[-1]]

    # Move model to CPU for Grad-CAM
    model_cpu = model.cpu()
    model_cpu.eval()

    # Enable gradients for Grad-CAM
    for param in model_cpu.parameters():
        param.requires_grad = True

    # Collect samples spread across the test set
    all_class_samples = [
        (img, lbl) for img, lbl in dataset
        if lbl == label_idx
    ]

    # Pick samples spread across the dataset for diversity
    # Select from beginning, middle and end of test set
    total = len(all_class_samples)
    if total >= num_samples:
        # Pick evenly spaced samples for diversity
        indices = [int(i * total / num_samples)
                for i in range(num_samples)]
        samples = [all_class_samples[i] for i in indices]
    else:
        samples = all_class_samples

    if not samples:
        print(f"No {class_name} images found in dataset")
        return None

    fig, axes = plt.subplots(
        len(samples), 3,
        figsize=(12, 4 * len(samples))
    )
    if len(samples) == 1:
        axes = [axes]

    with GradCAM(model=model_cpu, target_layers=target_layers) as cam:
        for i, (img_tensor, label) in enumerate(samples):
            # Prepare input — CPU tensor
            img_input = img_tensor.unsqueeze(0).cpu()

            # Generate Grad-CAM
            targets   = [ClassifierOutputTarget(label_idx)]
            grayscale = cam(input_tensor=img_input, targets=targets)
            cam_map   = grayscale[0]

            # Get prediction
            with torch.no_grad():
                output     = model_cpu(img_input)
                probs      = output.softmax(dim=1)[0].numpy()
                pred_idx   = int(probs.argmax())
                pred_label = class_labels[pred_idx]
                true_label = class_labels[label]

            # Log prediction result for each image
            print(f"Image {i+1}: True={true_label} Pred={pred_label} "
            f"Conf={probs[pred_idx]*100:.1f}% "
            f"{'✓ CORRECT' if pred_idx==label else '✗ FALSE NEG'}")
    
            # Original image
            original = denormalize(img_tensor)

            # Overlay
            overlaid = show_cam_on_image(
                original, cam_map, use_rgb=True
            )

            # Plot
            # Column 1: Original
            axes[i][0].imshow(original)
            axes[i][0].set_title(
                f"Original\nTrue: {true_label}",
                fontsize=10, color=colors[true_label]
            )
            axes[i][0].axis("off")

            # Column 2: Heatmap only
            axes[i][1].imshow(original)
            axes[i][1].imshow(
                cm.jet(cam_map), alpha=0.5
            )
            axes[i][1].set_title(
                "Grad-CAM Heatmap\n(red = high activation)",
                fontsize=10
            )
            axes[i][1].axis("off")

            # Column 3: Overlay + prediction
            correct = "✓" if pred_idx == label else "✗"
            axes[i][2].imshow(overlaid)
            axes[i][2].set_title(
                f"Overlay {correct}\n"
                f"Pred: {pred_label} ({probs[pred_idx]*100:.1f}%)",
                fontsize=10, color=colors[pred_label]
            )
            axes[i][2].axis("off")

    plt.suptitle(
        f"Grad-CAM Explainability — {class_name} Images\n"
        f"EfficientNet-B0 · FedOral-AI",
        fontsize=13, fontweight="bold", y=1.01
    )
    plt.tight_layout()

    fname = os.path.join(
        save_path, f"gradcam_{class_name.lower().replace(' ', '_')}.png"
    )
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fname}")

    # Move model back to original device
    model.to(device)
    return fname


if __name__ == "__main__":
    device = get_device()

    # Load model
    model = OralCancerEfficientNet(
        num_classes=2, freeze_backbone=True
    ).to(device)
    model.load_state_dict(torch.load(
        "models/checkpoints/best_model.pth",
        map_location=device
    ))
    model.eval()

    # Load test dataset
    root_dirs = [
        "data/raw/oral-cancer-dataset/Oral Cancer/Oral Cancer Dataset",
        "data/raw/oral-cancer-dataset/Oral cancer Dataset 2.0/"
        "OC Dataset kaggle new"
    ]
    test_ds = OralCancerDataset(
        root_dirs, "test", get_transforms("test")
    )

    print("\n── Generating Grad-CAM for CANCER images ──")
    generate_gradcam_figure(
        model, test_ds, device,
        num_samples=20, class_name="CANCER",
        save_path="results/gradcam/"
    )

    print("\n── Generating Grad-CAM for NON CANCER images ──")
    generate_gradcam_figure(
        model, test_ds, device,
        num_samples=20, class_name="NON CANCER",
        save_path="results/gradcam/"
    )

    print("\nGrad-CAM complete — check results/gradcam/")

