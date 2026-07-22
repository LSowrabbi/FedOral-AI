import os
import sys
import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score
)

sys.path.insert(0, '.')
from data.dataset import OralCancerDataset, get_transforms
from models.efficientnet_baseline import OralCancerEfficientNet
from utils.device import get_device
from torch.utils.data import DataLoader


def evaluate_subset(model, samples, transform, device, name=""):
    """Evaluate model on a specific subset of samples."""
    if len(samples) == 0:
        print(f"No samples found for {name}")
        return None

    from PIL import Image

    all_labels, all_preds, all_probs = [], [], []

    model.eval()
    with torch.no_grad():
        for img_path, label in samples:
            image = Image.open(img_path).convert('RGB')
            image = transform(image).unsqueeze(0).to(device)

            output = model(image)
            probs = torch.softmax(output, dim=1)[0]
            pred = torch.argmax(output, dim=1).item()

            all_labels.append(label)
            all_preds.append(pred)
            all_probs.append(probs[1].item())

    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)

    try:
        auc = roc_auc_score(all_labels, all_probs)
    except Exception:
        auc = 0.0

    print(f"\n{'='*55}")
    print(f"MODALITY: {name}")
    print(f"{'='*55}")
    print(f"Samples   : {len(samples)}")
    print(f"Accuracy  : {acc:.4f} ({acc*100:.1f}%)")
    print(f"Precision : {prec:.4f}")
    print(f"Recall    : {rec:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print(f"AUC-ROC   : {auc:.4f}")
    print(f"{'='*55}")

    return {
        'name': name, 'n': len(samples),
        'accuracy': acc, 'precision': prec,
        'recall': rec, 'f1': f1, 'auc': auc
    }


if __name__ == '__main__':
    device = get_device()

    # Load model
    model = OralCancerEfficientNet(
        num_classes=2, freeze_backbone=True
    ).to(device)
    model.load_state_dict(torch.load(
        'models/checkpoints/best_model.pth',
        map_location=device
    ))
    model.eval()

    # Load full test dataset
    root_dirs = [
        'data/raw/orca-deduplicated',
        'data/raw/ashen-deduplicated',
    ]
    test_ds = OralCancerDataset(
        root_dirs, 'test', get_transforms('test')
    )
    transform = get_transforms('test')

    # Split test samples by modality based on source folder
    clinical_samples = [
        (path, label) for path, label in test_ds.samples
        if 'orca-deduplicated' in path
    ]
    histo_samples = [
        (path, label) for path, label in test_ds.samples
        if 'ashen-deduplicated' in path
    ]

    print(f"\nTotal test samples: {len(test_ds.samples)}")
    print(f"Clinical photographs: {len(clinical_samples)}")
    print(f"Histopathology slides: {len(histo_samples)}")

    # Evaluate each modality separately
    clinical_results = evaluate_subset(
        model, clinical_samples, transform, device,
        name="Clinical Photographs (ORCA)"
    )

    histo_results = evaluate_subset(
        model, histo_samples, transform, device,
        name="Histopathology Slides (ashenafifasilkebede)"
    )

# Summary comparison
    print(f"\n{'='*55}")
    print(f"MODALITY COMPARISON SUMMARY")
    print(f"{'='*55}")
    print(f"{'Metric':<12} {'Clinical':<12} {'Histopath':<12} {'Diff':<10}")
    print(f"{'-'*55}")
    for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc']:
        c = clinical_results[metric]
        h = histo_results[metric]
        diff = c - h
        print(f"{metric:<12} {c:<12.4f} {h:<12.4f} {diff:+.4f}")
    print(f"{'='*55}")

    # Save results to file
    import csv
    os.makedirs('results/modality_analysis', exist_ok=True)

    with open('results/modality_analysis/modality_comparison.csv',
              'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['metric', 'clinical_photos',
                         'histopathology', 'difference'])
        for metric in ['accuracy', 'precision', 'recall',
                       'f1', 'auc']:
            c = clinical_results[metric]
            h = histo_results[metric]
            writer.writerow([metric, f'{c:.4f}',
                            f'{h:.4f}', f'{c-h:+.4f}'])

    with open('results/modality_analysis/modality_summary.txt',
              'w') as f:
        f.write("MODALITY ANALYSIS — FedOral-AI\n")
        f.write("="*55 + "\n\n")
        f.write(f"Total test samples: {len(test_ds.samples)}\n")
        f.write(f"Clinical photographs: {len(clinical_samples)}\n")
        f.write(f"Histopathology slides: {len(histo_samples)}\n\n")

        f.write("CLINICAL PHOTOGRAPHS (ORCA)\n")
        f.write("-"*55 + "\n")
        for k, v in clinical_results.items():
            if k not in ['name']:
                f.write(f"  {k}: {v}\n")

        f.write("\nHISTOPATHOLOGY SLIDES (ashenafifasilkebede)\n")
        f.write("-"*55 + "\n")
        for k, v in histo_results.items():
            if k not in ['name']:
                f.write(f"  {k}: {v}\n")

    print(f"\nResults saved to results/modality_analysis/")