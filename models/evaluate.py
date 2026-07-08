import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, auc, confusion_matrix,
    ConfusionMatrixDisplay
)

sys.path.insert(0, '.')
from data.dataset import OralCancerDataset, get_transforms
from models.efficientnet_baseline import OralCancerEfficientNet
from utils.device import get_device
from torch.utils.data import DataLoader

def evaluate_and_plot(checkpoint_path='models/checkpoints/best_model.pth'):
    device = get_device()

    # Load model
    model = OralCancerEfficientNet(
        num_classes=2, freeze_backbone=True
    ).to(device)
    model.load_state_dict(torch.load(
        checkpoint_path, map_location=device
    ))
    model.eval()

    # Load test set
    root_dirs = [
        'data/raw/oral-cancer-dataset/Oral Cancer/Oral Cancer Dataset',
        'data/raw/oral-cancer-dataset/Oral cancer Dataset 2.0/OC Dataset kaggle new',
        'data/raw/kaggle-oral-ashen/train',
        'data/raw/kaggle-oral-ashen/val',
        'data/raw/kaggle-oral-ashen/test',
    ]
    test_ds = OralCancerDataset(
        root_dirs, 'test', get_transforms('test')
    )
    test_loader = DataLoader(
        test_ds, batch_size=32,
        shuffle=False, num_workers=2
    )

    all_labels, all_preds, all_probs = [], [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = torch.argmax(outputs, dim=1)
            all_labels.extend(labels.numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_labels = np.array(all_labels)
    all_preds  = np.array(all_preds)
    all_probs  = np.array(all_probs)

    os.makedirs('results/figures', exist_ok=True)

    # ROC Curve
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='#534AB7', lw=2,
             label=f'EfficientNet-B0 (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=1,
             label='Random Classifier (AUC = 0.500)')
    plt.fill_between(fpr, tpr, alpha=0.1, color='#534AB7')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate\n(1 - Specificity)',
               fontsize=12)
    plt.ylabel('True Positive Rate\n(Sensitivity / Recall)',
               fontsize=12)
    plt.title(
        'ROC Curve — FedOral-AI Oral Cancer Detection\n'
        'EfficientNet-B0 · 6,892 images · Seed=42',
        fontsize=13, fontweight='bold'
    )
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/figures/roc_curve.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f'ROC curve saved — AUC: {roc_auc:.4f}')

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(7, 6))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=['NON CANCER', 'CANCER']
    )
    disp.plot(ax=ax, cmap='Blues', colorbar=False)
    ax.set_title(
        'Confusion Matrix — EfficientNet-B0\n'
        'FedOral-AI · Test Set',
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig('results/figures/confusion_matrix.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Confusion matrix saved')
    print(f'    TP={cm[1,1]}  FP={cm[0,1]}')
    print(f'    FN={cm[1,0]}  TN={cm[0,0]}')

    # Training Curves
    import pandas as pd
    log_path = 'results/training/training_log.csv'
    if os.path.exists(log_path):
        df = pd.read_csv(log_path)
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Loss curves
        axes[0].plot(df['epoch'], df['train_loss'],
                     color='#534AB7', lw=2, label='Train Loss')
        axes[0].plot(df['epoch'], df['val_loss'],
                     color='#D85A30', lw=2,
                     linestyle='--', label='Val Loss')
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Loss', fontsize=12)
        axes[0].set_title('Training vs Validation Loss',
                          fontsize=13, fontweight='bold')
        axes[0].legend(fontsize=11)
        axes[0].grid(alpha=0.3)

        # Accuracy curves
        axes[1].plot(df['epoch'], df['train_acc'],
                     color='#534AB7', lw=2,
                     label='Train Accuracy')
        axes[1].plot(df['epoch'], df['val_acc'],
                     color='#D85A30', lw=2,
                     linestyle='--', label='Val Accuracy')
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Accuracy', fontsize=12)
        axes[1].set_title('Training vs Validation Accuracy',
                          fontsize=13, fontweight='bold')
        axes[1].legend(fontsize=11)
        axes[1].grid(alpha=0.3)

        plt.suptitle(
            'FedOral-AI Training Curves — EfficientNet-B0',
            fontsize=14, fontweight='bold', y=1.02
        )
        plt.tight_layout()
        plt.savefig('results/figures/training_curves.png',
                    dpi=150, bbox_inches='tight')
        plt.close()
        print('Training curves saved')

    print('\nAll figures saved to results/figures/')
    return roc_auc, cm

if __name__ == '__main__':
    evaluate_and_plot()