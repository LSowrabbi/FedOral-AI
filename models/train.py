import os
import sys
import csv
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import OralCancerDataset, get_transforms
from models.resnet_baseline import OralCancerResNet
from models.efficientnet_baseline import OralCancerEfficientNet
from utils.device import get_device


# Early Stopping
class EarlyStopping:
    """
    Stops training when validation loss stops improving.
    Saves the best model checkpoint automatically.
    """

    def __init__(self, patience=5, min_delta=0.001,
                 checkpoint_path='models/checkpoints/best_model.pth'):
        self.patience = patience
        self.min_delta = min_delta
        self.checkpoint_path = checkpoint_path
        self.best_loss = float('inf')
        self.counter = 0
        self.early_stop = False
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            # Validation loss improved — save checkpoint
            self.best_loss = val_loss
            self.counter = 0
            torch.save(model.state_dict(), self.checkpoint_path)
            print(f"Val loss improved → checkpoint saved")
        else:
            # No improvement
            self.counter += 1
            print(f"No improvement {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
                print(f"Early stopping triggered")


# Metrics
def compute_metrics(all_labels, all_preds, all_probs):
    """Compute accuracy, precision, recall, F1, AUC-ROC."""
    from sklearn.metrics import (accuracy_score, precision_score,
                                  recall_score, f1_score, roc_auc_score)

    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)

    try:
        auc = roc_auc_score(all_labels, all_probs)
    except Exception:
        auc = 0.0

    return acc, prec, rec, f1, auc


# Train one epoch
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_labels, all_preds, all_probs = [], [], []

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass — update weights
        loss.backward()
        optimizer.step()

        # Track metrics
        running_loss += loss.item()
        probs = torch.softmax(outputs, dim=1)[:, 1]
        preds = torch.argmax(outputs, dim=1)

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.detach().cpu().numpy())

        # Progress indicator every 10 batches
        if (batch_idx + 1) % 10 == 0:
            print(f"Batch {batch_idx+1}/{len(loader)} "
                  f"loss: {loss.item():.4f}")

    avg_loss = running_loss / len(loader)
    acc, prec, rec, f1, auc = compute_metrics(
        all_labels, all_preds, all_probs)

    return avg_loss, acc, prec, rec, f1, auc


# Validate
def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_labels, all_preds, all_probs = [], [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = torch.argmax(outputs, dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    avg_loss = running_loss / len(loader)
    acc, prec, rec, f1, auc = compute_metrics(
        all_labels, all_preds, all_probs)

    return avg_loss, acc, prec, rec, f1, auc


# Main training loop
def train(num_epochs=30, batch_size=32, learning_rate=0.001,
          freeze_backbone=True):

    device = get_device()

    # Dataset & DataLoaders
    root_dirs = [
        'data/raw/oral-cancer-dataset/Oral Cancer/Oral Cancer Dataset',
        'data/raw/oral-cancer-dataset/Oral cancer Dataset 2.0/OC Dataset kaggle new',
        'data/raw/kaggle-oral-ashen/train',
        'data/raw/kaggle-oral-ashen/val',
        'data/raw/kaggle-oral-ashen/test',
        'data/raw/vidit-oral/oral_cancer/train',
        'data/raw/vidit-oral/oral_cancer/val',
        'data/raw/vidit-oral/oral_cancer/test',
    ]

    train_ds = OralCancerDataset(root_dirs, 'train', get_transforms('train'))
    val_ds   = OralCancerDataset(root_dirs, 'val',   get_transforms('val'))
    test_ds  = OralCancerDataset(root_dirs, 'test',  get_transforms('test'))

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                               shuffle=True, num_workers=2,
                               pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size,
                               shuffle=False, num_workers=2,
                               pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=batch_size,
                               shuffle=False, num_workers=2,
                               pin_memory=True)

    # Handle imbalance in class weights
    # Updated weights for combined dataset (14,532 images)
    # NON CANCER: 43.4% · CANCER: 56.6%
    total = 14532
    weight_non_cancer = total / (2 * 6308)   # 1.151
    weight_cancer     = total / (2 * 8224)   # 0.884
    class_weights = torch.tensor(
        [weight_non_cancer, weight_cancer]).to(device)

    print(f"Class weights: NON CANCER={weight_non_cancer:.3f}, "
        f"CANCER={weight_cancer:.3f}")

    # Model
    model = OralCancerEfficientNet(
        num_classes=2,
        freeze_backbone=freeze_backbone
    ).to(device)
    model.get_param_count()

    # Loss & Optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate
    )

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5,
        patience=3
    )

    # Early stopping
    early_stopping = EarlyStopping(
        patience=5,
        checkpoint_path='models/checkpoints/best_model.pth'
    )

    # CSV logging
    os.makedirs('results/training', exist_ok=True)
    log_path = 'results/training/training_log.csv'

    with open(log_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'train_acc',
                          'train_f1', 'val_loss', 'val_acc',
                          'val_f1', 'val_auc', 'lr'])

    # Training loop
    print(f"\n{'='*55}")
    print(f"Training FedOral-AI ResNet-18")
    print(f"Epochs: {num_epochs} | Batch: {batch_size} | "
          f"LR: {learning_rate} | Device: {device}")
    print(f"{'='*55}\n")

    best_val_acc = 0.0
    start_time = time.time()

    for epoch in range(1, num_epochs + 1):
        print(f"Epoch {epoch}/{num_epochs}")
        print(f"{'-'*40}")

        # Train
        t_loss, t_acc, t_prec, t_rec, t_f1, t_auc = \
            train_one_epoch(model, train_loader,
                            criterion, optimizer, device)

        # Validate
        v_loss, v_acc, v_prec, v_rec, v_f1, v_auc = \
            validate(model, val_loader, criterion, device)

        # Learning rate scheduler step
        scheduler.step(v_loss)
        current_lr = optimizer.param_groups[0]['lr']

        # Track best validation accuracy
        if v_acc > best_val_acc:
            best_val_acc = v_acc

        # Print epoch summary
        print(f"\n   Train → Loss: {t_loss:.4f} | "
              f"Acc: {t_acc:.4f} | F1: {t_f1:.4f}")
        print(f"   Val   → Loss: {v_loss:.4f} | "
              f"Acc: {v_acc:.4f} | F1: {v_f1:.4f} | "
              f"AUC: {v_auc:.4f}")

        # Log to CSV
        with open(log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, f"{t_loss:.4f}",
                              f"{t_acc:.4f}", f"{t_f1:.4f}",
                              f"{v_loss:.4f}", f"{v_acc:.4f}",
                              f"{v_f1:.4f}", f"{v_auc:.4f}",
                              f"{current_lr:.6f}"])

        # Early stopping check
        early_stopping(v_loss, model)
        if early_stopping.early_stop:
            print(f"\nTraining stopped at epoch {epoch}")
            break

        print()

    # Training complete
    elapsed = time.time() - start_time
    print(f"\n{'='*55}")
    print(f"Training complete in {elapsed/60:.1f} minutes")
    print(f"Best validation accuracy: {best_val_acc:.4f} "
          f"({best_val_acc*100:.1f}%)")
    print(f"Log saved to: {log_path}")
    print(f"{'='*55}")

    # Final test evaluation
    print(f"\nLoading best checkpoint for test evaluation...")
    model.load_state_dict(torch.load(
        'models/checkpoints/best_model.pth',
        map_location=device))

    test_loss, test_acc, test_prec, test_rec, test_f1, test_auc = \
        validate(model, test_loader, criterion, device)

    print(f"\n{'='*55}")
    print(f"TEST SET RESULTS")
    print(f"{'='*55}")
    print(f"Accuracy  : {test_acc:.4f} ({test_acc*100:.1f}%)")
    print(f"Precision : {test_prec:.4f}")
    print(f"Recall    : {test_rec:.4f}")
    print(f"F1 Score  : {test_f1:.4f}")
    print(f"AUC-ROC   : {test_auc:.4f}")
    print(f"{'='*55}")

    # Save test results
    with open('results/training/test_results.txt', 'w') as f:
        f.write(f"TEST SET RESULTS\n")
        f.write(f"{'='*40}\n")
        f.write(f"Accuracy  : {test_acc:.4f}\n")
        f.write(f"Precision : {test_prec:.4f}\n")
        f.write(f"Recall    : {test_rec:.4f}\n")
        f.write(f"F1 Score  : {test_f1:.4f}\n")
        f.write(f"AUC-ROC   : {test_auc:.4f}\n")

    print(f"\nTest results saved to results/training/test_results.txt")

    return model


# Entry point 
if __name__ == '__main__':
    train(
        num_epochs=30,
        batch_size=32,
        learning_rate=0.001,
        freeze_backbone=True
    )