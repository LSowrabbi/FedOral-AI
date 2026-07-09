import torch
import torch.nn as nn
from torchvision import models
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class OralCancerMobileNet(nn.Module):
    """
    MobileNetV2 fine-tuned for binary oral cancer classification.
    Designed for resource-constrained deployment — mobile/edge devices.
    Directly relevant to point-of-care screening in low-resource settings.
    
    Input:  batch of images (batch_size, 3, 224, 224)
    Output: class logits (batch_size, 2) → [NON CANCER, CANCER]
    """

    def __init__(self, num_classes=2, freeze_backbone=True):
        super(OralCancerMobileNet, self).__init__()

        # Load pretrained MobileNetV2
        self.model = models.mobilenet_v2(weights='IMAGENET1K_V1')

        # Freeze backbone layers
        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False

        # Replace classifier head
        # Original MobileNetV2 classifier: 1280 → 1000
        # Our classifier: 1280 → 2 (NON CANCER, CANCER)
        in_features = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.model(x)

    def unfreeze_backbone(self):
        """Unfreeze all layers for full fine-tuning."""
        for param in self.model.parameters():
            param.requires_grad = True
        print("Backbone unfrozen — all layers trainable")

    def get_param_count(self):
        """Print trainable vs total parameters."""
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters()
                       if p.requires_grad)
        print(f"Total parameters:     {total:,}")
        print(f"Trainable parameters: {trainable:,}")
        print(f"Frozen parameters:    {total - trainable:,}")
        return total, trainable


# Quick test
if __name__ == '__main__':
    from utils.device import get_device

    device = get_device()

    print("\n── Building OralCancerMobileNet ──")
    model = OralCancerMobileNet(num_classes=2, freeze_backbone=True)
    model = model.to(device)
    model.get_param_count()

    # Test forward pass
    print("\n── Testing forward pass ──")
    dummy_batch = torch.randn(4, 3, 224, 224).to(device)
    output = model(dummy_batch)
    print(f"Input shape:  {dummy_batch.shape}")
    print(f"Output shape: {output.shape}")
    print(f"\nMobileNetV2 working correctly on {device}")