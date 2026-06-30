import torch
import torch.nn as nn
from torchvision import models


class OralCancerResNet(nn.Module):
    """
    ResNet-18 fine-tuned for binary oral cancer classification.
    Uses ImageNet pretrained weights — transfer learning.
    
    Input:  batch of images (batch_size, 3, 224, 224)
    Output: class logits (batch_size, 2) → [NON CANCER, CANCER]
    """

    def __init__(self, num_classes=2, freeze_backbone=True):
        super(OralCancerResNet, self).__init__()

        # Load pretrained ResNet-18
        self.model = models.resnet18(weights='IMAGENET1K_V1')

        # Freeze backbone layers (optional)
        # Frozen layers keep ImageNet features intact
        # Only the final layer learns oral cancer features
        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False

        # Replace final fully connected layer
        # Original ResNet-18 final layer: 512 → 1000 (ImageNet classes)
        # Our layer: 512 → 2 (NON CANCER, CANCER)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(p=0.3),           # prevent overfitting
            nn.Linear(in_features, 256), # intermediate layer
            nn.ReLU(),                   # activation
            nn.Dropout(p=0.2),           # second dropout
            nn.Linear(256, num_classes)  # final classification
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


# Test
if __name__ == '__main__':
    import sys
    sys.path.append('..')
    from utils.device import get_device

    device = get_device()

    # Build model
    print("\nBuilding OralCancerResNet")
    model = OralCancerResNet(num_classes=2, freeze_backbone=True)
    model = model.to(device)
    model.get_param_count()

    # Test forward pass with dummy batch
    print("\nTesting forward pass")
    dummy_batch = torch.randn(4, 3, 224, 224).to(device)
    output = model(dummy_batch)
    print(f"Input shape:  {dummy_batch.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output (logits): {output}")

    print("\nResNet-18 model working correctly on", device)