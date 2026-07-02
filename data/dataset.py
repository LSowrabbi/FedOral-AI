import os
import random
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class OralCancerDataset(Dataset):
    """
    PyTorch Dataset for Oral Cancer binary classification.
    Combines both dataset versions into one unified dataset.
    
    Labels: 0 = NON CANCER, 1 = CANCER
    """

    def __init__(self, root_dirs, split='train', transform=None):
        """
        Args:
            root_dirs (list): List of root directories containing 
                              CANCER and NON CANCER folders
            split     (str):  'train', 'val', or 'test'
            transform:        Optional torchvision transforms
        """
        self.samples = []  # list of (image_path, label) tuples
        self.transform = transform
        self.split = split
        self.classes = ['NON CANCER', 'CANCER']
        self.class_to_idx = {'NON CANCER': 0, 'CANCER': 1}

        # Load images from all root directories
        for root_dir in root_dirs:
            for class_name in self.classes:
                class_dir = os.path.join(root_dir, class_name)
                if not os.path.exists(class_dir):
                    print(f"Warning: {class_dir} not found, skipping.")
                    continue
                label = self.class_to_idx[class_name]
                for img_file in sorted(os.listdir(class_dir)):
                    if img_file.lower().endswith(
                            ('.jpg', '.jpeg', '.png', '.bmp')):
                        self.samples.append(
                            (os.path.join(class_dir, img_file), label)
                        )

        #  Shuffle before splitting
        random.seed(42)  # fixed seed for reproducible splits
        random.shuffle(self.samples)

        # Split dataset: 70% train, 15% val, 15% test
        total = len(self.samples)
        train_end = int(0.70 * total)
        val_end = int(0.85 * total)

        if split == 'train':
            self.samples = self.samples[:train_end]
        elif split == 'val':
            self.samples = self.samples[train_end:val_end]
        elif split == 'test':
            self.samples = self.samples[val_end:]

        print(f"{split.upper()} set: {len(self.samples)} images")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label


# Default transforms

def get_transforms(split='train'):
    """
    Returns appropriate transforms for each split.
    Train: augmentation + normalization
    Val/Test: resize + normalization only
    """
    # ImageNet mean and std (standard for pretrained models)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if split == 'train':
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2,
                saturation=0.2, hue=0.1
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])


# Test

if __name__ == '__main__':
    root_dirs = [
        'data/raw/oral-cancer-dataset/Oral Cancer/Oral Cancer Dataset',
        'data/raw/oral-cancer-dataset/Oral cancer Dataset 2.0/OC Dataset kaggle new'
    ]

    for split in ['train', 'val', 'test']:
        dataset = OralCancerDataset(
            root_dirs=root_dirs,
            split=split,
            transform=get_transforms(split)
        )

    print(f"\nTotal images: {1700}")
    print(f"Classes: {dataset.classes}")
    print(f"Sample item shape: {dataset[0][0].shape}")
    print(f"Sample label: {dataset[0][1]}")