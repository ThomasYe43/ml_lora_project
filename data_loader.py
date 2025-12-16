import torch
import multiprocessing
from torch.utils.data import random_split, DataLoader
from torchvision import datasets, transforms
from pathlib import Path
import random
import numpy as np


# Configuration constants
BATCH_SIZE = 128
NUM_WORKERS = 8
IMG_SIZE = 224
NUM_CLASSES = 101

#print(multiprocessing.cpu_count())

def get_transforms():
    # BASE transforms 
    base_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet stats
            std=[0.229, 0.224, 0.225]
        )
    ])

    # TRAINING transforms 
    train_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.TrivialAugmentWide(),  # Random augmentation
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    return base_transforms, train_transforms

 
def get_food101dataloaders(data_dir="data", batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
                    train_split=0.8, seed=42):
    data_dir = Path(data_dir)
    base_transforms, train_transforms = get_transforms()
    
    # Load full training data 
    full_train_data_with_aug = datasets.Food101(
        root=data_dir,
        split="train",
        transform=train_transforms,  # Augmented
        download=True
    )

    # Load full training data WITHOUT augmentations (for validation subset)
    full_train_data_no_aug = datasets.Food101(
        root=data_dir,
        split="train",
        transform=base_transforms,  # No augmentation
        download=False  # Already downloaded
    )

    # Split indices (reproducible)
    train_size = int(train_split * len(full_train_data_with_aug))
    val_size = len(full_train_data_with_aug) - train_size

    generator = torch.Generator().manual_seed(seed)  # Reproducibility
    train_data, _ = random_split(
        full_train_data_with_aug,  # Training uses augmented data
        [train_size, val_size],
        generator=generator
    )

    _, val_data = random_split(
        full_train_data_no_aug,  # Validation uses clean data
        [train_size, val_size],
        generator=generator  # Same split!
    )

    # Test data (no augmentation)
    test_data = datasets.Food101(
        root=data_dir,
        split="test",
        transform=base_transforms,  # No augmentation
        download=False
    )
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_data,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )
    
    test_loader = DataLoader(
        test_data,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )
    
    print(f"Food-101 DataLoaders created:")
    print(f"  Training samples: {len(train_data)}")
    print(f"  Validation samples: {len(val_data)}")
    print(f"  Test samples: {len(test_data)}")
    
    return train_loader, val_loader, test_loader


def get_cifar100_dataloaders(data_dir="data", batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, val_split=0.5, seed=42):
    data_dir = Path(data_dir)
    base_transforms, train_transforms = get_transforms()
    
    # Load CIFAR-100 training data
    train_data = datasets.CIFAR100(
        root=data_dir,
        train=True,
        transform=train_transforms,
        download=True
    )
    
    # Load CIFAR-100 test data (will split into val and test)
    full_test_data = datasets.CIFAR100(
        root=data_dir,
        train=False,
        transform=base_transforms,
        download=True
    )
    
    # Split test data into validation and test sets
    val_size = int(val_split * len(full_test_data))
    test_size = len(full_test_data) - val_size
    
    generator = torch.Generator().manual_seed(seed)
    val_data, test_data = random_split(
        full_test_data,
        [val_size, test_size],
        generator=generator
    )
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_data,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )
    
    test_loader = DataLoader(
        test_data,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )
    
    print(f"CIFAR-100 DataLoaders created:")
    print(f"  Training samples: {len(train_data)}")
    print(f"  Validation samples: {len(val_data)}")
    print(f"  Test samples: {len(test_data)}")
    
    return train_loader, val_loader, test_loader

