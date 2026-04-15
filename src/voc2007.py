# -*- coding: utf-8 -*-
"""
Enhanced PASCAL VOC 2007 Data Loading for Semantic Segmentation
Includes comprehensive data exploration, visualization, and preprocessing.

@author: mohae
Updated for semantic segmentation project with U-Net
"""

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import VOCSegmentation
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import json

# ============================
# 1. VOC Dataset Configuration
# ============================

VOC_CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", 
    "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", 
    "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]
NUM_CLASSES = len(VOC_CLASSES)

# Color map for visualization (RGB)
VOC_COLORMAP = [
    (0, 0, 0),           # 0: background
    (128, 0, 0),         # 1: aeroplane (dark red)
    (0, 128, 0),         # 2: bicycle (dark green)
    (128, 128, 0),       # 3: bird (dark yellow)
    (0, 0, 128),         # 4: boat (dark blue)
    (128, 0, 128),       # 5: bottle (dark purple)
    (0, 128, 128),       # 6: bus (dark cyan)
    (192, 0, 0),         # 7: car (light red)
    (64, 128, 0),        # 8: cat (green-brown)
    (192, 128, 0),       # 9: chair (orange)
    (64, 0, 128),        # 10: cow (purple)
    (192, 0, 128),       # 11: diningtable (magenta)
    (64, 128, 128),      # 12: dog (cyan)
    (192, 128, 128),     # 13: horse (light gray)
    (0, 64, 0),          # 14: motorbike (forest green)
    (128, 64, 0),        # 15: person (brown)
    (0, 192, 0),         # 16: pottedplant (bright green)
    (128, 192, 0),       # 17: sheep (light yellow-green)
    (0, 64, 128),        # 18: sofa (teal)
    (128, 64, 128),      # 19: train (purple)
    (0, 192, 128),       # 20: tvmonitor (light cyan)
]

print("="*60)
print("PASCAL VOC 2007 Semantic Segmentation Dataset")
print("="*60)
print(f"Number of classes: {NUM_CLASSES}")
print("Classes:", VOC_CLASSES)


# ============================
# 2. Class Mapping Dictionary
# ============================
class_mapping = {i: cls for i, cls in enumerate(VOC_CLASSES)}

print("\n" + "="*60)
print("📊 PASCAL VOC 2007 Class Mapping:")
print("="*60)
for idx, name in class_mapping.items():
    print(f"{idx:2d} → {name}")


# ============================
# 3. Data Augmentation & Transforms
# ============================

# Training transforms with augmentation
train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.1),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # ImageNet statistics
        std=[0.229, 0.224, 0.225]
    ),
])

# Validation/Test transforms (no augmentation)
val_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

# Target transforms
def get_target_transform():
    return transforms.Compose([
        transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.PILToTensor(),
    ])

# ============================
# 4. Load PASCAL VOC 2007 Dataset
# ============================

print("\n" + "="*60)
print("📂 Loading Dataset...")
print("="*60)

train_dataset = VOCSegmentation(
    root="./archive/VOCtrainval_06-Nov-2007",
    year="2007",
    image_set="train",
    download=False,
    transform=train_transforms,
    target_transform=get_target_transform()
)

val_dataset = VOCSegmentation(
    root="./archive/VOCtrainval_06-Nov-2007",
    year="2007",
    image_set="val",
    download=False,
    transform=val_transforms,
    target_transform=get_target_transform()
)

print("✅ Train samples:", len(train_dataset))
print("✅ Validation samples:", len(val_dataset))

# =============================
# 5. Create DataLoaders Function
# =============================

def get_data_loaders(root_dir="./archive/", batch_size=4, num_workers=2, image_size=256):
    """
    Create train and validation DataLoaders.
    
    Args:
        root_dir: Root directory containing VOC data
        batch_size: Batch size for DataLoader
        num_workers: Number of worker processes
        image_size: Target image size for resizing
    
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Update transforms with new image size
    train_transforms_custom = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.1),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])
    
    val_transforms_custom = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])
    
    def get_target_transform_custom():
        return transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.PILToTensor(),
        ])
    
    train_dataset_custom = VOCSegmentation(
        root=root_dir + "VOCtrainval_06-Nov-2007",
        year="2007",
        image_set="train",
        download=False,
        transform=train_transforms_custom,
        target_transform=get_target_transform_custom()
    )
    
    val_dataset_custom = VOCSegmentation(
        root=root_dir + "VOCtrainval_06-Nov-2007",
        year="2007",
        image_set="val",
        download=False,
        transform=val_transforms_custom,
        target_transform=get_target_transform_custom()
    )
    
    train_loader_custom = DataLoader(
        train_dataset_custom,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader_custom = DataLoader(
        val_dataset_custom,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader_custom, val_loader_custom

# ============================
# 6. Visualization Utilities
# ============================

def denormalize_image(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """Denormalize ImageNet normalized image."""
    img = img.cpu().numpy() if isinstance(img, torch.Tensor) else img
    mean = np.array(mean).reshape(3, 1, 1)
    std = np.array(std).reshape(3, 1, 1)
    img = np.clip((img * std + mean), 0, 1)
    img = np.transpose(img, (1, 2, 0))
    return img


def mask_to_rgb(mask, colormap=VOC_COLORMAP):
    """Convert segmentation mask to RGB using colormap."""
    mask_rgb = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    colormap = np.array(colormap)
    
    for class_idx in range(len(colormap)):
        class_mask = mask == class_idx
        mask_rgb[class_mask] = colormap[class_idx]
    
    return mask_rgb


def show_sample(img, mask):
    """Visualize a single sample with image and segmentation mask."""
    # Denormalize image
    img_visualize = denormalize_image(img)
    
    # Get mask
    mask_np = mask.squeeze().cpu().numpy().astype(np.int64)
    
    # Clean mask: convert invalid values to 0
    mask_np[mask_np > 20] = 0
    
    # Convert mask to RGB
    mask_rgb = mask_to_rgb(mask_np)
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(img_visualize)
    axes[0].set_title("Input Image")
    axes[0].axis("off")
    
    axes[1].imshow(mask_np, cmap="tab20", vmin=0, vmax=20)
    axes[1].set_title("Segmentation Mask (Class Indices)")
    axes[1].axis("off")
    
    axes[2].imshow(mask_rgb)
    axes[2].set_title("Segmentation Mask (Colored)")
    axes[2].axis("off")
    
    plt.tight_layout()
    plt.show()
    
    # Print statistics
    unique_classes, counts = np.unique(mask_np, return_counts=True)
    print(f"\nClass Distribution in Sample:")
    for cls_idx, count in zip(unique_classes, counts):
        if cls_idx < len(VOC_CLASSES):
            pct = 100 * count / mask_np.size
            print(f"  {VOC_CLASSES[cls_idx]:15s}: {count:6d} pixels ({pct:5.2f}%)")


def show_batch_samples(images, masks, num_samples=4):
    """Visualize multiple samples from a batch."""
    num_samples = min(num_samples, len(images))
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))
    
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_samples):
        # Denormalize image
        img = denormalize_image(images[i])
        
        # Get mask
        mask = masks[i].squeeze().cpu().numpy().astype(np.int64)
        mask[mask > 20] = 0
        
        # Convert to RGB
        mask_rgb = mask_to_rgb(mask)
        
        # Plot
        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f"Image {i+1}")
        axes[i, 0].axis("off")
        
        axes[i, 1].imshow(mask, cmap="tab20", vmin=0, vmax=20)
        axes[i, 1].set_title(f"Mask {i+1} (Class Indices)")
        axes[i, 1].axis("off")
        
        axes[i, 2].imshow(mask_rgb)
        axes[i, 2].set_title(f"Mask {i+1} (Colored)")
        axes[i, 2].axis("off")
    
    plt.tight_layout()
    return fig


# ============================
# 8. Show Sample Data
# ============================

if __name__ == "__main__":
    # Create data loaders
    train_loader, val_loader = get_data_loaders(
        root_dir="./archive/",
        batch_size=4,
        num_workers=2,
        image_size=256
    )
    
    print("✅ Train batches:", len(train_loader))
    print("✅ Validation batches:", len(val_loader))
    
    # Get a sample batch
    images, masks = next(iter(train_loader))
    print("\nImage batch shape:", images.shape)    # (B, 3, 256, 256)
    print("Mask batch shape:", masks.shape)      # (B, 1, 256, 256)
    print("Classes in masks:", torch.unique(masks))
    
    print("\n" + "="*60)
    print("🎨 Displaying Sample Image and Mask...")
    print("="*60)
    
    # Show first sample individually
    show_sample(images[0], masks[0])
    
    # Show batch of samples
    print("\n" + "="*60)
    print("🎨 Displaying Batch of Samples...")
    print("="*60)
    fig = show_batch_samples(images, masks, num_samples=2)
    plt.savefig("voc2007_batch_samples.png", dpi=150, bbox_inches='tight')
    print("✅ Batch samples saved to 'voc2007_batch_samples.png'")
    
    print("\n" + "="*60)
    print("✨ Data Loading Complete!")
    print("="*60)
    print(f"\nDataset ready for training:")
    print(f"  • Train set: {len(train_dataset)} samples")
    print(f"  • Validation set: {len(val_dataset)} samples")
    print(f"  • Image size: 256x256")
    print(f"  • Number of classes: {NUM_CLASSES}")
    print(f"  • Batch size: 4")

