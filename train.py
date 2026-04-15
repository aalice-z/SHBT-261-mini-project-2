# -*- coding: utf-8 -*-
"""
Training Script for PASCAL VOC 2007 Semantic Segmentation
Trains a U-Net model on the PASCAL VOC 2007 dataset.
"""

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import VOCSegmentation
from torchvision import transforms
import numpy as np
from pathlib import Path
import sys

# Import custom modules
from models import get_model
from losses import get_loss
from metrics import MetricTracker, compute_metrics
from training_utils import Trainer, create_optimizer, LearningRateScheduler


# VOC Classes
VOC_CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", 
    "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", 
    "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]


def get_transforms(image_size=512):
    """Get data transforms for training and validation."""
    
    train_transform = transforms.Compose([
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
    
    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])
    
    target_transform = transforms.Compose([
        transforms.Resize((image_size, image_size), 
                         interpolation=transforms.InterpolationMode.NEAREST),
        transforms.PILToTensor(),
    ])
    
    return train_transform, val_transform, target_transform


def get_data_loaders(dataset_root, batch_size, num_workers, image_size=512):
    """Create training and validation data loaders."""
    
    train_transform, val_transform, target_transform = get_transforms(image_size)
    
    # Load datasets
    train_dataset = VOCSegmentation(
        root=dataset_root,
        year="2007",
        image_set="train",
        download=False,
        transform=train_transform,
        target_transform=target_transform
    )
    
    val_dataset = VOCSegmentation(
        root=dataset_root,
        year="2007",
        image_set="val",
        download=False,
        transform=val_transform,
        target_transform=target_transform
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


def train(args):
    """Main training function."""
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Get data loaders
    print("\n" + "="*70)
    print("📂 Loading Data...")
    print("="*70)
    
    train_loader, val_loader = get_data_loaders(
        args.dataset_root,
        args.batch_size,
        args.num_workers,
        args.image_size
    )
    
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    
    # Create model
    print("\n" + "="*70)
    print("🧠 Creating Model...")
    print("="*70)
    
    model = get_model(
        args.model,
        num_classes=21,
        pretrained=args.pretrained
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {args.model}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Create loss function
    loss_fn = get_loss(args.loss, num_classes=21, ignore_index=255).to(device)
    print(f"Loss function: {args.loss}")
    
    # Create optimizer
    optimizer = create_optimizer(
        model,
        optimizer_type=args.optimizer,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay
    )
    print(f"Optimizer: {args.optimizer}")
    print(f"Learning rate: {args.learning_rate}")
    
    # Create scheduler
    if args.scheduler != 'none':
        scheduler = LearningRateScheduler.get_scheduler(
            optimizer,
            scheduler_type=args.scheduler,
            T_max=args.epochs,
            step_size=args.step_size,
            gamma=args.gamma
        )
    else:
        scheduler = None
    
    if scheduler is not None:
        print(f"Scheduler: {args.scheduler}")
    
    # Create trainer
    print("\n" + "="*70)
    print("🚀 Starting Training...")
    print("="*70)
    
    trainer = Trainer(
        model=model,
        device=device,
        loss_fn=loss_fn,
        metrics_fn=lambda p, t: compute_metrics(
            p, t, num_classes=21, ignore_index=255,
            class_names=VOC_CLASSES
        ),
        checkpoint_dir=args.checkpoint_dir
    )
    
    # Train model
    train_history, val_history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=args.epochs,
        log_interval=args.log_interval
    )
    
    # Save training state
    trainer.save_training_state("training_state.json")
    
    print("\n" + "="*70)
    print("✨ Training Complete!")
    print("="*70)


def main():
    parser = argparse.ArgumentParser(
        description="Train U-Net for PASCAL VOC 2007 semantic segmentation"
    )
    
    # Model arguments
    parser.add_argument('--model', type=str, default='unet',
                       choices=['unet', 'deeplab'],
                       help='Model architecture')
    parser.add_argument('--pretrained', action='store_true',
                       help='Use pretrained weights')
    
    # Data arguments
    parser.add_argument('--dataset-root', type=str, 
                       default='./archive/VOCtrainval_06-Nov-2007',
                       help='Path to dataset root')
    parser.add_argument('--batch-size', type=int, default=4,
                       help='Batch size')
    parser.add_argument('--num-workers', type=int, default=2,
                       help='Number of data loading workers')
    parser.add_argument('--image-size', type=int, default=512,
                       help='Image size')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-5,
                       help='Weight decay')
    parser.add_argument('--optimizer', type=str, default='adam',
                       choices=['adam', 'adamw', 'sgd'],
                       help='Optimizer type')
    
    # Scheduler arguments
    parser.add_argument('--scheduler', type=str, default='cosine',
                       choices=['step', 'cosine', 'polynomial', 'none'],
                       help='Learning rate scheduler')
    parser.add_argument('--step-size', type=int, default=10,
                       help='Step size for step scheduler')
    parser.add_argument('--gamma', type=float, default=0.5,
                       help='Gamma for step scheduler')
    
    # Loss function arguments
    parser.add_argument('--loss', type=str, default='combined',
                       choices=['ce', 'dice', 'focal', 'combined', 'weighted_ce'],
                       help='Loss function')
    
    # Checkpoint arguments
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints',
                       help='Checkpoint directory')
    parser.add_argument('--log-interval', type=int, default=5,
                       help='Logging interval')
    
    args = parser.parse_args()
    
    # Create checkpoint directory
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    # Train
    train(args)


if __name__ == "__main__":
    main()
