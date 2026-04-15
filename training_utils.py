# -*- coding: utf-8 -*-
"""
Training Utilities for Semantic Segmentation
Includes checkpointing, learning rate scheduling, and training helpers.
"""

import os
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, PolynomialLR
import json
from pathlib import Path
from datetime import datetime


class CheckpointManager:
    """Manages model checkpoints and training state."""
    
    def __init__(self, checkpoint_dir='./checkpoints', keep_best=True, keep_last_n=3):
        """
        Args:
            checkpoint_dir: Directory to save checkpoints
            keep_best: Keep the best checkpoint based on validation metric
            keep_last_n: Number of latest checkpoints to keep
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.keep_best = keep_best
        self.keep_last_n = keep_last_n
        self.best_metric = None
        self.checkpoint_history = []
    
    def save(self, model, optimizer, scheduler, epoch, metrics, 
            is_best=False, tag=None):
        """
        Save checkpoint.
        
        Args:
            model: Model to save
            optimizer: Optimizer state
            scheduler: Scheduler state
            epoch: Current epoch
            metrics: Dictionary of metrics
            is_best: Whether this is the best model
            tag: Custom tag for checkpoint name
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat(),
        }
        
        # Save checkpoint
        if tag:
            filename = f"checkpoint_epoch_{epoch:03d}_{tag}.pt"
        else:
            filename = f"checkpoint_epoch_{epoch:03d}.pt"
        
        filepath = self.checkpoint_dir / filename
        torch.save(checkpoint, filepath)
        self.checkpoint_history.append(str(filepath))
        
        print(f"Checkpoint saved: {filepath}")
        
        # Save best checkpoint
        if is_best and self.keep_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            print(f"Best checkpoint saved: {best_path}")
        
        # Clean old checkpoints
        self._cleanup_old_checkpoints()
        
        return filepath
    
    def load(self, model, optimizer=None, scheduler=None, 
            checkpoint_path=None, best=False):
        """
        Load checkpoint.
        
        Args:
            model: Model to load into
            optimizer: Optimizer to load into (optional)
            scheduler: Scheduler to load into (optional)
            checkpoint_path: Path to checkpoint (or use best model)
            best: Load best model
        """
        if best:
            checkpoint_path = self.checkpoint_dir / "best_model.pt"
        elif checkpoint_path is None:
            # Load latest checkpoint
            checkpoint_path = self.checkpoint_dir / "latest_checkpoint.pt"
        
        if not Path(checkpoint_path).exists():
            print(f"Checkpoint not found: {checkpoint_path}")
            return 0
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            if checkpoint['scheduler_state_dict'] is not None:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        epoch = checkpoint.get('epoch', 0)
        metrics = checkpoint.get('metrics', {})
        
        print(f"Checkpoint loaded from: {checkpoint_path}")
        print(f"Epoch: {epoch}, Metrics: {metrics}")
        
        return epoch
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints keeping only the last N."""
        if len(self.checkpoint_history) > self.keep_last_n:
            old_checkpoint = self.checkpoint_history.pop(0)
            if os.path.exists(old_checkpoint):
                os.remove(old_checkpoint)
                print(f"Removed old checkpoint: {old_checkpoint}")


class LearningRateScheduler:
    """Wrapper for learning rate scheduling strategies."""
    
    @staticmethod
    def get_scheduler(optimizer, scheduler_type='step', **kwargs):
        """
        Create learning rate scheduler.
        
        Args:
            optimizer: PyTorch optimizer
            scheduler_type: Type of scheduler ('step', 'cosine', 'polynomial')
            **kwargs: Arguments for specific scheduler
        
        Returns:
            Scheduler instance
        """
        if scheduler_type == 'step':
            return StepLR(
                optimizer,
                step_size=kwargs.get('step_size', 10),
                gamma=kwargs.get('gamma', 0.5)
            )
        elif scheduler_type == 'cosine':
            return CosineAnnealingLR(
                optimizer,
                T_max=kwargs.get('T_max', 100),
                eta_min=kwargs.get('eta_min', 0)
            )
        elif scheduler_type == 'polynomial':
            return PolynomialLR(
                optimizer,
                total_iters=kwargs.get('total_iters', 100),
                power=kwargs.get('power', 0.9)
            )
        elif scheduler_type == 'none':
            return None
        else:
            raise ValueError(f"Unknown scheduler: {scheduler_type}")


class Trainer:
    """Trainer class for semantic segmentation models."""
    
    def __init__(self, model, device, loss_fn, metrics_fn, 
                checkpoint_dir='./checkpoints'):
        """
        Args:
            model: Segmentation model
            device: Device to train on
            loss_fn: Loss function
            metrics_fn: Metrics computation function
            checkpoint_dir: Directory for checkpoints
        """
        self.model = model
        self.device = device
        self.loss_fn = loss_fn
        self.metrics_fn = metrics_fn
        self.checkpoint_manager = CheckpointManager(checkpoint_dir)
        
        self.best_miou = 0.0
        self.train_history = {
            'loss': [],
            'miou': [],
            'pixel_acc': [],
        }
        self.val_history = {
            'loss': [],
            'miou': [],
            'pixel_acc': [],
        }
    
    def train_epoch(self, train_loader, optimizer):
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            optimizer: Optimizer instance
        
        Returns:
            Average loss and metrics for the epoch
        """
        self.model.train()
        total_loss = 0.0
        batch_count = 0
        
        for batch_idx, (images, masks) in enumerate(train_loader):
            images = images.to(self.device)
            masks = masks.to(self.device).squeeze(1).long()
            
            # Forward pass
            optimizer.zero_grad()
            logits = self.model(images)
            loss = self.loss_fn(logits, masks)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            batch_count += 1
            
            if (batch_idx + 1) % 10 == 0:
                print(f"  Batch [{batch_idx+1}/{len(train_loader)}], "
                      f"Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / batch_count
        return {'loss': avg_loss}
    
    def validate(self, val_loader):
        """
        Validate the model.
        
        Args:
            val_loader: Validation data loader
        
        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        batch_count = 0
        
        with torch.no_grad():
            for batch_idx, (images, masks) in enumerate(val_loader):
                images = images.to(self.device)
                masks = masks.to(self.device).squeeze(1).long()
                
                # Forward pass
                logits = self.model(images)
                loss = self.loss_fn(logits, masks)
                
                # Get predictions
                predictions = torch.argmax(logits, dim=1)
                
                all_predictions.append(predictions.cpu())
                all_targets.append(masks.cpu())
                total_loss += loss.item()
                batch_count += 1
        
        # Compute metrics
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        metrics = self.metrics_fn(all_predictions, all_targets)
        metrics['loss'] = total_loss / batch_count
        
        return metrics
    
    def fit(self, train_loader, val_loader, optimizer, scheduler=None,
           num_epochs=100, log_interval=5):
        """
        Full training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            optimizer: Optimizer
            scheduler: Learning rate scheduler (optional)
            num_epochs: Number of epochs to train
            log_interval: Interval for logging
        """
        print(f"Starting training for {num_epochs} epochs...")
        
        for epoch in range(num_epochs):
            print(f"\nEpoch [{epoch+1}/{num_epochs}]")
            
            # Train
            train_metrics = self.train_epoch(train_loader, optimizer)
            self.train_history['loss'].append(train_metrics['loss'])
            
            # Validate
            if val_loader is not None:
                val_metrics = self.validate(val_loader)
                self.val_history['loss'].append(val_metrics['loss'])
                self.val_history['miou'].append(val_metrics.get('miou', 0.0))
                self.val_history['pixel_acc'].append(
                    val_metrics.get('pixel_accuracy', 0.0)
                )
                
                is_best = val_metrics.get('miou', 0.0) > self.best_miou
                if is_best:
                    self.best_miou = val_metrics.get('miou', 0.0)
                
                print(f"Train Loss: {train_metrics['loss']:.4f}")
                print(f"Val Loss: {val_metrics['loss']:.4f}, "
                      f"mIoU: {val_metrics.get('miou', 0.0):.4f}, "
                      f"Pixel Acc: {val_metrics.get('pixel_accuracy', 0.0):.4f}")
                
                # Save checkpoint
                if (epoch + 1) % log_interval == 0 or is_best:
                    self.checkpoint_manager.save(
                        self.model, optimizer, scheduler, epoch + 1,
                        val_metrics, is_best=is_best
                    )
            else:
                print(f"Train Loss: {train_metrics['loss']:.4f}")
                
                if (epoch + 1) % log_interval == 0:
                    self.checkpoint_manager.save(
                        self.model, optimizer, scheduler, epoch + 1,
                        train_metrics, is_best=False
                    )
            
            # Update learning rate
            if scheduler is not None:
                scheduler.step()
        
        print("\nTraining completed!")
        return self.train_history, self.val_history
    
    def save_training_state(self, filepath):
        """Save training history."""
        state = {
            'train_history': self.train_history,
            'val_history': self.val_history,
            'best_miou': self.best_miou,
        }
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=4)
        print(f"Training state saved: {filepath}")
    
    def load_training_state(self, filepath):
        """Load training history."""
        with open(filepath, 'r') as f:
            state = json.load(f)
        self.train_history = state['train_history']
        self.val_history = state['val_history']
        self.best_miou = state['best_miou']
        print(f"Training state loaded: {filepath}")


def create_optimizer(model, optimizer_type='adam', learning_rate=0.001, **kwargs):
    """
    Create optimizer for training.
    
    Args:
        model: Model to optimize
        optimizer_type: Type of optimizer ('adam', 'adamw', 'sgd')
        learning_rate: Learning rate
        **kwargs: Additional optimizer arguments
    
    Returns:
        Optimizer instance
    """
    params = model.parameters()
    
    if optimizer_type.lower() == 'adam':
        return optim.Adam(params, lr=learning_rate, **kwargs)
    elif optimizer_type.lower() == 'adamw':
        return optim.AdamW(params, lr=learning_rate, **kwargs)
    elif optimizer_type.lower() == 'sgd':
        return optim.SGD(params, lr=learning_rate, momentum=0.9, **kwargs)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")


if __name__ == "__main__":
    # Test checkpoint manager
    print("Testing CheckpointManager...")
    checkpoint_mgr = CheckpointManager('./test_checkpoints')
    
    # Create dummy model and optimizer
    model = torch.nn.Linear(10, 5)
    optimizer = torch.optim.Adam(model.parameters())
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
    
    # Save checkpoint
    metrics = {'loss': 0.5, 'miou': 0.7}
    checkpoint_mgr.save(model, optimizer, scheduler, epoch=1, 
                       metrics=metrics, is_best=True)
    
    print("CheckpointManager test passed!")
