# -*- coding: utf-8 -*-
"""
Evaluation Metrics for Semantic Segmentation
Includes IoU, mIoU, Dice, accuracy, and other metrics.
"""

import torch
import numpy as np
from collections import defaultdict


class MetricTracker:
    """Tracks and computes segmentation metrics during training/evaluation."""
    
    def __init__(self, num_classes=21, ignore_index=255):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.intersection = np.zeros(self.num_classes)
        self.union = np.zeros(self.num_classes)
        self.target = np.zeros(self.num_classes)
        self.total_loss = 0.0
        self.loss_count = 0
    
    def update(self, predictions, targets, loss=None):
        """
        Update metrics with new predictions and targets.
        
        Args:
            predictions: Model output logits of shape (B, C, H, W)
            targets: Ground truth labels of shape (B, H, W)
            loss: Optional loss value to track
        """
        # Convert logits to class predictions
        if predictions.dim() == 4:  # (B, C, H, W)
            preds = torch.argmax(predictions, dim=1)
        else:
            preds = predictions
        
        # Convert to numpy
        if isinstance(preds, torch.Tensor):
            preds = preds.cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.cpu().numpy()
        
        # Flatten batch
        preds = preds.reshape(-1)
        targets = targets.reshape(-1)
        
        # Ignore index
        mask = targets != self.ignore_index
        preds = preds[mask]
        targets = targets[mask]
        
        # Update intersection and union
        for c in range(self.num_classes):
            intersection = ((preds == c) & (targets == c)).sum()
            union = ((preds == c) | (targets == c)).sum()
            target = (targets == c).sum()
            
            self.intersection[c] += intersection
            self.union[c] += union
            self.target[c] += target
        
        # Update loss
        if loss is not None:
            self.total_loss += loss
            self.loss_count += 1
    
    def get_iou(self):
        """Get IoU per class."""
        iou = np.zeros(self.num_classes)
        for c in range(self.num_classes):
            if self.union[c] == 0:
                iou[c] = 0.0
            else:
                iou[c] = self.intersection[c] / self.union[c]
        return iou
    
    def get_miou(self):
        """Get mean IoU."""
        iou = self.get_iou()
        # Only consider classes that exist in the dataset
        valid = self.target > 0
        if valid.sum() == 0:
            return 0.0
        return iou[valid].mean()
    
    def get_f1(self):
        """Get F1 (Dice) score per class."""
        f1 = np.zeros(self.num_classes)
        for c in range(self.num_classes):
            if (self.intersection[c] * 2 + self.target[c]) == 0:
                f1[c] = 0.0
            else:
                f1[c] = (2 * self.intersection[c]) / (self.intersection[c] + self.target[c])
        return f1
    
    def get_mf1(self):
        """Get mean F1 score."""
        f1 = self.get_f1()
        valid = self.target > 0
        if valid.sum() == 0:
            return 0.0
        return f1[valid].mean()
    
    def get_pixel_accuracy(self):
        """Get overall pixel accuracy."""
        correct = self.intersection.sum()
        total = self.target.sum()
        if total == 0:
            return 0.0
        return correct / total
    
    def get_mean_accuracy(self):
        """Get mean class accuracy."""
        accuracy = np.zeros(self.num_classes)
        for c in range(self.num_classes):
            if self.target[c] == 0:
                accuracy[c] = 0.0
            else:
                accuracy[c] = self.intersection[c] / self.target[c]
        
        valid = self.target > 0
        if valid.sum() == 0:
            return 0.0
        return accuracy[valid].mean()
    
    def get_loss(self):
        """Get average loss."""
        if self.loss_count == 0:
            return 0.0
        return self.total_loss / self.loss_count
    
    def get_summary(self, class_names=None):
        """Get summary of all metrics."""
        summary = {
            'loss': self.get_loss(),
            'pixel_accuracy': self.get_pixel_accuracy(),
            'mean_accuracy': self.get_mean_accuracy(),
            'miou': self.get_miou(),
            'mf1': self.get_mf1(),
        }
        
        # Per-class metrics
        iou = self.get_iou()
        f1 = self.get_f1()
        
        if class_names is not None:
            summary['iou_per_class'] = {
                class_names[c]: iou[c] for c in range(len(class_names))
            }
            summary['f1_per_class'] = {
                class_names[c]: f1[c] for c in range(len(class_names))
            }
        else:
            summary['iou_per_class'] = {f'class_{c}': iou[c] 
                                       for c in range(self.num_classes)}
            summary['f1_per_class'] = {f'class_{c}': f1[c] 
                                      for c in range(self.num_classes)}
        
        return summary
    
    def __str__(self):
        """Format metrics as string."""
        summary = self.get_summary()
        lines = [
            f"Loss: {summary['loss']:.4f}",
            f"Pixel Accuracy: {summary['pixel_accuracy']:.4f}",
            f"Mean Accuracy: {summary['mean_accuracy']:.4f}",
            f"mIoU: {summary['miou']:.4f}",
            f"mF1: {summary['mf1']:.4f}",
        ]
        return "\n".join(lines)


class SegmentationMetrics:
    """Compute various segmentation metrics from predictions and targets."""
    
    @staticmethod
    def iou(predictions, targets, num_classes=21, ignore_index=255):
        """
        Compute Intersection over Union (IoU) for each class.
        
        Args:
            predictions: Predicted class indices (B, H, W)
            targets: Ground truth labels (B, H, W)
            num_classes: Number of classes
            ignore_index: Index to ignore
        
        Returns:
            IoU for each class
        """
        iou_per_class = []
        
        for class_idx in range(num_classes):
            tp = ((predictions == class_idx) & (targets == class_idx) & 
                  (targets != ignore_index)).sum().item()
            fp = ((predictions == class_idx) & (targets != class_idx) & 
                  (targets != ignore_index)).sum().item()
            fn = ((predictions != class_idx) & (targets == class_idx) & 
                  (targets != ignore_index)).sum().item()
            
            denominator = tp + fp + fn
            if denominator == 0:
                iou_per_class.append(0.0)
            else:
                iou_per_class.append(tp / denominator)
        
        return np.array(iou_per_class)
    
    @staticmethod
    def miou(predictions, targets, num_classes=21, ignore_index=255):
        """Compute mean IoU."""
        iou_array = SegmentationMetrics.iou(predictions, targets, 
                                           num_classes, ignore_index)
        return np.nanmean(iou_array)
    
    @staticmethod
    def dice(predictions, targets, num_classes=21, ignore_index=255):
        """Compute Dice coefficient (F1 score) for each class."""
        dice_per_class = []
        
        for class_idx in range(num_classes):
            tp = ((predictions == class_idx) & (targets == class_idx) & 
                  (targets != ignore_index)).sum().item()
            fp = ((predictions == class_idx) & (targets != class_idx) & 
                  (targets != ignore_index)).sum().item()
            fn = ((predictions != class_idx) & (targets == class_idx) & 
                  (targets != ignore_index)).sum().item()
            
            denominator = 2 * tp + fp + fn
            if denominator == 0:
                dice_per_class.append(0.0)
            else:
                dice_per_class.append(2 * tp / denominator)
        
        return np.array(dice_per_class)
    
    @staticmethod
    def pixel_accuracy(predictions, targets, ignore_index=255):
        """Compute overall pixel accuracy."""
        valid = targets != ignore_index
        correct = ((predictions == targets) & valid).sum().item()
        total = valid.sum().item()
        
        if total == 0:
            return 0.0
        return correct / total
    
    @staticmethod
    def mean_class_accuracy(predictions, targets, num_classes=21, 
                           ignore_index=255):
        """Compute mean accuracy per class."""
        class_acc = []
        
        for class_idx in range(num_classes):
            tp = ((predictions == class_idx) & (targets == class_idx) & 
                  (targets != ignore_index)).sum().item()
            total = ((targets == class_idx) & 
                    (targets != ignore_index)).sum().item()
            
            if total == 0:
                class_acc.append(0.0)
            else:
                class_acc.append(tp / total)
        
        return np.nanmean(class_acc)


def compute_metrics(predictions, targets, num_classes=21, 
                   ignore_index=255, class_names=None):
    """
    Compute comprehensive segmentation metrics.
    
    Args:
        predictions: Model predictions (B, H, W)
        targets: Ground truth labels (B, H, W)
        num_classes: Number of classes
        ignore_index: Index to ignore in computation
        class_names: Optional list of class names
    
    Returns:
        Dictionary with computed metrics
    """
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()
    
    metrics = {
        'pixel_accuracy': SegmentationMetrics.pixel_accuracy(
            predictions, targets, ignore_index),
        'mean_accuracy': SegmentationMetrics.mean_class_accuracy(
            predictions, targets, num_classes, ignore_index),
        'miou': SegmentationMetrics.miou(
            predictions, targets, num_classes, ignore_index),
        'mdice': SegmentationMetrics.dice(
            predictions, targets, num_classes, ignore_index).mean(),
    }
    
    # Per-class metrics
    iou_per_class = SegmentationMetrics.iou(
        predictions, targets, num_classes, ignore_index)
    dice_per_class = SegmentationMetrics.dice(
        predictions, targets, num_classes, ignore_index)
    
    if class_names is not None:
        metrics['iou_per_class'] = {
            class_names[i]: iou_per_class[i] 
            for i in range(len(class_names))
        }
        metrics['dice_per_class'] = {
            class_names[i]: dice_per_class[i] 
            for i in range(len(class_names))
        }
    
    return metrics


if __name__ == "__main__":
    # Test metrics computation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create dummy data
    num_classes = 21
    H, W = 128, 128
    
    predictions = torch.randint(0, num_classes, (4, H, W), device=device)
    targets = torch.randint(0, num_classes, (4, H, W), device=device)
    
    # Compute metrics
    metrics = compute_metrics(predictions, targets, num_classes=num_classes)
    
    print("Segmentation Metrics:")
    print(f"Pixel Accuracy: {metrics['pixel_accuracy']:.4f}")
    print(f"Mean Accuracy: {metrics['mean_accuracy']:.4f}")
    print(f"mIoU: {metrics['miou']:.4f}")
    print(f"mDice: {metrics['mdice']:.4f}")
    
    # Test metric tracker
    print("\n" + "="*50)
    print("MetricTracker Test:")
    tracker = MetricTracker(num_classes=num_classes)
    
    # Simulate multiple batches
    for _ in range(3):
        logits = torch.randn(4, num_classes, H, W, device=device)
        targets = torch.randint(0, num_classes, (4, H, W), device=device)
        loss = torch.tensor(0.5)
        
        tracker.update(logits, targets, loss=loss.item())
    
    print(tracker)
