# -*- coding: utf-8 -*-
"""
Loss Functions for Semantic Segmentation
Includes standard cross-entropy, focal loss, and dice loss implementations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropyLoss(nn.Module):
    """Standard cross-entropy loss with optional class weighting and ignore index."""
    def __init__(self, num_classes=21, ignore_index=255, weight=None, reduction='mean'):
        super(CrossEntropyLoss, self).__init__()
        self.loss = nn.CrossEntropyLoss(
            weight=weight,
            ignore_index=ignore_index,
            reduction=reduction
        )
    
    def forward(self, logits, targets):
        """
        Args:
            logits: Predicted logits of shape (B, C, H, W)
            targets: Ground truth labels of shape (B, H, W)
        """
        return self.loss(logits, targets)


class DiceLoss(nn.Module):
    """
    Dice Loss (F1 Loss) for semantic segmentation.
    Useful for imbalanced datasets.
    
    Formula: Loss = 1 - (2 * |X ∩ Y| / (|X| + |Y|))
    """
    def __init__(self, smooth=1.0, ignore_index=255):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index
    
    def forward(self, logits, targets):
        """
        Args:
            logits: Predicted logits of shape (B, C, H, W)
            targets: Ground truth labels of shape (B, H, W)
        """
        # Convert logits to probabilities
        probs = F.softmax(logits, dim=1)
        
        # Convert targets to one-hot encoding
        B, C, H, W = probs.shape
        targets_one_hot = torch.zeros_like(probs)
        for c in range(C):
            targets_one_hot[:, c, :, :] = (targets == c).float()
        
        # Mask out ignore index
        if self.ignore_index is not None:
            mask = (targets != self.ignore_index).float()
            mask = mask.unsqueeze(1).expand_as(probs)
            probs = probs * mask
            targets_one_hot = targets_one_hot * mask
        
        # Calculate dice loss for each class
        intersection = (probs * targets_one_hot).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + targets_one_hot.sum(dim=(2, 3))
        
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1.0 - dice.mean()
        
        return dice_loss


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    Reduces weight for easy examples and focuses on hard examples.
    
    Reference: "Focal Loss for Dense Object Detection"
    Lin et al., ICCV 2017
    """
    def __init__(self, num_classes=21, alpha=0.25, gamma=2.0, 
                 ignore_index=255, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.num_classes = num_classes
    
    def forward(self, logits, targets):
        """
        Args:
            logits: Predicted logits of shape (B, C, H, W)
            targets: Ground truth labels of shape (B, H, W)
        """
        # Reshape for easier computation
        B, C, H, W = logits.shape
        logits = logits.permute(0, 2, 3, 1).contiguous().view(-1, C)
        targets = targets.view(-1)
        
        # Ignore index
        valid = targets != self.ignore_index
        logits = logits[valid]
        targets = targets[valid]
        
        # Calculate focal loss
        probs = F.softmax(logits, dim=1)
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        
        # Get probability of the true class
        p_t = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        # Apply focal term: (1 - p_t)^gamma
        focal_term = (1.0 - p_t) ** self.gamma
        focal_loss = self.alpha * focal_term * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class CombinedLoss(nn.Module):
    """Combined loss function mixing CE and Dice losses."""
    def __init__(self, num_classes=21, weight_ce=1.0, weight_dice=1.0, 
                 ignore_index=255):
        super(CombinedLoss, self).__init__()
        self.ce_loss = CrossEntropyLoss(num_classes=num_classes, 
                                        ignore_index=ignore_index)
        self.dice_loss = DiceLoss(ignore_index=ignore_index)
        self.weight_ce = weight_ce
        self.weight_dice = weight_dice
    
    def forward(self, logits, targets):
        ce = self.ce_loss(logits, targets)
        dice = self.dice_loss(logits, targets)
        return self.weight_ce * ce + self.weight_dice * dice


class WeightedCrossEntropyLoss(nn.Module):
    """
    Weighted cross-entropy loss that accounts for class imbalance.
    Automatically computes class weights from training data distribution.
    """
    def __init__(self, num_classes=21, ignore_index=255):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.register_buffer('class_weights', torch.ones(num_classes))
    
    def compute_weights(self, targets):
        """Compute class weights from target distribution."""
        valid = targets != self.ignore_index
        targets_valid = targets[valid]
        
        # Count samples per class
        class_counts = torch.bincount(targets_valid, minlength=self.num_classes)
        
        # Weight: inverse of class frequency
        weights = torch.zeros(self.num_classes, device=targets.device)
        total = class_counts.sum().float()
        
        for c in range(self.num_classes):
            if class_counts[c] > 0:
                weights[c] = 1.0 / (class_counts[c].float() / total)
            else:
                weights[c] = 1.0
        
        # Normalize weights
        weights = weights / weights.sum() * self.num_classes
        self.class_weights = weights
        
        return weights.to(targets.device)
    
    def forward(self, logits, targets):
        """
        Args:
            logits: Predicted logits of shape (B, C, H, W)
            targets: Ground truth labels of shape (B, H, W)
        """
        loss_fn = nn.CrossEntropyLoss(
            weight=self.class_weights,
            ignore_index=self.ignore_index
        )
        return loss_fn(logits, targets)


def get_loss(loss_name='ce', num_classes=21, ignore_index=255, **kwargs):
    """
    Factory function to instantiate loss functions.
    
    Args:
        loss_name (str): Name of loss ('ce', 'dice', 'focal', 'combined')
        num_classes (int): Number of classes
        ignore_index (int): Index to ignore in loss computation
        **kwargs: Additional arguments for loss function
    
    Returns:
        Loss function instance
    """
    loss_name = loss_name.lower()
    
    if loss_name == 'ce':
        return CrossEntropyLoss(num_classes=num_classes, 
                               ignore_index=ignore_index, **kwargs)
    elif loss_name == 'dice':
        return DiceLoss(ignore_index=ignore_index, **kwargs)
    elif loss_name == 'focal':
        return FocalLoss(num_classes=num_classes, 
                        ignore_index=ignore_index, **kwargs)
    elif loss_name == 'combined':
        return CombinedLoss(num_classes=num_classes, 
                           ignore_index=ignore_index, **kwargs)
    elif loss_name == 'weighted_ce':
        return WeightedCrossEntropyLoss(num_classes=num_classes, 
                                       ignore_index=ignore_index)
    else:
        raise ValueError(f"Unknown loss: {loss_name}")


if __name__ == "__main__":
    # Test losses
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create dummy data
    batch_size, num_classes, h, w = 4, 21, 128, 128
    logits = torch.randn(batch_size, num_classes, h, w, device=device)
    targets = torch.randint(0, num_classes, (batch_size, h, w), device=device)
    
    # Test each loss
    losses_to_test = ['ce', 'dice', 'focal', 'combined']
    
    for loss_name in losses_to_test:
        loss_fn = get_loss(loss_name, num_classes=num_classes)
        loss_fn = loss_fn.to(device)
        loss = loss_fn(logits, targets)
        print(f"{loss_name.upper():15} Loss: {loss.item():.4f}")
