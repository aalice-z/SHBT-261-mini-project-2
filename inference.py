# -*- coding: utf-8 -*-
"""
Inference and Testing Utilities for Semantic Segmentation
Provides utilities for making predictions and evaluating trained models.
"""

import torch
import torch.nn.functional as F
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


class SegmentationPredictor:
    """Make predictions with trained segmentation model."""
    
    def __init__(self, model_path, model_class, num_classes=21, device='cuda'):
        """
        Initialize predictor.
        
        Args:
            model_path: Path to pretrained model weights
            model_class: Model class or factory function
            num_classes: Number of output classes
            device: Device to run inference on
        """
        self.device = device
        self.num_classes = num_classes
        
        # Load model
        if callable(model_class):
            self.model = model_class(num_classes=num_classes)
        else:
            self.model = model_class
        
        checkpoint = torch.load(model_path, map_location=device)
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model.to(device)
        self.model.eval()
    
    def predict(self, image, return_logits=False):
        """
        Predict segmentation for a single image.
        
        Args:
            image: Input image (H, W, 3) with values in [0, 1]
            return_logits: Return logits instead of class indices
        
        Returns:
            Predicted class indices or logits
        """
        # Prepare image
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Normalize if needed
        if image.max() > 1:
            image = image / 255.0
        
        # Convert to tensor
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float()
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        
        # Normalize with ImageNet statistics
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)
        image_tensor = (image_tensor - mean) / std
        
        # Forward pass
        with torch.no_grad():
            logits = self.model(image_tensor)
        
        if return_logits:
            return logits.squeeze(0).cpu()
        else:
            predictions = torch.argmax(logits, dim=1)
            return predictions.squeeze(0).cpu()
    
    def predict_batch(self, images, return_logits=False):
        """
        Predict segmentation for multiple images.
        
        Args:
            images: Batch of images (B, H, W, 3)
            return_logits: Return logits instead of class indices
        
        Returns:
            Batch of predictions
        """
        # Convert to tensor
        images_tensor = torch.from_numpy(images).permute(0, 3, 1, 2).float()
        images_tensor = images_tensor.to(self.device)
        
        # Normalize
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)
        images_tensor = (images_tensor - mean) / std
        
        # Forward pass
        with torch.no_grad():
            logits = self.model(images_tensor)
        
        if return_logits:
            return logits.cpu()
        else:
            predictions = torch.argmax(logits, dim=1)
            return predictions.cpu()


class MaskVisualizer:
    """Colorize and visualize segmentation masks."""
    
    # Standard VOC colormap
    VOC_COLORMAP = np.array([
        [0, 0, 0],           # 0: background
        [128, 0, 0],         # 1: aeroplane
        [0, 128, 0],         # 2: bicycle
        [128, 128, 0],       # 3: bird
        [0, 0, 128],         # 4: boat
        [128, 0, 128],       # 5: bottle
        [0, 128, 128],       # 6: bus
        [192, 0, 0],         # 7: car
        [64, 128, 0],        # 8: cat
        [192, 128, 0],       # 9: chair
        [64, 0, 128],        # 10: cow
        [192, 0, 128],       # 11: diningtable
        [64, 128, 128],      # 12: dog
        [192, 128, 128],     # 13: horse
        [0, 64, 0],          # 14: motorbike
        [128, 64, 0],        # 15: person
        [0, 192, 0],         # 16: pottedplant
        [128, 192, 0],       # 17: sheep
        [0, 64, 128],        # 18: sofa
        [128, 64, 128],      # 19: train
        [0, 192, 128],       # 20: tvmonitor
    ], dtype=np.uint8)
    
    @staticmethod
    def colorize_mask(mask, colormap=None):
        """
        Convert class indices to RGB image.
        
        Args:
            mask: Segmentation mask (H, W) with class indices
            colormap: Color map (num_classes, 3)
        
        Returns:
            RGB image (H, W, 3)
        """
        if colormap is None:
            colormap = MaskVisualizer.VOC_COLORMAP
        
        mask_rgb = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        
        for class_idx in range(len(colormap)):
            class_mask = mask == class_idx
            mask_rgb[class_mask] = colormap[class_idx]
        
        return mask_rgb
    
    @staticmethod
    def visualize_prediction(image, prediction, class_names=None, alpha=0.5):
        """
        Visualize image with segmentation overlay.
        
        Args:
            image: Original image (H, W, 3)
            prediction: Segmentation mask (H, W)
            class_names: List of class names
            alpha: Overlay transparency
        
        Returns:
            Overlayed image
        """
        # Ensure image is in [0, 1]
        if image.max() > 1:
            image = image / 255.0
        
        # Colorize mask
        mask_rgb = MaskVisualizer.colorize_mask(prediction) / 255.0
        
        # Blend
        blended = (1 - alpha) * image + alpha * mask_rgb
        
        return blended
    
    @staticmethod
    def plot_predictions(images, predictions, titles=None, class_names=None):
        """
        Plot images and their predictions side by side.
        
        Args:
            images: List of images (H, W, 3)
            predictions: List of predictions (H, W)
            titles: List of titles
            class_names: List of class names
        """
        n = len(images)
        fig, axes = plt.subplots(n, 2, figsize=(10, 5*n))
        
        if n == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(n):
            # Plot image
            axes[i, 0].imshow(images[i])
            axes[i, 0].set_title(f"Image {i+1}")
            axes[i, 0].axis("off")
            
            # Plot prediction
            mask_rgb = MaskVisualizer.colorize_mask(predictions[i])
            axes[i, 1].imshow(mask_rgb)
            axes[i, 1].set_title(f"Prediction {i+1}")
            axes[i, 1].axis("off")
            
            if titles:
                axes[i, 0].set_title(titles[i] if i < len(titles) else "")
        
        plt.tight_layout()
        return fig


def evaluate_model(model, test_loader, device='cuda', num_classes=21, 
                  class_names=None):
    """
    Evaluate model on test set.
    
    Args:
        model: Trained model
        test_loader: Test data loader
        device: Device for computation
        num_classes: Number of classes
        class_names: List of class names
    
    Returns:
        Dictionary with evaluation metrics
    """
    from metrics import SegmentationMetrics
    
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for images, masks in test_loader:
            images = images.to(device)
            masks = masks.to(device).squeeze(1)
            
            # Forward pass
            logits = model(images)
            predictions = torch.argmax(logits, dim=1)
            
            all_predictions.append(predictions.cpu())
            all_targets.append(masks.cpu())
    
    # Concatenate batches
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    # Compute metrics
    metrics = {
        'pixel_accuracy': SegmentationMetrics.pixel_accuracy(
            all_predictions, all_targets),
        'mean_accuracy': SegmentationMetrics.mean_class_accuracy(
            all_predictions, all_targets, num_classes),
        'miou': SegmentationMetrics.miou(
            all_predictions, all_targets, num_classes),
        'mdice': SegmentationMetrics.dice(
            all_predictions, all_targets, num_classes).mean(),
    }
    
    # Per-class metrics
    iou_per_class = SegmentationMetrics.iou(
        all_predictions, all_targets, num_classes)
    
    if class_names:
        metrics['iou_per_class'] = {
            class_names[i]: iou_per_class[i] 
            for i in range(len(class_names))
        }
    
    return metrics


def load_checkpoint(model, checkpoint_path, device='cuda'):
    """
    Load model from checkpoint.
    
    Args:
        model: Model to load into
        checkpoint_path: Path to checkpoint
        device: Device for loading
    
    Returns:
        Model and checkpoint data
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    return model, checkpoint


def save_predictions(predictions, output_dir, file_names, colormap=None):
    """
    Save predictions as colored PNG images.
    
    Args:
        predictions: List or batch of predictions
        output_dir: Directory to save images
        file_names: Names for output files
        colormap: Color map for visualization
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    
    for i, (pred, fname) in enumerate(zip(predictions, file_names)):
        # Colorize
        mask_rgb = MaskVisualizer.colorize_mask(pred, colormap)
        
        # Save
        output_path = output_dir / f"{Path(fname).stem}_pred.png"
        Image.fromarray(mask_rgb).save(output_path)


if __name__ == "__main__":
    print("Inference utilities loaded successfully!")
