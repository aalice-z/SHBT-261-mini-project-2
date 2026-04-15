#!/usr/bin/env python
"""
Evaluate trained model and generate results report.
"""
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from voc2007 import get_data_loaders
from models import get_model
from metrics import MetricTracker
import argparse


def evaluate_on_split(model, dataloader, device, dataset_name='validation', max_batches=None):
    """Evaluate model on dataset split."""
    model.eval()
    metric_tracker = MetricTracker()
    
    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(dataloader):
            if max_batches and batch_idx >= max_batches:
                break
                
            images = images.to(device)
            masks = masks.to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Get predictions
            preds = torch.argmax(outputs, dim=1)
            
            # Update metrics
            for pred, mask in zip(preds, masks):
                metric_tracker.update(pred.unsqueeze(0), mask.unsqueeze(0))
            
            if (batch_idx + 1) % 10 == 0:
                print(f"  Batch {batch_idx + 1}/{len(dataloader)}")
    
    metrics = {
        'miou': metric_tracker.get_miou(),
        'pixel_acc': metric_tracker.get_pixel_accuracy(),
        'iou_per_class': metric_tracker.get_iou(),
        'class_counts': metric_tracker.target
    }
    return metrics


def plot_training_curves(output_dir='results'):
    """Plot training curves from training_state.json."""
    Path(output_dir).mkdir(exist_ok=True)
    
    with open('training_state.json') as f:
        data = json.load(f)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Training loss
    axes[0, 0].plot(data['train_history']['loss'], label='Train Loss', linewidth=2)
    axes[0, 0].plot(data['val_history']['loss'], label='Val Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training & Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Validation mIoU
    axes[0, 1].plot(data['val_history']['miou'], label='Val mIoU', linewidth=2, color='green')
    axes[0, 1].axhline(y=data['best_miou'], color='r', linestyle='--', label=f'Best mIoU ({data["best_miou"]:.4f})')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('mIoU')
    axes[0, 1].set_title('Validation Mean IoU')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Validation pixel accuracy
    axes[1, 0].plot(data['val_history']['pixel_acc'], label='Val Pixel Acc', linewidth=2, color='orange')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Pixel Accuracy')
    axes[1, 0].set_title('Validation Pixel Accuracy')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Combined metrics comparison
    epochs = range(1, len(data['train_history']['loss']) + 1)
    ax2 = axes[1, 1]
    ax2_2 = ax2.twinx()
    
    l1 = ax2.plot(epochs, data['val_history']['loss'], 'b-', label='Loss', linewidth=2)
    l2 = ax2_2.plot(epochs, data['val_history']['miou'], 'g-', label='mIoU', linewidth=2)
    
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss', color='b')
    ax2_2.set_ylabel('mIoU', color='g')
    ax2.set_title('Loss vs mIoU')
    ax2.tick_params(axis='y', labelcolor='b')
    ax2_2.tick_params(axis='y', labelcolor='g')
    
    lines = l1 + l2
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/training_curves.png', dpi=150, bbox_inches='tight')
    print(f"✓ Training curves saved to {output_dir}/training_curves.png")
    plt.close()


def generate_results_report(val_metrics, output_dir='results'):
    """Generate comprehensive results report."""
    Path(output_dir).mkdir(exist_ok=True)
    
    with open('training_state.json') as f:
        train_data = json.load(f)
    
    report = f"""
# Training Results Report

## Training Summary
- **Total Epochs**: {len(train_data['train_history']['loss'])}
- **Final Train Loss**: {train_data['train_history']['loss'][-1]:.4f}
- **Best Train Loss**: {min(train_data['train_history']['loss']):.4f}
- **Final Val Loss**: {train_data['val_history']['loss'][-1]:.4f}

## Validation Results (Final Checkpoint)
- **Best mIoU during training**: {train_data['best_miou']:.4f}
- **Final mIoU**: {val_metrics['miou']:.4f}
- **Pixel Accuracy**: {val_metrics['pixel_acc']:.4f}
- **Dice Coefficient**: {val_metrics.get('dice', 'N/A')}

## Per-Class Performance

| Class | IoU | Count |
|-------|-----|-------|
"""
    
    class_names = [
        'Background', 'Aeroplane', 'Bicycle', 'Bird', 'Boat', 'Bottle',
        'Bus', 'Car', 'Cat', 'Chair', 'Cow', 'Diningtable', 'Dog',
        'Horse', 'Motorbike', 'Person', 'Pottedplant', 'Sheep', 'Sofa',
        'Train', 'Tvmonitor'
    ]
    
    if 'iou_per_class' in val_metrics:
        for idx, (class_name, iou) in enumerate(zip(class_names, val_metrics['iou_per_class'])):
            count = val_metrics.get('class_counts', [0]*21)[idx]
            report += f"| {class_name} | {iou:.4f} | {int(count)} |\n"
    
    report += f"""
## Key Observations

1. **Model Convergence**: Training loss decreased from {train_data['train_history']['loss'][0]:.4f} to {train_data['train_history']['loss'][-1]:.4f}
2. **Validation Performance**: mIoU of {val_metrics['miou']:.4f} achieved on validation set
3. **Pixel Accuracy**: {val_metrics['pixel_acc']:.4f} indicates balanced predictions across pixels
4. **Training Stability**: Loss curves show {('stable' if np.std(train_data['train_history']['loss'][-10:]) < 0.1 else 'variable')} convergence

## Files Generated
- `checkpoint_best.pt` - Best model checkpoint
- `checkpoint_epoch_050.pt` - Final epoch checkpoint
- `training_state.json` - Complete training history
- `training_curves.png` - Loss and metric curves
- `results_report.md` - This report

## Recommendations

1. **Next Steps**:
   - Examine per-class performance to identify problem classes
   - Consider additional training (more epochs, different LR schedule)
   - Try data augmentation improvements
   - Compare with DeepLabV3 architecture

2. **Model Fine-tuning**:
   - Adjust learning rate for slower convergence
   - Increase augmentation intensity
   - Use different loss function (Dice, Focal, Combined)
   - Implement learning rate warmup

3. **Evaluation**:
   - Test on held-out test set
   - Generate visualizations of predictions
   - Analyze failure cases
"""
    
    report_path = f'{output_dir}/results_report.md'
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"✓ Results report saved to {report_path}")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='unet', choices=['unet', 'deeplab'])
    parser.add_argument('--checkpoint', default='checkpoints/best_model.pt')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--output_dir', default='results')
    parser.add_argument('--max_batches', type=int, default=None, help='Limit evaluation batches (for quick testing)')
    args = parser.parse_args()
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    Path(args.output_dir).mkdir(exist_ok=True)
    
    # Load model
    print("\n📦 Loading model...")
    model = get_model(args.model, num_classes=21)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif isinstance(checkpoint, dict) and 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()
    print(f"✓ Model loaded from {args.checkpoint}")
    
    # Load data
    print("\n📊 Loading dataset...")
    _, val_loader = get_data_loaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    print(f"✓ Validation set: {len(val_loader)} batches")
    
    # Evaluate
    print(f"\n🔍 Evaluating on validation set...")
    val_metrics = evaluate_on_split(model, val_loader, device, 'validation', args.max_batches)
    
    print("\n📈 Validation Metrics:")
    print(f"  mIoU: {val_metrics['miou']:.4f}")
    print(f"  Pixel Accuracy: {val_metrics['pixel_acc']:.4f}")
    if 'dice' in val_metrics:
        print(f"  Dice: {val_metrics['dice']:.4f}")
    
    # Generate visualizations
    print("\n📊 Generating visualizations...")
    plot_training_curves(args.output_dir)
    
    # Generate report
    print("\n📄 Generating report...")
    report = generate_results_report(val_metrics, args.output_dir)
    
    print(f"\n✅ Evaluation complete!")
    print(f"All results saved to: {args.output_dir}/")
    
    # Save metrics as JSON
    metrics_file = f'{args.output_dir}/final_metrics.json'
    with open(metrics_file, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        metrics_to_save = {k: (v.tolist() if isinstance(v, np.ndarray) else v) 
                          for k, v in val_metrics.items()}
        json.dump(metrics_to_save, f, indent=4)
    print(f"✓ Final metrics saved to {metrics_file}")


if __name__ == '__main__':
    main()
