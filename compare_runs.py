#!/usr/bin/env python3
"""
Compare training runs and generate comparison visualizations.
Use this after all training runs are complete.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def load_training_state():
    """Load current training state with all runs."""
    try:
        with open('training_state.json') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading training_state.json: {e}")
        return None

def compare_runs():
    """Generate comparison visualizations and analysis."""
    data = load_training_state()
    if not data:
        print("❌ Could not load training_state.json")
        return
    
    # Extract run information
    train_history = data.get('train_history', {})
    val_history = data.get('val_history', {})
    
    epochs = len(train_history.get('loss', []))
    
    if epochs == 0:
        print("❌ No training history found")
        return
    
    # Print comparison table
    print("\n" + "="*80)
    print("TRAINING RUN COMPARISON")
    print("="*80)
    
    print(f"\n📊 Total Epochs Trained: {epochs}")
    
    # Final metrics
    final_train_loss = train_history.get('loss', [0])[-1] if train_history.get('loss') else 0
    final_val_loss = val_history.get('loss', [0])[-1] if val_history.get('loss') else 0
    final_miou = val_history.get('miou', [0])[-1] if val_history.get('miou') else 0
    best_miou = data.get('best_miou', 0)
    
    print(f"\n Current Run Metrics:")
    print(f"  • Final Training Loss:  {final_train_loss:.4f}")
    print(f"  • Final Validation Loss: {final_val_loss:.4f}")
    print(f"  • Current mIoU:         {final_miou:.4f} ({final_miou*100:.2f}%)")
    print(f"  • Best mIoU:            {best_miou:.4f} ({best_miou*100:.2f}%)")
    
    # Comparison to baseline
    baseline_miou = 0.0412
    improvement = final_miou - baseline_miou
    improvement_pct = (final_miou / baseline_miou - 1) * 100
    
    print(f"\n📈 Improvement vs Baseline (4.12%):")
    print(f"  • Absolute: +{improvement:.4f} (+{improvement_pct:.1f}%)")
    if improvement > 0:
        print(f"  • Status: ✅ IMPROVED")
    else:
        print(f"  • Status: ⚠️  MONITOR")
    
    # Per-class metrics if available
    per_class = data.get('per_class_iou', {})
    if per_class:
        print(f"\n📋 Per-Class IoU (Current Best):")
        class_names = ['Background', 'Person', 'Aeroplane', 'Bicycle', 'Bird', 'Boat',
                      'Bottle', 'Bus', 'Car', 'Cat', 'Chair', 'Cow', 'Diningtable',
                      'Dog', 'Horse', 'Motorbike', 'Mountain', 'Pottedplant',
                      'Sheep', 'Sofa', 'Train', 'Tvmonitor']
        for i, (class_id, iou_val) in enumerate(sorted(per_class.items(), key=lambda x: float(x[0]))):
            if i < len(class_names):
                class_name = class_names[int(class_id)]
                iou_pct = float(iou_val) * 100
                bar = "█" * int(iou_pct / 5)
                print(f"  {i:2d}. {class_name:15s}: {iou_pct:5.1f}% {bar}")
    
    print("\n" + "="*80)

def generate_comparison_plot():
    """Generate comparison plots if multiple runs are available."""
    data = load_training_state()
    if not data:
        return
    
    train_history = data.get('train_history', {})
    val_history = data.get('val_history', {})
    
    epochs = len(train_history.get('loss', []))
    if epochs == 0:
        return
    
    # Create comparison figure
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('Training Progress & Comparison', fontsize=16, fontweight='bold')
    
    # Training Loss
    if train_history.get('loss'):
        axes[0, 0].plot(train_history['loss'], label='Training', linewidth=2)
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()
    
    # Validation Loss
    if val_history.get('loss'):
        axes[0, 1].plot(val_history['loss'], label='Validation', linewidth=2, color='orange')
        axes[0, 1].set_title('Validation Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend()
    
    # mIoU Progress
    if val_history.get('miou'):
        axes[0, 2].plot(val_history['miou'], label='Current', linewidth=2, color='green')
        axes[0, 2].axhline(y=0.0412, color='red', linestyle='--', label='Baseline', linewidth=2)
        axes[0, 2].set_title('mIoU Progress')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('mIoU')
        axes[0, 2].grid(True, alpha=0.3)
        axes[0, 2].legend()
    
    # Learning curve
    if train_history.get('loss') and val_history.get('loss'):
        axes[1, 0].plot(train_history['loss'], label='Train', linewidth=2)
        axes[1, 0].plot(val_history['loss'], label='Validation', linewidth=2)
        axes[1, 0].set_title('Loss Comparison')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend()
    
    # Pixel Accuracy
    if val_history.get('pixel_acc'):
        axes[1, 1].plot(val_history['pixel_acc'], color='purple', linewidth=2)
        axes[1, 1].set_title('Pixel Accuracy')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Accuracy')
        axes[1, 1].grid(True, alpha=0.3)
    
    # Summary statistics
    axes[1, 2].axis('off')
    stats_text = f"""
    Training Summary
    ━━━━━━━━━━━━━━━━━━━━
    
    Total Epochs: {epochs}
    
    Final mIoU: {val_history.get('miou', [0])[-1]*100:.2f}%
    Best mIoU:  {data.get('best_miou', 0)*100:.2f}%
    
    Train Loss: {train_history.get('loss', [0])[-1]:.4f}
    Val Loss:   {val_history.get('loss', [0])[-1]:.4f}
    
    Baseline:   4.12%
    Improvement: +{(val_history.get('miou', [0])[-1] - 0.0412)*100:.2f}%
    """
    axes[1, 2].text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
                   verticalalignment='center',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('training_comparison.png', dpi=150, bbox_inches='tight')
    print("\n✅ Comparison plot saved to: training_comparison.png")
    plt.close()

if __name__ == '__main__':
    print("\n🔍 Analyzing training run results...\n")
    compare_runs()
    generate_comparison_plot()
    print("\n✅ Comparison complete!")
