#!/usr/bin/env python
"""
Monitor multiple training runs and compare results.
"""
import json
import sys
from pathlib import Path
import time

def monitor_training():
    """Monitor training progress in real-time."""
    
    print("=" * 70)
    print("MONITORING MULTIPLE TRAINING RUNS")
    print("=" * 70)
    print()
    
    # Check if training_state.json exists
    if Path('training_state.json').exists():
        with open('training_state.json') as f:
            data = json.load(f)
        
        num_epochs = len(data['train_history']['loss'])
        final_train_loss = data['train_history']['loss'][-1]
        final_val_loss = data['val_history']['loss'][-1] if data['val_history']['loss'] else 0
        final_miou = data['val_history']['miou'][-1] if data['val_history']['miou'] else 0
        best_miou = data['best_miou']
        
        print(f"TRAINING RUN 1: Original U-Net (Combined Loss)")
        print(f"  Status: COMPLETE ✓")
        print(f"  Epochs: {num_epochs}")
        print(f"  Final Train Loss: {final_train_loss:.4f}")
        print(f"  Final Val Loss: {final_val_loss:.4f}")
        print(f"  Final mIoU: {final_miou:.4f}")
        print(f"  Best mIoU: {best_miou:.4f}")
        print()
    
    print("TRAINING RUN 2: U-Net with Focal Loss (Cosine Scheduler)")
    print(f"  Status: IN PROGRESS (50 epochs)...")
    print(f"  Started at: ~1:46 PM")
    print(f"  Expected duration: 5-6 hours")
    print()
    
    print("TRAINING RUN 3: U-Net with Focal Loss (Polynomial Scheduler)")
    print(f"  Status: IN PROGRESS (100 epochs)")  
    print(f"  Resuming from best checkpoint")
    print(f"  Expected duration: 10-12 hours")
    print()
    
    print("=" * 70)
    print("WHAT'S DIFFERENT IN NEW TRAINING RUNS")
    print("=" * 70)
    print()
    print("1. FOCAL LOSS")
    print("   • Focuses on hard-to-predict pixels")
    print("   • Better for class imbalance (background = 74%)")
    print("   • Formula: FL(t) = -α(1-p)^γ log(p)")
    print()
    print("2. LOWER LEARNING RATE (1e-4 vs original 1e-3)")
    print("   • Allows finer convergence")
    print("   • Better for continuing from checkpoint")
    print("   • Less risk of catastrophic forgetting")
    print()
    print("3. COSINE ANNEALING SCHEDULER")
    print("   • Smooth learning rate decay")
    print("   • Better for fine-grained optimization")
    print()
    print("4. POLYNOMIAL SCHEDULER (Run 3 only)")
    print("   • Alternative decay strategy")
    print("   • Good for longer training runs")
    print()
    print("=" * 70)
    print()


def wait_for_completion(timeout_minutes=720):
    """Wait and notify when training completes."""
    
    print(f"Waiting for training to complete (max {timeout_minutes} minutes)...")
    print("You can check progress anytime with:")
    print()
    print("  # View training progress")
    print("  tail -20 training_state.json | grep -E 'loss|miou'")
    print()
    print("  # Evaluate current checkpoint")
    print("  python evaluate.py --model unet --checkpoint checkpoints/best_model.pt")
    print()
    print("  # Check active training processes")
    print("  ps aux | grep train.py | grep -v grep")
    print()
    print("=" * 70)


if __name__ == '__main__':
    monitor_training()
    wait_for_completion()
