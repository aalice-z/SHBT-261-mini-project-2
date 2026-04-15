#!/usr/bin/env python3
"""Quick status check for training."""
import json
import sys

try:
    with open('training_state.json') as f:
        data = json.load(f)
    
    epochs = len(data['train_history']['loss'])
    loss = data['train_history']['loss'][-1]
    val_loss = data['val_history']['loss'][-1] if data['val_history']['loss'] else 0
    miou = data['val_history']['miou'][-1] if data['val_history']['miou'] else 0
    best_miou = data['best_miou']
    
    print(f"\n{'='*60}")
    print(f"CURRENT TRAINING STATUS")
    print(f"{'='*60}")
    print(f"Epochs Trained:       {epochs}/100 ({epochs}%)")
    print(f"Training Loss:        {loss:.4f}")
    print(f"Validation Loss:      {val_loss:.4f}")
    print(f"Current mIoU:         {miou:.4f} ({miou*100:.2f}%)")
    print(f"Best mIoU:            {best_miou:.4f} ({best_miou*100:.2f}%)")
    print(f"Baseline mIoU:        0.0412 (4.12%)")
    improvement = miou - 0.0412
    improvement_pct = (miou / 0.0412 - 1) * 100 if miou > 0 else 0
    print(f"\n📈 Improvement:        {improvement:+.4f} ({improvement_pct:+.1f}%)")
    print(f"{'='*60}\n")
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)
