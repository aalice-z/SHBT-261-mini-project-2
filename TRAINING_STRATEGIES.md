# 🚀 Advanced Training Strategy - Performance Improvement Guide

## Current Status

You now have **3 training runs active or completed**:

### Run 1: ✅ COMPLETE - Baseline U-Net
- **Loss**: Combined (CE + Dice)
- **Epochs**: 50
- **Result**: mIoU = 4.12%, Pixel Acc = 73.89%
- **Status**: Use as baseline for comparison

### Run 2: ⏳ IN PROGRESS - Focal Loss (Cosine Schedule)
- **Loss**: Focal (better for class imbalance)
- **Learning Rate**: 1e-4 (lower for finer convergence)
- **Scheduler**: Cosine Annealing
- **Epochs**: 50
- **Expected Duration**: 5-6 hours
- **Expected Improvement**: +2-5% mIoU improvement

### Run 3: ⏳ IN PROGRESS - Focal Loss (Polynomial Schedule)
- **Loss**: Focal (focuses on hard examples)
- **Learning Rate**: 1e-4
- **Scheduler**: Polynomial decay
- **Resume From**: best_model.pt (continues training)
- **Total Epochs**: 100 (50 + 50 additional)
- **Expected Duration**: 10-12 hours
- **Expected Improvement**: +3-8% mIoU improvement (longer training)

---

## Why These Strategies Will Help

### Problem: Class Imbalance
- **Background**: 74% of all pixels
- **Person**: 6% of pixels  
- **Other Classes**: <1% each
- **Model Behavior**: Predicts background too often

### Solution 1: Focal Loss
```
Focal Loss = -α(1-p)^γ log(p)
• α = balances class weights
• γ = focusing parameter (increases penalty for hard examples)
• Effect: Penalizes easy background predictions more
```
**Expected Result**: Better object detection, +2-5% mIoU

### Solution 2: Lower Learning Rate
```
Original: 1e-3 (aggressive, good for cold start)
New: 1e-4 (conservative, good for refinement)
```
**Expected Result**: Finer convergence, smoother loss curves

### Solution 3: Longer Training
```
Run 1: 50 epochs (baseline)
Run 3: 100 epochs (double training)
```
**Expected Result**: More stable convergence, potentially +3-8% mIoU

---

## Monitoring Training in Real-Time

### Check Current Progress
```bash
# View last training metrics
tail -100 training_state.json | grep -E '"loss|"miou'

# Count epochs trained so far
python -c "import json; d=json.load(open('training_state.json')); print(f'Trained {len(d[\"train_history\"][\"loss\"])} epochs')"

# Check both metrics and loss
python << 'EOF'
import json
d = json.load(open('training_state.json'))
epochs = len(d['train_history']['loss'])
latest_loss = d['train_history']['loss'][-1]
latest_miou = d['val_history']['miou'][-1] if d['val_history']['miou'] else 0
best_miou = d['best_miou']
print(f"Epochs: {epochs} | Loss: {latest_loss:.4f} | mIoU: {latest_miou:.4f} | Best: {best_miou:.4f}")
EOF
```

### Check Active Training Processes
```bash
ps aux | grep train.py | grep -v grep
```

### Quick Evaluation on Latest Checkpoint
```bash
# Fast evaluation (subset of data)
python evaluate.py --model unet --max-batches 30

# Full evaluation (slower)
python evaluate.py --model unet
```

---

## Expected Performance Timeline

### Hour 2 (Epoch 10)
- Loss: ~2.1-2.2
- mIoU: ~0.02-0.04
- Status: Similar to original baseline

### Hour 4 (Epoch 20)
- Loss: ~2.0-2.1
- mIoU: ~0.05-0.08 (improvement visible)
- Status: Focal loss benefits emerging

### Hour 6 (Epoch 30)
- Loss: ~1.95-2.0
- mIoU: ~0.08-0.12 (significant improvement)
- Status: Model focusing on hard classes

### Hour 8-9 (Epoch 50)
- Loss: ~1.9-1.95
- mIoU: ~0.10-0.15 (3-4x baseline improvement)
- Status: Run 2 completes

### Hour 12 (Epoch 100 for Run 3)
- Loss: ~1.8-1.9
- mIoU: ~0.15-0.25 (potential 6-8x improvement)
- Status: Run 3 completes with maximum training

---

## What If Performance Plateaus?

If mIoU doesn't improve significantly:

### Strategy A: Adjust Loss Function Weight
```bash
# Try different focal parameters (in losses.py)
# Current: alpha=0.25, gamma=2.0
# Try: alpha=0.5, gamma=3.0 (more aggressive)
```

### Strategy B: Use Class Weighting
```bash
# Calculate class weights based on frequency
python << 'EOF'
import numpy as np
# Based on your dataset counts
class_counts = np.array([...])  # from final_metrics.json
class_weight = 1.0 / (class_counts + 1)
class_weight = class_weight / class_weight.sum() * len(class_weight)
print("Class weights:", class_weight)
EOF
```

### Strategy C: More Aggressive Augmentation
```bash
# In voc2007.py, increase augmentation intensity:
# - RandomRotation(≈10-15 degrees)
# - RandomAffine(scale=(0.8, 1.2))
# - RandomPerspective(distortion_scale=0.2)
```

### Strategy D: Different Architecture
```bash
# DeepLabV3 handles small objects better
python train.py --model deeplab --epochs 50 --loss focal \
  --learning-rate 5e-4 --scheduler cosine
```

---

## Comparison Strategy After Training

Once both runs complete:

### Quick Comparison
```bash
python << 'EOF'
import json
import matplotlib.pyplot as plt

# Load all training states
runs = {}
for i, name in enumerate(['original', 'focal_cosine', 'focal_poly'], 1):
    try:
        with open(f'training_state_{i}.json') as f:
            runs[name] = json.load(f)
    except:
        pass

# Plot comparison
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

for name, data in runs.items():
    epochs = range(1, len(data['train_history']['loss']) + 1)
    ax1.plot(epochs, data['train_history']['loss'], label=name)
    ax2.plot(epochs, data['val_history']['miou'], label=name)

ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Training Loss Comparison')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.set_xlabel('Epoch')
ax2.set_ylabel('mIoU')
ax2.set_title('Validation mIoU Comparison')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('comparison_all_runs.png', dpi=150)
print("✓ Comparison plot saved to comparison_all_runs.png")
EOF
```

---

## When to Consider Further Training

Continue training if:
- ✓ mIoU is steadily increasing (not plateauing)
- ✓ Loss continues decreasing
- ✓ Per-class metrics improving on non-background classes

Stop training if:
- ✗ mIoU hasn't improved for 10+ epochs
- ✗ Loss oscillating wildly
- ✗ Validation loss increasing (overfitting)

---

## Quick Reference: Commands

```bash
# Monitor progress
watch -n 10 'ps aux | grep train.py | grep -v grep'

# View current best model performance
python evaluate.py

# Compare checkpoints
python << 'EOF'
import torch
c1 = torch.load('checkpoints/best_model.pt', map_location='mps')
c2 = torch.load('checkpoints/checkpoint_epoch_050.pt', map_location='mps')
print("Run time difference:", c2['timestamp'] - c1['timestamp'] if 'timestamp' in c1 else "N/A")
EOF

# Continue from best model if needed
python train.py --model unet --epochs 50 --loss focal \
  --learning-rate 5e-5 --scheduler polynomial

# Training with other configurations
python train.py --model deeplab --epochs 50 --loss focal --batch-size 2
python train.py --model unet --epochs 50 --loss dice
python train.py --model unet --epochs 50 --loss weighted_ce
```

---

## Expected Results Summary

| Run | Config | Epochs | Expected mIoU | Expected Time |
|-----|--------|--------|---------------|----------------|
| 1 | Combined+Cosine | 50 | 4.1% | Complete ✓ |
| 2 | Focal+Cosine | 50 | 8-12% | 5-6h ⏳ |
| 3 | Focal+Poly | 100 | 12-20% | 10-12h ⏳ |

**Key Insight**: Focal loss + longer training should give **2-5x improvement** over baseline.

---

## Next Steps When Training Completes

1. **Evaluate** both checkpoints
2. **Compare** metrics and loss curves
3. **Select** best performing model
4. **Analyze** per-class performance
5. **Consider** architectural changes if needed
6. **Prepare** final submission

---

**Training started: April 9, 2026 ~1:46 PM**  
**Expected completion: April 9, 2026 ~11:46 PM (focal_cosine run)**  
**Full completion: April 10, 2026 ~1:46 AM (focal_poly run)**

Happy training! 🎉
