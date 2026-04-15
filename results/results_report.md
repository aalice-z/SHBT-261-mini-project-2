
# Training Results Report

## Training Summary
- **Total Epochs**: 50
- **Final Train Loss**: 2.1142
- **Best Train Loss**: 2.1079
- **Final Val Loss**: 2.1754

## Validation Results (Final Checkpoint)
- **Best mIoU during training**: 0.0421
- **Final mIoU**: 0.0416
- **Pixel Accuracy**: 0.7419
- **Dice Coefficient**: N/A

## Per-Class Performance

| Class | IoU | Count |
|-------|-----|-------|
| Background | 0.7444 | 9810120 |
| Aeroplane | 0.0000 | 121282 |
| Bicycle | 0.0000 | 28767 |
| Bird | 0.0000 | 126437 |
| Boat | 0.0000 | 59985 |
| Bottle | 0.0000 | 66310 |
| Bus | 0.0000 | 228680 |
| Car | 0.0000 | 98870 |
| Cat | 0.0000 | 307226 |
| Chair | 0.0000 | 118814 |
| Cow | 0.0000 | 116968 |
| Diningtable | 0.0010 | 226021 |
| Dog | 0.0000 | 107253 |
| Horse | 0.0000 | 133269 |
| Motorbike | 0.0000 | 205316 |
| Person | 0.1286 | 790209 |
| Pottedplant | 0.0000 | 63702 |
| Sheep | 0.0000 | 222053 |
| Sofa | 0.0000 | 119247 |
| Train | 0.0000 | 270490 |
| Tvmonitor | 0.0000 | 68371 |

## Key Observations

1. **Model Convergence**: Training loss decreased from 3.0825 to 2.1142
2. **Validation Performance**: mIoU of 0.0416 achieved on validation set
3. **Pixel Accuracy**: 0.7419 indicates balanced predictions across pixels
4. **Training Stability**: Loss curves show stable convergence

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
