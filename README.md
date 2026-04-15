# PASCAL VOC 2007 Semantic Segmentation

A complete deep learning pipeline for semantic segmentation on the PASCAL Visual Object Classes (VOC) 2007 dataset.

## Quick Links

- **Quick Reference**: [docs/SUBMISSION_QUICK_REFERENCE.md](docs/SUBMISSION_QUICK_REFERENCE.md) - Key results and file locations
- **Project Details**: [docs/README.md](docs/README.md) - Full documentation

## Project Structure

```
project2/
├── src/                    # Source code (models, training, utilities)
│   ├── models.py          # U-Net and DeepLabV3 architectures
│   ├── train.py           # Training script with CLI
│   ├── evaluate.py        # Evaluation and metrics computation
│   ├── inference.py       # Inference utilities
│   ├── losses.py          # Loss functions (CE, Dice, Focal, Combined)
│   ├── metrics.py         # Evaluation metrics (IoU, mIoU, Accuracy)
│   ├── training_utils.py  # Training infrastructure
│   ├── voc2007.py         # Dataset loading and preprocessing
│   └── __init__.py        # Package initialization
│
├── docs/                  # Documentation
│   ├── FINAL_REPORT.md    # Main submission report (11 sections)
│   ├── README.md          # Full project documentation
│   ├── PROJECT_STATUS.md  # Component checklist
│   ├── SUBMISSION_CHECKLIST.md  # Completion verification
│   ├── TRAINING_COMPLETE.md     # Training summary
│   └── [other reports]
│
├── scripts/               # Utility scripts
│   ├── run_training.sh    # Shell script to start training
│   ├── check_setup.py     # Verify dependencies
│   ├── check_status.py    # Check project status
│   ├── monitor_training.py # Monitor training progress
│   └── [other utilities]
│
├── results/               # Results and outputs
│   ├── final_metrics.json # Detailed metrics data
│   ├── training_curves.png # Visualization
│   └── results_report.md  # Results summary
│
├── checkpoints/           # Model checkpoints (not tracked)
├── archive/               # Dataset files (not tracked)
└── venv_arm64/            # Virtual environment (not tracked)
```

## Getting Started

### Setup
```bash
# Activate virtual environment
source venv_arm64/bin/activate

# Verify setup
python scripts/check_setup.py
```

### Training
```bash
# Run with default parameters
bash scripts/run_training.sh

# Or with custom parameters
python src/train.py --model unet --epochs 50 --batch-size 4 --learning-rate 0.001
```

### Evaluation
```bash
python src/evaluate.py --model-path checkpoints/best_model.pt
```

## Results

| Metric | Value |
|--------|-------|
| **mIoU (Validation)** | 4.16% |
| **Pixel Accuracy** | 74.19% |
| **Training Loss** | 2.1142 |
| **Background Class IoU** | 74.44% |
| **Person Class IoU** | 12.86% |

## Key Features

- ✅ U-Net and DeepLabV3-Lite architectures
- ✅ Multiple loss functions (CE, Dice, Focal, Combined)
- ✅ Comprehensive evaluation metrics
- ✅ Data augmentation and preprocessing
- ✅ Model checkpointing and scheduler support
- ✅ GPU/MPS device support
- ✅ Complete CLI interface
