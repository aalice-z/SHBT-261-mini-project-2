# Semantic Segmentation on PASCAL VOC 2007

A complete deep learning pipeline for semantic segmentation on the PASCAL Visual Object Classes (VOC) 2007 dataset, featuring U-Net and DeepLabV3-Lite architectures.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Training](#training)
- [Evaluation and Inference](#evaluation-and-inference)
- [Project Structure](#project-structure)
- [Results](#results)
- [Configuration Options](#configuration-options)
- [References](#references)

## Overview

This project implements a complete semantic segmentation framework for the PASCAL VOC 2007 dataset. Semantic segmentation assigns a class label to every pixel in an image, enabling dense pixel-level predictions. The pipeline includes:

- **Data Loading & Preprocessing**: VOC 2007 dataset with configurable transforms and augmentation
- **Multiple Model Architectures**: U-Net (encoder-decoder with skip connections) and DeepLabV3-Lite (atrous convolutions)
- **Flexible Loss Functions**: Cross-entropy, Dice, Focal, Combined, and Weighted Cross-entropy losses
- **Comprehensive Metrics**: IoU, mIoU, Dice coefficient, pixel accuracy, per-class metrics
- **Training Infrastructure**: Checkpoint management, learning rate scheduling, gradient clipping
- **Inference Utilities**: Batch prediction, visualization, and evaluation on test sets

## Dataset

### PASCAL VOC 2007

The PASCAL VOC 2007 dataset contains:
- **Training set**: ~5,011 images
- **Validation set**: ~5,823 images
- **Image resolution**: 500×375 (resized to 512×512 for training)
- **Classes**: 21 (20 object classes + 1 background)

### Class List

| ID | Class | ID | Class | ID | Class |
|---|---|---|---|---|---|
| 0 | Background | 7 | Car | 14 | Motorbike |
| 1 | Aeroplane | 8 | Cat | 15 | Person |
| 2 | Bicycle | 9 | Chair | 16 | Pottedplant |
| 3 | Bird | 10 | Cow | 17 | Sheep |
| 4 | Boat | 11 | Diningtable | 18 | Sofa |
| 5 | Bottle | 12 | Dog | 19 | Train |
| 6 | Bus | 13 | Horse | 20 | Tvmonitor |

### Dataset Features

- **Augmentation**: Random horizontal flip (p=0.5), vertical flip (p=0.1), color jitter, and resizing
- **Normalization**: ImageNet statistics (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
- **Ignore Index**: Boundary pixels marked as 255 are ignored during loss computation
- **Batch Size**: Default 4 with 2 worker processes and pinned memory for efficient loading

## Architecture

### U-Net

A classic encoder-decoder architecture with skip connections for preserving spatial information.

**Architecture Details:**
- **Encoder Path**: 4 downsampling blocks (max pooling + double convolutions)
- **Bottleneck**: Deepest layer at 1/16 resolution
- **Decoder Path**: 4 upsampling blocks with concatenated skip connections
- **Output**: Logit maps for 21 classes

**Key Components:**
```
DoubleConv: Conv2d → BatchNorm2d → ReLU → Conv2d → BatchNorm2d → ReLU
DownBlock: MaxPool2d → DoubleConv
UpBlock: Bilinear Upsample → Concatenate → DoubleConv
```

**Parameters**: ~28M (can be reduced by scaling channel widths)

### DeepLabV3-Lite

A lighter alternative using atrous (dilated) convolutions for multi-scale feature extraction.

**Architecture Details:**
- **Backbone**: ResNet50 pretrained on ImageNet
- **ASPP Module**: Atrous Spatial Pyramid Pooling with rates [6, 12, 18]
- **Decoder**: 2× upsampling with skip connections from backbone
- **Output**: Logit maps for 21 classes

**Key Components:**
```
ASPP: Parallel atrous convolutions (dilation rates 1, 6, 12, 18) + image pooling
AtrousConv: Conv2d with dilation rate
DecoderModule: Upsampling + feature refinement
```

**Parameters**: ~39M (pretrained backbone reduces training time)

## Installation

### Requirements

- Python 3.7+
- PyTorch 1.9+
- torchvision 0.10+
- NumPy
- Matplotlib

### Setup

1. Clone or download this repository:
```bash
cd /path/to/project2
```

2. Install dependencies:
```bash
pip install torch torchvision
pip install numpy matplotlib pillow
```

3. Verify installation by running the data loader:
```bash
python voc2007.py
```

This will display sample images with their segmentation masks and compute per-class statistics.

## Quick Start

### 1. Prepare Data

Download PASCAL VOC 2007 dataset and ensure the directory structure is:
```
archive/
  VOCtrainval_06-Nov-2007/
    VOCdevkit/VOC2007/
      Annotations/      # XML files
      JPEGImages/       # RGB images
      ImageSets/        # Train/val splits
      SegmentationClass/ # Segmentation masks
```

### 2. Train U-Net Model

```bash
python train.py \
  --model unet \
  --epochs 50 \
  --batch-size 4 \
  --learning-rate 0.001 \
  --scheduler cosine \
  --optimizer adam \
  --loss combined
```

### 3. Evaluate

After training completes, the best model is saved in `./checkpoints/`. To make predictions:

```python
from inference import SegmentationPredictor, MaskVisualizer
from voc2007 import VOC_CLASSES

# Load predictor
predictor = SegmentationPredictor(
    model_path="./checkpoints/best_model.pth",
    model_class="unet",
    num_classes=21,
    device="cuda"
)

# Load and predict on an image
from PIL import Image
image = Image.open("path/to/image.jpg")
prediction = predictor.predict(image)

# Visualize
mask_rgb = MaskVisualizer.colorize_mask(prediction)
MaskVisualizer.plot_predictions([np.array(image)], [prediction])
```

## Training

### Command-Line Arguments

The training script supports extensive configuration via command-line arguments:

#### Model Selection
- `--model {unet|deeplab}`: Model architecture (default: unet)
- `--pretrained`: Load ImageNet pretrained weights (for DeepLab)

#### Data Configuration
- `--dataset-root`: Path to PASCAL VOC directory (default: ./archive/)
- `--batch-size`: Training batch size (default: 4)
- `--num-workers`: Data loader workers (default: 2)
- `--image-size`: Input image size (default: 512)

#### Training Parameters
- `--epochs`: Number of training epochs (default: 50)
- `--learning-rate`: Initial learning rate (default: 0.001)
- `--weight-decay`: L2 regularization (default: 1e-5)

#### Optimizer & Scheduler
- `--optimizer {adam|adamw|sgd}`: Optimizer choice (default: adam)
- `--scheduler {step|cosine|polynomial|none}`: LR scheduler (default: cosine)

#### Loss Function
- `--loss {ce|dice|focal|combined|weighted_ce}`: Loss function (default: combined)
  - **ce**: Cross-entropy loss with class weighting
  - **dice**: F1/Dice loss (good for imbalanced data)
  - **focal**: Focal loss (handles hard negatives)
  - **combined**: Weighted sum of CE and Dice (recommended)
  - **weighted_ce**: Auto-computed class-weighted CE

#### Checkpointing
- `--checkpoint-dir`: Directory for saving checkpoints (default: ./checkpoints/)
- `--log-interval`: Logging interval in batches (default: 5)

### Example Training Runs

**Basic U-Net training:**
```bash
python train.py --model unet --epochs 50
```

**DeepLab with pretrained backbone:**
```bash
python train.py --model deeplab --pretrained --epochs 30
```

**Custom configuration:**
```bash
python train.py \
  --model unet \
  --batch-size 8 \
  --learning-rate 0.0005 \
  --optimizer adamw \
  --scheduler polynomial \
  --loss combined \
  --epochs 100
```

## Evaluation and Inference

### Using the Inference Module

```python
import torch
from inference import SegmentationPredictor, MaskVisualizer, evaluate_model
from voc2007 import get_data_loaders, VOC_CLASSES

# 1. Create predictor from checkpoint
predictor = SegmentationPredictor(
    model_path="./checkpoints/best_model.pth",
    model_class=YourModelClass,  # Import from models.py
    num_classes=21,
    device="cuda"
)

# 2. Predict on single image
image = # Load image
prediction = predictor.predict(image)

# 3. Predict on batch
train_loader, val_loader = get_data_loaders("./archive/", batch_size=4)
images_batch = next(iter(val_loader))[0]
predictions_batch = predictor.predict_batch(images_batch.numpy())

# 4. Visualize predictions
mask_rgb = MaskVisualizer.colorize_mask(prediction)
fig = MaskVisualizer.plot_predictions([image], [prediction], 
                                      class_names=VOC_CLASSES)

# 5. Evaluate on test set
model = YourModel(num_classes=21)
metrics = evaluate_model(model, val_loader, num_classes=21, 
                        class_names=VOC_CLASSES)
print(f"mIoU: {metrics['miou']:.4f}")
print(f"Pixel Accuracy: {metrics['pixel_accuracy']:.4f}")
```

### Metrics Computed

- **IoU (Intersection over Union)**: Standard metric for segmentation
  - Formula: IoU = TP / (TP + FP + FN)
- **mIoU**: Mean IoU across all classes
- **Dice Coefficient**: Similar to IoU, emphasizes overlap
  - Formula: Dice = 2*TP / (2*TP + FP + FN)
- **Pixel Accuracy**: Percentage of correctly classified pixels
- **Per-class Accuracy**: Accuracy for each individual class

## Project Structure

```
project2/
├── voc2007.py              # Data loading and preprocessing
├── models.py               # U-Net and DeepLab implementations
├── losses.py               # Loss function definitions
├── metrics.py              # Evaluation metrics and trackers
├── training_utils.py       # Training infrastructure (Trainer, checkpointing)
├── train.py               # Main training script with CLI
├── inference.py           # Inference and evaluation utilities
├── README.md              # This file
├── checkpoints/           # Saved model checkpoints (created during training)
└── archive/               # PASCAL VOC dataset
    ├── VOCtrainval_06-Nov-2007/
    └── VOCtest_06-Nov-2007/
```

### File Descriptions

| File | Purpose | Key Components |
|---|---|---|
| `voc2007.py` | Dataset loading & preprocessing | VOC class definitions, transforms, data loaders, visualization |
| `models.py` | Neural network architectures | UNet, DeepLabV3Lite, ASPP module, factory function |
| `losses.py` | Loss functions | CE, Dice, Focal, Combined losses, factory function |
| `metrics.py` | Evaluation metrics | IoU, Dice, accuracy, MetricTracker class |
| `training_utils.py` | Training infrastructure | Trainer class, CheckpointManager, optimizers, schedulers |
| `train.py` | Training entry point | CLI arguments, training orchestration |
| `inference.py` | Inference & evaluation | SegmentationPredictor, evaluation functions, visualization |

## Results

### Expected Performance on PASCAL VOC 2007

| Model | mIoU | Pixel Accuracy | Training Time |
|---|---|---|---|
| U-Net (baseline) | 55-60% | 85-88% | ~4-6 hours (GPU) |
| U-Net (tuned) | 60-65% | 88-90% | ~8-10 hours (GPU) |
| DeepLab-V3 Lite | 62-67% | 89-91% | ~3-4 hours (GPU) |

### Performance Tips

1. **Class Imbalance**: Use `weighted_ce` or `combined` loss to handle class imbalance
2. **Augmentation**: Increase augmentation (flip, rotation) for better generalization
3. **Learning Rate**: Start with 0.001 and decay with cosine annealing
4. **Iterations**: Train for 50+ epochs for convergence
5. **Batch Size**: Larger batches (8+) improve stability but require more memory
6. **Pretrained Backbone**: DeepLab with pretrained ResNet50 converges faster

## Configuration Options

### Loss Functions

```python
from losses import get_loss

# Cross-entropy with automatic class weighting
loss = get_loss('weighted_ce', num_classes=21, ignore_index=255)

# Dice loss (good for imbalanced classes)
loss = get_loss('dice', num_classes=21)

# Focal loss (emphasizes hard examples)
loss = get_loss('focal', num_classes=21, alpha=0.25, gamma=2.0)

# Combined CE + Dice (recommended)
loss = get_loss('combined', num_classes=21, ce_weight=1.0, dice_weight=1.0)
```

### Optimizers

```python
from training_utils import create_optimizer

# Adam optimizer
optimizer = create_optimizer(model, 'adam', lr=0.001, weight_decay=1e-5)

# AdamW with decoupled weight decay
optimizer = create_optimizer(model, 'adamw', lr=0.001, weight_decay=1e-2)

# SGD with momentum
optimizer = create_optimizer(model, 'sgd', lr=0.01, weight_decay=1e-4)
```

### Learning Rate Schedules

```python
from training_utils import LearningRateScheduler

scheduler = LearningRateScheduler.get_scheduler(
    optimizer,
    'cosine',        # Options: 'step', 'cosine', 'polynomial', 'none'
    T_max=50         # For cosine schedule, max epochs
)
```

## References

1. **U-Net**: Ronneberger et al., "U-Net: Convolutional Networks for Biomedical Image Segmentation" (2015)
   - https://arxiv.org/abs/1505.04597

2. **DeepLabV3**: Chen et al., "Rethinking Atrous Convolution for Semantic Image Segmentation" (2017)
   - https://arxiv.org/abs/1706.05587

3. **PASCAL VOC Dataset**: Everingham et al., "The PASCAL Visual Object Classes Challenge"
   - http://www.pascalvoc.org/

4. **Focal Loss**: Lin et al., "Focal Loss for Dense Object Detection" (2017)
   - https://arxiv.org/abs/1708.02002

5. **Dice Loss**: Milletari et al., "V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation" (2016)
   - https://arxiv.org/abs/1606.06650

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{voc2007_segmentation,
  title={Semantic Segmentation on PASCAL VOC 2007},
  author={Your Name},
  year={2024},
  howpublished={\url{https://github.com/yourusername/project2}}
}
```

## License

This project is provided for educational purposes. The PASCAL VOC dataset has its own license terms.

## Troubleshooting

### GPU Memory Issues
- Reduce `--batch-size` (try 2 or 1)
- Reduce `--image-size` (try 256 instead of 512)
- Use gradient accumulation (modify training loop)

### Slow Training
- Increase `--num-workers` for data loading
- Use `--pretrained` flag for DeepLab to start with trained weights
- Try smaller model or reduce `--image-size`

### Poor Accuracy
- Train for more epochs (`--epochs 100`)
- Use `--scheduler cosine` with longer training
- Try `--loss combined` for better convergence
- Use data augmentation (already enabled in voc2007.py)

## Contact & Support

For issues or questions, refer to the PyTorch documentation:
- PyTorch: https://pytorch.org/docs/stable/index.html
- Torchvision: https://pytorch.org/vision/stable/
