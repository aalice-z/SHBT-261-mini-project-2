#!/usr/bin/env python3
"""
Verification script to check that all project dependencies and modules are working.
"""

import sys

print("=" * 60)
print("Project Dependencies Verification")
print("=" * 60)

# Check Python version
print(f"\n✓ Python {sys.version}")

# Check core dependencies
try:
    import torch
    print(f"✓ PyTorch {torch.__version__}")
except ImportError as e:
    print(f"✗ PyTorch: {e}")
    sys.exit(1)

try:
    import torchvision
    print(f"✓ torchvision {torchvision.__version__}")
except ImportError as e:
    print(f"✗ torchvision: {e}")
    sys.exit(1)

try:
    import numpy as np
    print(f"✓ NumPy {np.__version__}")
except ImportError as e:
    print(f"✗ NumPy: {e}")
    sys.exit(1)

try:
    import matplotlib
    print(f"✓ Matplotlib {matplotlib.__version__}")
except ImportError as e:
    print(f"✗ Matplotlib: {e}")
    sys.exit(1)

try:
    from PIL import Image
    print(f"✓ Pillow (PIL) installed")
except ImportError as e:
    print(f"✗ Pillow: {e}")
    sys.exit(1)

# Check project modules
print("\n" + "=" * 60)
print("Project Modules Verification")
print("=" * 60)

try:
    from models import get_model, UNet, DeepLabV3Lite
    print("✓ models.py - UNet & DeepLabV3Lite loaded")
except ImportError as e:
    print(f"✗ models.py: {e}")
    sys.exit(1)

try:
    from losses import get_loss
    print("✓ losses.py - Loss functions loaded")
except ImportError as e:
    print(f"✗ losses.py: {e}")
    sys.exit(1)

try:
    from metrics import MetricTracker, SegmentationMetrics
    print("✓ metrics.py - Metrics loaded")
except ImportError as e:
    print(f"✗ metrics.py: {e}")
    sys.exit(1)

try:
    from training_utils import Trainer, CheckpointManager, LearningRateScheduler
    print("✓ training_utils.py - Training utilities loaded")
except ImportError as e:
    print(f"✗ training_utils.py: {e}")
    sys.exit(1)

try:
    from voc2007 import VOC_CLASSES, get_data_loaders
    print("✓ voc2007.py - VOC dataset utilities loaded")
except ImportError as e:
    print(f"✗ voc2007.py: {e}")
    sys.exit(1)

try:
    from inference import SegmentationPredictor, MaskVisualizer
    print("✓ inference.py - Inference utilities loaded")
except ImportError as e:
    print(f"✗ inference.py: {e}")
    sys.exit(1)

# Test model instantiation
print("\n" + "=" * 60)
print("Model Instantiation Test")
print("=" * 60)

try:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Test U-Net with batch size 2 (for batch norm compatibility)
    unet_model = get_model("unet", num_classes=21)
    x = torch.randn(2, 3, 512, 512)
    with torch.no_grad():
        out = unet_model(x)
    print(f"✓ U-Net: Input {x.shape} → Output {out.shape}")
    
    # Test DeepLab with batch size 2
    deeplab_model = get_model("deeplab", num_classes=21, pretrained=False)
    with torch.no_grad():
        out = deeplab_model(x)
    print(f"✓ DeepLabV3-Lite: Input {x.shape} → Output {out.shape}")
    
except Exception as e:
    print(f"✗ Model test failed: {e}")
    sys.exit(1)

# Test loss functions
print("\n" + "=" * 60)
print("Loss Function Test")
print("=" * 60)

try:
    loss_fns = ["ce", "dice", "focal", "combined", "weighted_ce"]
    for loss_name in loss_fns:
        loss_fn = get_loss(loss_name, num_classes=21)
        print(f"✓ {loss_name.upper()} loss initialized")
except Exception as e:
    print(f"✗ Loss function test failed: {e}")
    sys.exit(1)

# Summary
print("\n" + "=" * 60)
print("✅ All checks passed! Project is ready to train.")
print("=" * 60)
print("\nQuick start commands:")
print("  1. Activate venv: source venv_arm64/bin/activate")
print("  2. Run training: python train.py --model unet --epochs 50")
print("  3. Or use script: ./run_training.sh --model unet --epochs 50")
print("=" * 60)
