#!/bin/bash
# Training script for PASCAL VOC 2007 Semantic Segmentation

# Activate virtual environment
source venv_arm64/bin/activate

#  Default values
MODEL="unet"
EPOCHS=50
BATCH_SIZE=4
LEARNING_RATE=0.001
SCHEDULER="cosine"
LOSS="combined"
OPTIMIZER="adam"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --lr|--learning-rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --scheduler)
            SCHEDULER="$2"
            shift 2
            ;;
        --loss)
            LOSS="$2"
            shift 2
            ;;
        --optimizer)
            OPTIMIZER="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Run training
echo "Starting training..."
echo "Model: $MODEL"
echo "Epochs: $EPOCHS"
echo "Batch Size: $BATCH_SIZE"
echo "Learning Rate: $LEARNING_RATE"
echo "Scheduler: $SCHEDULER"
echo "Loss: $LOSS"
echo "Optimizer: $OPTIMIZER"
echo "============================================================"

cd "$(dirname "$0")/.."
python src/train.py \
    --model "$MODEL" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --learning-rate "$LEARNING_RATE" \
    --scheduler "$SCHEDULER" \
    --loss "$LOSS" \
    --optimizer "$OPTIMIZER" \
    --dataset-root archive/

deactivate
