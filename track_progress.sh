#!/bin/bash
# Track progress of all training runs every 5 minutes

set -e

OUTPUT_FILE="training_progress.log"
INTERVAL=300  # 5 minutes

echo "Starting training progress monitor..."
echo "Logging to: $OUTPUT_FILE"
echo "Update interval: $INTERVAL seconds (5 minutes)"
echo ""

iteration=0
while true; do
    iteration=$((iteration + 1))
    timestamp=$(date "+%Y-%m-%d %H:%M:%S")
    
    echo "[$timestamp] Update #$iteration" >> "$OUTPUT_FILE"
    echo "================================================" >> "$OUTPUT_FILE"
    
    # Check active processes
    process_count=$(ps aux | grep "train.py" | grep -v grep | wc -l)
    echo "Active training processes: $process_count" >> "$OUTPUT_FILE"
    
    if [ -f "training_state.json" ]; then
        python << 'EOF' >> "$OUTPUT_FILE"
import json
import datetime
d = json.load(open('training_state.json'))
epochs = len(d['train_history']['loss'])
loss = d['train_history']['loss'][-1]
val_loss = d['val_history']['loss'][-1] if d['val_history']['loss'] else 0
miou = d['val_history']['miou'][-1] if d['val_history']['miou'] else 0
best_miou = d['best_miou']
print(f"Epochs trained: {epochs}")
print(f"Training loss: {loss:.4f}")
print(f"Validation loss: {val_loss:.4f}")
print(f"Current mIoU: {miou:.4f}")
print(f"Best mIoU: {best_miou:.4f}")
print(f"Improvement: {abs(miou - 0.0412):.4f} (+{(miou/0.0412 - 1)*100:.1f}% vs baseline)")
EOF
    fi
    
    echo "" >> "$OUTPUT_FILE"
    echo "================================================" >> "$OUTPUT_FILE"
    echo "" >> "$OUTPUT_FILE"
    
    # Show last update
    echo "✓ Progress logged"
    tail -15 "$OUTPUT_FILE"
    echo ""
    echo "Next update in $INTERVAL seconds... (Press Ctrl+C to stop)"
    
    sleep "$INTERVAL"
done
