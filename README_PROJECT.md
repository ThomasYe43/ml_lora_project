# LoRA vs Baseline ResNet34 Comparison Project

## Project Overview
This project compares baseline ResNet34 fine-tuning against LoRA (Low-Rank Adaptation) on CIFAR-100 and Food-101 datasets, tracking accuracy, training time, and parameter efficiency.

## Project Structure

### Python Modules

1. **`models.py`** - Model Definitions
   - `Trained_Resnet34`: Baseline ResNet34 with frozen backbone
   - `LoRAResnet34`: ResNet34 with LoRA applied to layer3 and layer4
   - Both support configurable number of classes
   - LoRA config: rank=8, alpha=16

2. **`data_loader.py`** - Data Loading Utilities
   - `get_transforms()`: Returns base and training transforms
   - `get_dataloaders()`: Loads Food-101 dataset with train/val/test splits
   - `get_cifar100_dataloaders()`: Loads CIFAR-100 dataset
   - Configuration: BATCH_SIZE=64, IMG_SIZE=224, NUM_WORKERS=8

3. **`train.py`** - Training Utilities
   - `accuracy()`: Compute model accuracy
   - `compute_loss()`: Compute average loss
   - `train_model()`: Train with comprehensive metrics tracking
   - `compare_models()`: Train and compare two models with identical settings

### Notebook Structure (`Untitled1 (4).ipynb`)

**Cell 0**: Imports
- Import PyTorch and matplotlib
- Import custom modules (models, data_loader, train)

**Cell 1**: Device Check
- Verify CUDA availability and GPU info

**Cell 2**: CIFAR-100 Quick Validation Experiment (Markdown header)

**Cell 3**: Load CIFAR-100 Data
- Quick dataset for validation (100 classes, ~10 min training)

**Cell 4**: Create CIFAR-100 Models
- Baseline and LoRA models with 100 classes

**Cell 5**: Run CIFAR-100 Comparison
- 10 epochs, batch_size=128
- Quick validation of implementation

**Cell 6**: Food-101 Comprehensive Comparison (Markdown header)

**Cell 7**: Load Food-101 Data
- Full dataset (101 classes, ~30-60 min training)

**Cell 8**: Create Food-101 Models
- Baseline and LoRA models with 101 classes

**Cell 9**: Run Food-101 Comparison
- 20 epochs, batch_size=64
- Comprehensive comparison

**Cell 10**: Comparison Visualizations (Markdown header)

**Cell 11**: Side-by-side Plots
- Accuracy curves (train/val)
- Loss curves
- Training time per epoch
- Trainable parameters bar chart

**Cell 12**: Summary Comparison Table
- Parameters, accuracy, training time, efficiency metrics

## How to Run

1. **Quick Validation (CIFAR-100)**:
   - Run cells 0-5
   - Takes ~15-20 minutes total
   - Validates that LoRA implementation works

2. **Comprehensive Comparison (Food-101)**:
   - Run cells 0-2, then 6-12
   - Takes ~2-3 hours total
   - Full comparison with visualizations

3. **Complete Pipeline**:
   - Run all cells in order
   - CIFAR-100 first (quick check), then Food-101 (comprehensive)

## Expected Results

### CIFAR-100 (Quick Validation)
- Baseline: ~70-75% test accuracy
- LoRA: ~68-73% test accuracy
- Parameter reduction: ~99%
- Training speedup: ~1.2-1.3x

### Food-101 (Comprehensive)
- Baseline: ~70-75% test accuracy
- LoRA: ~68-73% test accuracy (90-95% of baseline)
- Parameter reduction: ~99%
- Training speedup: ~1.2-1.3x
- Memory reduction: ~15-25%

## Key Features

1. **Modular Design**: Clean separation of models, data loading, and training
2. **Metrics Tracking**: Comprehensive tracking of accuracy, loss, time, and parameters
3. **Visualization**: Side-by-side plots and summary tables
4. **Reproducibility**: Fixed random seeds, consistent hyperparameters
5. **Efficiency**: AMP training, gradient clipping, data loader optimizations

## Configuration

Default hyperparameters (can be modified in function calls):
- Learning rate: 0.0003 (Food-101), 0.001 (CIFAR-100)
- Batch size: 64 (Food-101), 128 (CIFAR-100)
- Optimizer: Adam
- Scheduler: CosineAnnealingLR
- LoRA rank: 8
- LoRA alpha: 16

## Notes

- Models are saved as `baseline_best.pth` and `lora_best.pth`
- Training uses mixed precision (AMP) on GPU for speed
- Data augmentation only applied to training set
- Validation uses clean data without augmentation

