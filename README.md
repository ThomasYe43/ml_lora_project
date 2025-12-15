# CNN with LoRA - ResNet34 Fine-tuning

Fine-tuning ResNet34 on CIFAR-100 and Food-101 datasets, comparing baseline training vs LoRA (Low-Rank Adaptation).

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/IamMichael23/CNN-with-LoRA.git
cd CNN-with-LoRA
```

### 2. Install Dependencies

```bash
# Create a virtual environment (recommended)
python -m venv .venv

# Activate virtual environment
# On Windows:
.venv\Scripts\activate
# On Linux/Mac:
source .venv/bin/activate

# Install required packages
pip install -r requirements.txt
```

### 3. Run Training

Open the Jupyter notebook:

```bash
jupyter notebook main.ipynb
```

Or run training directly:

```bash
python train.py
```

## 📋 Requirements

- Python 3.10+
- CUDA-capable GPU (recommended)
- ~10GB disk space for datasets

## 🎯 What This Does

- **Baseline Model**: Fine-tunes ResNet34 / ViT-B/16 with frozen backbone
- **LoRA Model**: Uses Low-Rank Adaptation for parameter-efficient fine-tuning
- **Datasets**: CIFAR-100 (quick test) and Food-101 (full evaluation)
- **Comparison**: Tracks accuracy, training time, and parameter efficiency

## ⚙️ Key Features

- ✅ **224x224 image size** (proper for ResNet34 / ViT-B/16)
- ✅ **Mixed precision training** (faster on GPU)
- ✅ **Data augmentation** with TrivialAugmentWide
- ✅ **Learning rate scheduling** with CosineAnnealingLR
- ✅ **Automatic model checkpointing** (saves best model)

## 📊 Expected Results

### CIFAR-100 (100 classes)
- **Training time**: ~15-30 minutes (15 epochs) (depends on device)
- **Expected accuracy**: 65-75%

### Food-101 (101 food classes)
- **Training time**: ~2-3 hours (20 epochs) (depends on device)
- **Expected accuracy**: 70-80%

## 📁 Project Structure

```
.
├── main.ipynb           # Main training notebook
├── models.py            # Model definitions (Baseline & LoRA)
├── transformer.py       # ViT-B/16 (Baseline & LoRA)
├── data_loader.py       # Dataset loading utilities
├── train.py             # Training functions
├── requirements.txt     # Python dependencies
└── README.md            # This file
```

## 🔧 Configuration

Key hyperparameters in notebook (Cell 3):

```python
BATCH_SIZE = 256         # Adjust based on GPU memory
LEARNING_RATE = 0.0003   # Lower for stable fine-tuning
NUM_EPOCHS = 10          # More epochs = better accuracy (usually below 20)
```

## ⚠️ Important Notes

### Windows Users
If you get data loader errors, set `num_workers=0`:

```python
train_loader, val_loader, test_loader = get_cifar100_dataloaders(
    batch_size=BATCH_SIZE,
    num_workers=0  # Fix for Windows
)
```

### GPU Memory Issues
If you run out of GPU memory:
- Reduce `BATCH_SIZE` (try 16 or 8)
- Close other GPU applications
- Restart kernel to clear memory

### Dataset Download
Datasets download automatically on first run:
- **CIFAR-100**: ~170MB
- **Food-101**: ~5GB

## 📈 Monitoring Training

The notebook displays:
- Training/validation loss curves
- Training/validation accuracy curves
- Iteration-by-iteration progress
- Best model checkpointing

Models are saved as:
- `baseline_best.pth` - Best baseline model
- `lora_best.pth` - Best LoRA model

## 🤝 Contributing

Feel free to open issues or submit pull requests!

## 📝 License

MIT License - feel free to use for your projects!

## 🙏 Acknowledgments

- ResNet34 pretrained weights from torchvision
- PEFT library for LoRA implementation
- Food-101 and CIFAR-100 datasets


