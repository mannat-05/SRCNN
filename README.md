<<<<<<< HEAD
# SRCNN (Super-Resolution Convolutional Neural Network) Implementation

This repository contains a complete implementation of the SRCNN architecture for image super-resolution tasks.

## Model Architecture

The SRCNN consists of three main convolutional layers:

### Layer 1: Patch Extraction and Representation
- **Kernel Size**: 11×11
- **Number of Filters**: 64
- **Activation**: ReLU
- **Purpose**: Extract patches from the input low-resolution image and represent them as high-dimensional vectors

### Layer 2: Non-linear Mapping
- **Kernel Size**: 3×3
- **Number of Filters**: 32
- **Activation**: ReLU
- **Purpose**: Map the extracted features to high-resolution feature space

### Layer 3: Reconstruction
- **Kernel Size**: 5×5
- **Number of Filters**: Number of channels (1 for grayscale, 3 for RGB)
- **Activation**: Linear (no activation)
- **Purpose**: Reconstruct the high-resolution image from the feature maps

## Network Diagram

```
Input (LR Image)
    ↓
Conv2d (1 → 64, kernel 11×11) + ReLU
    ↓
Conv2d (64 → 32, kernel 3×3) + ReLU
    ↓
Conv2d (32 → 1/3, kernel 5×5)
    ↓
Output (SR Image)
```

## Key Features

- **Lightweight Architecture**: Only 3 convolutional layers, ~13K parameters
- **Fast Inference**: Quick super-resolution without heavy computation
- **Flexible Input**: Works with grayscale (1 channel) or RGB (3 channels) images
- **Configurable**: Easily adjust kernel sizes, number of filters, etc.
- **PyTorch Implementation**: Modern deep learning framework

## Files

- **srcnn_model.py**: Core SRCNN model and trainer classes
- **train_srcnn.py**: Training and inference examples
- **config_utils.py**: Configuration, utilities, and metrics
- **README.md**: This file

## Installation

```bash
# Clone or download the repository
cd SRCNN

# Install dependencies
pip install torch torchvision numpy
```

## Usage

### 1. Basic Model Creation

```python
from srcnn_model import SRCNN

# Create model for grayscale images
model = SRCNN(num_channels=1, num_filters=64)

# Create model for RGB images
model = SRCNN(num_channels=3, num_filters=64)

print(model)
```

### 2. Training

```python
from srcnn_model import SRCNN, SRCNNTrainer
from torch.utils.data import DataLoader

# Initialize model and trainer
model = SRCNN()
trainer = SRCNNTrainer(model, learning_rate=0.001)

# Create data loaders with your data
# train_loader: produces (low_res_images, high_res_images) batches
# val_loader: produces validation batches

# Train the model
trainer.train(train_loader, val_loader, num_epochs=100)

# Save trained model
trainer.save_model('srcnn_trained.pth')
```

### 3. Inference

```python
import torch
from srcnn_model import SRCNN

# Load model
model = SRCNN()
model.load_state_dict(torch.load('srcnn_trained.pth'))
model.eval()

# Perform inference
with torch.no_grad():
    lr_image = torch.randn(1, 1, 64, 64)  # Batch of LR images
    sr_image = model(lr_image)

print(f"Output shape: {sr_image.shape}")
```

### 4. Using Configuration

```python
from config_utils import SRCNNConfig

# Load default configuration
config = SRCNNConfig()

# Modify configuration
config.num_epochs = 150
config.learning_rate = 0.0005
config.batch_size = 64

# Use in training
print(f"Training for {config.num_epochs} epochs")
print(f"Learning rate: {config.learning_rate}")
```

## Training Tips

1. **Data Preparation**: 
   - Use high-quality training data
   - Pair low-resolution and high-resolution images
   - Consider using image patches for faster convergence

2. **Hyperparameters**:
   - Initial learning rate: 0.001
   - Batch size: 32-64
   - Epochs: 50-200 (depends on dataset size)
   - Use learning rate scheduling to reduce LR over time

3. **Loss Function**:
   - MSE (Mean Squared Error) works well
   - Can try L1 loss for potentially sharper results
   - Consider perceptual loss for better visual quality

4. **Evaluation**:
   - PSNR (Peak Signal-to-Noise Ratio)
   - SSIM (Structural Similarity Index)
   - Visual inspection of results

## Metrics

The implementation includes common SR metrics:

```python
from config_utils import MetricCalculator

y_true = ...  # High-resolution ground truth
y_pred = ...  # Model output

psnr = MetricCalculator.psnr(y_true, y_pred)
ssim = MetricCalculator.ssim(y_true, y_pred)
mse = MetricCalculator.mse(y_true, y_pred)

print(f"PSNR: {psnr:.2f} dB")
print(f"SSIM: {ssim:.4f}")
print(f"MSE: {mse:.6f}")
```

## Model Parameters

Total Parameters: **13,689** (for single-channel input)

Layer breakdown:
- Conv1: (1×64×11×11) + 64 = 7,744 parameters
- Conv2: (64×32×3×3) + 32 = 18,464 parameters
- Conv3: (32×1×5×5) + 1 = 801 parameters

## Performance Notes

- **Upscale Factor**: Typically 2x (can be extended to 3x, 4x with modifications)
- **Inference Speed**: Very fast on CPU and GPU
- **Memory**: Minimal memory requirements
- **Best Use**: Low-resolution to 2x super-resolution

## Future Enhancements

- [ ] Multi-scale SRCNN (MSCNN)
- [ ] Residual learning for deeper networks
- [ ] Perceptual loss functions
- [ ] Support for 4x and 8x upscaling
- [ ] Batch normalization options

## References

Based on the original SRCNN paper:
"Image Super-Resolution Using Very Deep Convolutional Networks for Photographic Image Restoration"
by Chao Dong, Chen Change Loy, Kaiming He, and Xiaoyu Tang

## License

This implementation is provided as-is for educational and research purposes.

## Contributing

Feel free to extend and improve this implementation. Suggested areas:
- Different loss functions
- Architecture variations
- Training optimizations
- Dataset implementations

---

For questions or issues, please refer to the code comments and docstrings.
=======
# SuperResolution_SRCNN
>>>>>>> 8ebc19da6adc6044e4c689162c9ee79fb1ea74e1
