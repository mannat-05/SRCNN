"""
SRCNN Configuration and Utilities
"""

from dataclasses import dataclass
from typing import Tuple
import torch
import numpy as np
from pathlib import Path


@dataclass
class SRCNNConfig:
    """Configuration class for SRCNN training"""
    
    # Model architecture
    num_channels: int = 1  # 1 for grayscale, 3 for RGB
    num_filters: int = 64
    kernel_size_1: int = 11  # Patch extraction layer
    kernel_size_2: int = 3   # Non-linear mapping layer
    kernel_size_3: int = 5   # Reconstruction layer
    
    # Training hyperparameters
    batch_size: int = 32
    num_epochs: int = 100
    learning_rate: float = 0.001
    weight_decay: float = 0.0
    
    # Data parameters
    patch_size: int = 64
    stride: int = 32
    upscale_factor: int = 2
    
    # Device
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Paths
    data_dir: str = 'data'
    output_dir: str = 'Outputs'
    checkpoint_dir: str = 'Outputs/checkpoints'
    
    def to_dict(self):
        """Convert config to dictionary"""
        return self.__dict__


class MetricCalculator:
    """Calculate super-resolution metrics"""
    
    @staticmethod
    def psnr(y_true: np.ndarray, y_pred: np.ndarray, max_val: float = 1.0) -> float:
        """
        Calculate Peak Signal-to-Noise Ratio (PSNR)
        
        Args:
            y_true: Ground truth image
            y_pred: Predicted image
            max_val: Maximum value (1.0 for normalized, 255 for uint8)
            
        Returns:
            PSNR value in dB
        """
        mse = np.mean((y_true - y_pred) ** 2)
        if mse == 0:
            return float('inf')
        
        psnr_val = 20 * np.log10(max_val / np.sqrt(mse))
        return psnr_val
    
    @staticmethod
    def ssim(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate Structural Similarity Index (SSIM)
        
        Args:
            y_true: Ground truth image
            y_pred: Predicted image
            
        Returns:
            SSIM value (between -1 and 1, higher is better)
        """
        c1, c2 = 0.01 ** 2, 0.03 ** 2
        
        mean_true = np.mean(y_true)
        mean_pred = np.mean(y_pred)
        var_true = np.var(y_true)
        var_pred = np.var(y_pred)
        cov = np.mean((y_true - mean_true) * (y_pred - mean_pred))
        
        ssim_val = ((2 * mean_true * mean_pred + c1) * (2 * cov + c2)) / \
                   ((mean_true ** 2 + mean_pred ** 2 + c1) * (var_true + var_pred + c2))
        
        return ssim_val
    
    @staticmethod
    def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Mean Squared Error"""
        return np.mean((y_true - y_pred) ** 2)


class ImageUtils:
    """Image processing utilities"""
    
    @staticmethod
    def normalize(image: np.ndarray, max_val: float = 255.0) -> np.ndarray:
        """Normalize image to [0, 1]"""
        return image.astype(np.float32) / max_val
    
    @staticmethod
    def denormalize(image: np.ndarray, max_val: float = 255.0) -> np.ndarray:
        """Denormalize image from [0, 1] to [0, max_val]"""
        return (image * max_val).clip(0, max_val).astype(np.uint8)
    
    @staticmethod
    def rgb2ycbcr(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Convert RGB to YCbCr color space
        Useful for super-resolution as Y channel contains luminance info
        """
        image = image.astype(np.float32)
        y = 0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2]
        cb = -0.169 * image[..., 0] - 0.331 * image[..., 1] + 0.5 * image[..., 2]
        cr = 0.5 * image[..., 0] - 0.419 * image[..., 1] - 0.081 * image[..., 2]
        
        return y, cb, cr
    
    @staticmethod
    def ycbcr2rgb(y: np.ndarray, cb: np.ndarray, cr: np.ndarray) -> np.ndarray:
        """Convert YCbCr back to RGB"""
        r = y + 1.402 * cr
        g = y - 0.34414 * cb - 0.71414 * cr
        b = y + 1.772 * cb
        
        image = np.stack([r, g, b], axis=-1)
        return image.clip(0, 255).astype(np.uint8)
    
    @staticmethod
    def downsample(image: np.ndarray, factor: int = 2) -> np.ndarray:
        """Downsample image by factor"""
        return image[::factor, ::factor]
    
    @staticmethod
    def upsample_bilinear(image: np.ndarray, factor: int = 2) -> np.ndarray:
        """Upsample image using bilinear interpolation"""
        h, w = image.shape[:2]
        new_h, new_w = h * factor, w * factor
        
        upsampled = np.zeros((*image.shape[:-1], new_h, new_w) if image.ndim > 2 else (new_h, new_w), 
                            dtype=image.dtype)
        
        for i in range(new_h):
            for j in range(new_w):
                # Original coordinates
                orig_i = i / factor
                orig_j = j / factor
                
                # Integer and fractional parts
                i0, i1 = int(orig_i), min(int(orig_i) + 1, h - 1)
                j0, j1 = int(orig_j), min(int(orig_j) + 1, w - 1)
                
                di = orig_i - i0
                dj = orig_j - j0
                
                # Bilinear interpolation
                val = (1 - di) * (1 - dj) * image[i0, j0] + \
                      di * (1 - dj) * image[i1, j0] + \
                      (1 - di) * dj * image[i0, j1] + \
                      di * dj * image[i1, j1]
                
                upsampled[i, j] = val
        
        return upsampled


class CheckpointManager:
    """Manage model checkpoints"""
    
    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def save_checkpoint(self, model, optimizer, epoch, loss, filename: str = None):
        """Save model checkpoint"""
        if filename is None:
            filename = f"checkpoint_epoch_{epoch}.pth"
        
        filepath = self.checkpoint_dir / filename
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }
        
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved: {filepath}")
    
    def load_checkpoint(self, model, optimizer, filename: str):
        """Load model checkpoint"""
        filepath = self.checkpoint_dir / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"Checkpoint not found: {filepath}")
        
        checkpoint = torch.load(filepath)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        print(f"Checkpoint loaded: {filepath}")
        print(f"Epoch: {checkpoint['epoch']}, Loss: {checkpoint['loss']}")
        
        return checkpoint['epoch']
    
    def get_latest_checkpoint(self):
        """Get latest checkpoint"""
        checkpoints = list(self.checkpoint_dir.glob('checkpoint_*.pth'))
        if not checkpoints:
            return None
        
        return max(checkpoints, key=lambda p: p.stat().st_mtime)


if __name__ == "__main__":
    # Example usage
    config = SRCNNConfig()
    print("SRCNN Configuration:")
    for key, value in config.to_dict().items():
        print(f"  {key}: {value}")
    
    # Test metrics
    y_true = np.random.rand(100, 100)
    y_pred = y_true + np.random.randn(100, 100) * 0.01
    
    print("\nMetric Examples:")
    print(f"  PSNR: {MetricCalculator.psnr(y_true, y_pred):.2f} dB")
    print(f"  SSIM: {MetricCalculator.ssim(y_true, y_pred):.4f}")
    print(f"  MSE: {MetricCalculator.mse(y_true, y_pred):.6f}")
