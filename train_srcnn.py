"""
SRCNN Training and Inference Example
This script demonstrates how to train and use the SRCNN model
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
from srcnn_model import SRCNN, SRCNNTrainer
from config_utils import MetricCalculator
from pathlib import Path
from PIL import Image
import random
import os
import time
from datetime import datetime
import torchvision.utils as vutils
from torch.utils.tensorboard import SummaryWriter



def create_dummy_dataset(num_images=10, img_size=256):
    """
    Create dummy dataset for demonstration
    In practice, use real low-resolution and high-resolution image pairs
    """
    lr_images = []
    hr_images = []
    
    for _ in range(num_images):
        # Create synthetic LR image (downsampled version)
        hr_image = np.random.rand(img_size, img_size).astype(np.float32)
        # Simulate downsampling and upsampling
        lr_image = hr_image[::2, ::2]  # Downsample by factor of 2
        lr_image = np.kron(lr_image, np.ones((2, 2)))  # Upsample back
        
        lr_images.append(lr_image)
        hr_images.append(hr_image)
    
    return lr_images, hr_images


def train_srcnn(
    num_epochs=100,
    batch_size=32,
    learning_rate=0.001,
    patch_size=64,
    stride=32,
):
    """
    Train SRCNN model
    
    Args:
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        patch_size: Size of patches for training
        stride: Stride for patch extraction
    """
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    model = SRCNN(
        num_channels=1,
        num_filters=64,
        kernel_size_1=11,
        kernel_size_2=3,
        kernel_size_3=5
    )
    
    print("\nModel Architecture:")
    print(model)
    
    # Create dummy dataset (replace with real dataset)
    print("\nCreating dataset...")
    lr_images, hr_images = create_dummy_dataset(num_images=100)
    
    # Split into train and validation
    split_idx = int(0.8 * len(lr_images))
    train_lr = lr_images[:split_idx]
    train_hr = hr_images[:split_idx]
    val_lr = lr_images[split_idx:]
    val_hr = hr_images[split_idx:]
    
    # Create datasets and dataloaders
    train_dataset = SRCNNDataset(train_lr, train_hr, patch_size=patch_size, stride=stride)
    val_dataset = SRCNNDataset(val_lr, val_hr, patch_size=patch_size, stride=stride)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Train model
    trainer = SRCNNTrainer(model, learning_rate=learning_rate, device=device)
    trainer.train(train_loader, val_loader, num_epochs=num_epochs)
    
    # Save model
    output_dir = Path("Outputs")
    output_dir.mkdir(exist_ok=True)
    model_path = output_dir / "srcnn_model.pth"
    trainer.save_model(str(model_path))
    
    return model, trainer


def inference_example(model_path=None, device=None):
    """
    Example of using SRCNN for inference
    """
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = SRCNN().to(device)
    
    if model_path and Path(model_path).exists():
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model from {model_path}")
    
    model.eval()
    
    # Create dummy low-resolution image
    lr_image = torch.randn(1, 1, 64, 64).to(device)
    
    with torch.no_grad():
        sr_image = model(lr_image)
    
    print(f"\nInference Example:")
    print(f"Input (LR) shape: {lr_image.shape}")
    print(f"Output (SR) shape: {sr_image.shape}")
    
    return sr_image


def model_summary():
    """Print detailed model summary"""
    from srcnn_model import count_parameters
    
    model = SRCNN()
    
    print("\n" + "="*60)
    print("SRCNN Model Summary")
    print("="*60)
    
    print("\nLayer Details:")
    print("-" * 60)
    for name, module in model.named_children():
        if isinstance(module, nn.Conv2d):
            print(f"{name}:")
            print(f"  Input channels: {module.in_channels}")
            print(f"  Output channels: {module.out_channels}")
            print(f"  Kernel size: {module.kernel_size}")
            print(f"  Padding: {module.padding}")
            print(f"  Parameters: {module.in_channels * module.out_channels * module.kernel_size[0] * module.kernel_size[1]:,}")
    
    print("-" * 60)
    print(f"Total Parameters: {count_parameters(model):,}")
    print("="*60)


# ----------------------------
# Dataset + Folder training
# ----------------------------

class TrainFolderDataset(Dataset):
    """Dataset that loads HR images from a folder and generates LR inputs on-the-fly.
    Returns (lr_tensor, hr_tensor) with values in [0, 1]
    """
    def __init__(self, hr_dir, patch_size=64, upscale=2, patches_per_image=10, transform=None):
        self.hr_dir = Path(hr_dir)
        self.image_paths = sorted([p for p in self.hr_dir.glob('*.jpg')])
        self.patch_size = patch_size
        self.upscale = upscale
        self.patches_per_image = patches_per_image
        self.transform = transform

        if len(self.image_paths) == 0:
            raise RuntimeError(f"No images found in {hr_dir}")

    def __len__(self):
        return len(self.image_paths) * self.patches_per_image

    def __getitem__(self, idx):
        img_idx = idx // self.patches_per_image
        img_path = self.image_paths[img_idx]

        # Load image and convert to Y channel (luminance)
        img = Image.open(img_path).convert('YCbCr')
        y, _, _ = img.split()

        # Convert to numpy float32 [0,1]
        hr = np.array(y).astype(np.float32) / 255.0

        h, w = hr.shape
        if h < self.patch_size or w < self.patch_size:
            # Resize up to patch size if too small
            y_resized = y.resize((max(self.patch_size, w), max(self.patch_size, h)), Image.BICUBIC)
            hr = np.array(y_resized).astype(np.float32) / 255.0
            h, w = hr.shape

        # Random crop HR patch
        top = random.randint(0, h - self.patch_size)
        left = random.randint(0, w - self.patch_size)
        hr_patch = hr[top:top + self.patch_size, left:left + self.patch_size]

        # Create LR by downsampling and upsampling (bicubic)
        hr_patch_img = Image.fromarray((hr_patch * 255.0).astype(np.uint8))
        small = hr_patch_img.resize((self.patch_size // self.upscale, self.patch_size // self.upscale), Image.BICUBIC)
        lr_patch_img = small.resize((self.patch_size, self.patch_size), Image.BICUBIC)
        lr_patch = np.array(lr_patch_img).astype(np.float32) / 255.0

        # Convert to tensors with channel dim
        lr_tensor = torch.from_numpy(lr_patch).unsqueeze(0).float()
        hr_tensor = torch.from_numpy(hr_patch).unsqueeze(0).float()

        return lr_tensor, hr_tensor


def train_on_folder(
    hr_dir='train',
    num_epochs=50,
    batch_size=32,
    learning_rate=0.001,
    patch_size=64,
    patches_per_image=20,
    val_split=0.1,
    stream_interval=10,
    device=None,
):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Prepare image paths
    hr_dir_path = Path(hr_dir)
    all_images = sorted([p for p in hr_dir_path.glob('*.jpg')])
    if not all_images:
        raise RuntimeError(f"No images found in {hr_dir}")

    split = int(len(all_images) * (1 - val_split))
    train_imgs = all_images[:split]
    val_imgs = all_images[split:]

    train_dataset = TrainFolderDataset(hr_dir_path, patch_size=patch_size, patches_per_image=patches_per_image)
    val_dataset = TrainFolderDataset(hr_dir_path, patch_size=patch_size, patches_per_image=5)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Model
    model = SRCNN(num_channels=1).to(device)
    trainer = SRCNNTrainer(model, learning_rate=learning_rate, device=device)

    best_psnr = 0.0

    # Setup streaming / logging
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = Path('Outputs') / f'runs_{ts}'
    run_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(run_dir))
    stream_dir = run_dir / 'stream'
    stream_dir.mkdir(parents=True, exist_ok=True)

    # Callback to stream per-batch images and loss
    def batch_callback(batch_idx, lr_cpu, hr_cpu, out_cpu, loss_value):
        # stream every stream_interval batches
        if batch_idx % stream_interval != 0:
            return

        # Take first sample in batch
        lr_img = lr_cpu[0].numpy()  # (1, H, W)
        sr_img = out_cpu[0].numpy()
        hr_img = hr_cpu[0].numpy()

        # Stack to 3 channels for visualization
        def to_3ch(x):
            if x.ndim == 2:
                return np.stack([x, x, x], axis=0)
            return x

        lr_vis = to_3ch(lr_img)
        sr_vis = to_3ch(sr_img)
        hr_vis = to_3ch(hr_img)

        # Create grid: LR | SR | HR
        grid = vutils.make_grid([torch.from_numpy(lr_vis), torch.from_numpy(sr_vis), torch.from_numpy(hr_vis)], nrow=3, normalize=True, scale_each=True)

        # Log to TensorBoard
        writer.add_image('batch_sample', grid, global_step=epoch * len(train_loader) + batch_idx)

        # Save image to disk
        np_grid = (grid.mul(255).byte().permute(1, 2, 0).numpy())
        out_path = stream_dir / f'epoch{epoch+1:03d}_batch{batch_idx:04d}.png'
        Image.fromarray(np_grid).save(out_path)
        print(f"Saved stream image: {out_path}")

        # Also log scalar
        writer.add_scalar('train/batch_loss', loss_value, global_step=epoch * len(train_loader) + batch_idx)

    # Training loop with batch callback
    for epoch in range(num_epochs):
        train_loss = trainer.train_epoch(train_loader, batch_callback=batch_callback)
        val_loss = trainer.validate(val_loader)

        # Compute PSNR on validation set
        model.eval()
        psnr_sum = 0.0
        count = 0
        with torch.no_grad():
            for lr_batch, hr_batch in val_loader:
                lr_batch = lr_batch.to(device)
                hr_batch = hr_batch.to(device)
                out = model(lr_batch)

                out_np = out.cpu().numpy()
                hr_np = hr_batch.cpu().numpy()
                for i in range(out_np.shape[0]):
                    psnr_sum += MetricCalculator.psnr(hr_np[i,0], out_np[i,0], max_val=1.0)
                    count += 1

        avg_psnr = psnr_sum / max(1, count)

        print(f'Epoch [{epoch + 1}/{num_epochs}]  Train Loss: {train_loss:.6f}  Val Loss: {val_loss:.6f}  Val PSNR: {avg_psnr:.4f} dB')

        # Log epoch scalars
        writer.add_scalar('epoch/train_loss', train_loss, epoch+1)
        writer.add_scalar('epoch/val_loss', val_loss, epoch+1)
        writer.add_scalar('epoch/val_psnr', avg_psnr, epoch+1)

        # Save best model
        output_dir = Path('Outputs')
        output_dir.mkdir(exist_ok=True)
        model_path = output_dir / 'srcnn_latest.pth'
        trainer.save_model(str(model_path))

        if avg_psnr > best_psnr:
            best_psnr = avg_psnr
            trainer.save_model(str(output_dir / 'srcnn_best.pth'))

    # Close the writer
    writer.close()

    print('Training finished')
    return model, trainer


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train SRCNN on folder')
    parser.add_argument('--hr_dir', type=str, default='train', help='Directory with HR training images')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--patch_size', type=int, default=64, help='Patch size')
    parser.add_argument('--patches_per_image', type=int, default=20, help='Patches per image')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--stream_interval', type=int, default=10, help='Stream/save one batch image every N batches')

    args = parser.parse_args()

    print("Starting folder-based training")
    train_on_folder(hr_dir=args.hr_dir, num_epochs=args.epochs, batch_size=args.batch_size,
                    learning_rate=args.lr, patch_size=args.patch_size, patches_per_image=args.patches_per_image,
                    stream_interval=args.stream_interval)
