import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from pathlib import Path


class SRCNN(nn.Module):
    """
    Super-Resolution Convolutional Neural Network (SRCNN)
    
    Architecture:
    1. Patch extraction and representation layer
    2. Non-linear mapping layer
    3. Reconstruction layer
    """
    
    def __init__(self, num_channels=1, num_filters=64, kernel_size_1=11, kernel_size_2=3, kernel_size_3=5):
        super(SRCNN, self).__init__()
        
        # Layer 1: Patch extraction and representation
        # Input: Low-resolution image
        # Output: Feature maps
        self.conv1 = nn.Conv2d(num_channels, num_filters, kernel_size=kernel_size_1, padding=kernel_size_1//2)
        
        # Layer 2: Non-linear mapping
        # Input: Feature maps from layer 1
        # Output: High-dimensional feature maps
        self.conv2 = nn.Conv2d(num_filters, 32, kernel_size=kernel_size_2, padding=kernel_size_2//2)
        
        # Layer 3: Reconstruction
        # Input: Feature maps from layer 2
        # Output: Super-resolved image
        self.conv3 = nn.Conv2d(32, num_channels, kernel_size=kernel_size_3, padding=kernel_size_3//2)
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        """
        Forward pass through SRCNN
        
        Args:
            x: Input low-resolution image
            
        Returns:
            Super-resolved image
        """
        # Patch extraction
        x = self.relu(self.conv1(x))
        
        # Non-linear mapping
        x = self.relu(self.conv2(x))
        
        # Reconstruction
        x = self.conv3(x)
        
        return x


class SRCNNTrainer:
    """
    Trainer class for SRCNN model
    """
    
    def __init__(self, model, learning_rate=0.001, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=50, gamma=0.1)
        
    def train_epoch(self, train_loader, batch_callback=None):
        """Train for one epoch

        Args:
            train_loader: DataLoader for training
            batch_callback: Optional function called after each batch with signature
                batch_callback(batch_idx, lr_images_cpu, hr_images_cpu, outputs_cpu, loss_value)
        """
        self.model.train()
        total_loss = 0.0
        
        for batch_idx, (lr_images, hr_images) in enumerate(train_loader):
            lr_images = lr_images.to(self.device)
            hr_images = hr_images.to(self.device)
            
            # Forward pass
            outputs = self.model(lr_images)
            loss = self.criterion(outputs, hr_images)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Call optional batch callback with CPU tensors and detached outputs
            if batch_callback is not None:
                try:
                    batch_callback(batch_idx,
                                   lr_images.detach().cpu(),
                                   hr_images.detach().cpu(),
                                   outputs.detach().cpu(),
                                   loss.item())
                except Exception as e:
                    # Avoid stopping training due to callback errors
                    print(f"Warning: batch_callback raised an exception: {e}")
            
            if (batch_idx + 1) % 100 == 0:
                print(f'Batch [{batch_idx + 1}], Loss: {loss.item():.6f}')
        
        return total_loss / len(train_loader)
    
    def validate(self, val_loader):
        """Validate the model"""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for lr_images, hr_images in val_loader:
                lr_images = lr_images.to(self.device)
                hr_images = hr_images.to(self.device)
                
                outputs = self.model(lr_images)
                loss = self.criterion(outputs, hr_images)
                total_loss += loss.item()
        
        return total_loss / len(val_loader)
    
    def train(self, train_loader, val_loader, num_epochs=100):
        """Full training loop"""
        print("Starting SRCNN Training...")
        
        for epoch in range(num_epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)
            self.scheduler.step()
            
            print(f'Epoch [{epoch + 1}/{num_epochs}], '
                  f'Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
        
        print("Training completed!")
    
    def save_model(self, save_path):
        """Save model checkpoint"""
        torch.save(self.model.state_dict(), save_path)
        print(f"Model saved to {save_path}")
    
    def load_model(self, load_path):
        """Load model checkpoint"""
        self.model.load_state_dict(torch.load(load_path, map_location=self.device))
        print(f"Model loaded from {load_path}")


class SRCNNDataset(Dataset):
    """
    Custom Dataset for SRCNN training
    Requires low-resolution and high-resolution image pairs
    """
    
    def __init__(self, lr_images, hr_images, patch_size=64, stride=32):
        """
        Args:
            lr_images: List of low-resolution image paths or arrays
            hr_images: List of high-resolution image paths or arrays
            patch_size: Size of patches to extract
            stride: Stride for patch extraction
        """
        self.patches_lr = []
        self.patches_hr = []
        self.patch_size = patch_size
        
        # Extract patches from images
        for lr_img, hr_img in zip(lr_images, hr_images):
            self._extract_patches(lr_img, hr_img, stride)
    
    def _extract_patches(self, lr_img, hr_img, stride):
        """Extract patches from image pairs"""
        if isinstance(lr_img, str):
            # Load from file if string path is provided
            # This is a placeholder - implement actual image loading based on your needs
            pass
        else:
            # Assume numpy array
            h, w = lr_img.shape[:2]
            for i in range(0, h - self.patch_size + 1, stride):
                for j in range(0, w - self.patch_size + 1, stride):
                    lr_patch = lr_img[i:i+self.patch_size, j:j+self.patch_size]
                    hr_patch = hr_img[i:i+self.patch_size, j:j+self.patch_size]
                    self.patches_lr.append(lr_patch)
                    self.patches_hr.append(hr_patch)
    
    def __len__(self):
        return len(self.patches_lr)
    
    def __getitem__(self, idx):
        lr_patch = torch.from_numpy(self.patches_lr[idx]).float()
        hr_patch = torch.from_numpy(self.patches_hr[idx]).float()
        
        # Add channel dimension if grayscale
        if lr_patch.dim() == 2:
            lr_patch = lr_patch.unsqueeze(0)
            hr_patch = hr_patch.unsqueeze(0)
        
        return lr_patch, hr_patch


def count_parameters(model):
    """Count total trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Example usage
    print("SRCNN Model Architecture")
    print("-" * 50)
    
    # Initialize model
    model = SRCNN(
        num_channels=1,
        num_filters=64,
        kernel_size_1=11,
        kernel_size_2=3,
        kernel_size_3=5
    )
    
    print(model)
    print(f"\nTotal Parameters: {count_parameters(model):,}")
    
    # Test forward pass with dummy data
    dummy_input = torch.randn(1, 1, 64, 64)  # Batch size 1, 1 channel, 64x64 image
    dummy_output = model(dummy_input)
    
    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Output shape: {dummy_output.shape}")
