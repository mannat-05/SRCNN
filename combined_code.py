import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms, models
from torchvision.models import VGG19_Weights
from PIL import Image
import matplotlib.pyplot as plt

# ==========================================
# 1. SMART PATH FINDER
# ==========================================
print("🔍 Searching for the 'train' folder...")
DATASET_PATH = None
base_search = '/kaggle/input'

for root, dirs, files in os.walk(base_search):
    if 'train' in root.lower():
        image_count = len([f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        if image_count > 10:
            DATASET_PATH = root
            print(f"✅ Found TRAINING folder with {image_count} images: {DATASET_PATH}")
            break

if DATASET_PATH is None:
    # Fallback search
    for root, dirs, files in os.walk(base_search):
        image_count = len([f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        if image_count > 10:
            DATASET_PATH = root
            print(f"⚠️ 'train' folder not found. Using fallback folder: {DATASET_PATH}")
            break

if DATASET_PATH is None:
    raise ValueError("❌ Could not find any images! Please check your Input directory.")

# ==========================================
# 2. CONFIGURATION
# ==========================================
BATCH_SIZE = 16
NUM_EPOCHS = 100        # Reduced slightly as VGG loss converges differently
LEARNING_RATE = 0.0001 # Reduced LR for stability with Perceptual Loss
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using Device: {DEVICE}")

# ==========================================
# 3. PROFICIENT LOSS FUNCTIONS
# ==========================================

# A. VGG Feature Loss (Perceptual Loss)
class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()
        # Load VGG19 pretrained on ImageNet
        vgg = models.vgg19(weights=VGG19_Weights.DEFAULT)
        # Extract features up to the 35th layer (conv5_4) - standard for SR
        self.feature_extractor = nn.Sequential(*list(vgg.features.children())[:35]).eval().to(DEVICE)
        
        # Freeze parameters (we don't train VGG)
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
            
        self.criterion = nn.L1Loss() # Using L1 for feature distance

    def forward(self, sr, hr):
        # VGG expects normalized images. We assume inputs are [0,1] tensors.
        # Standard ImageNet normalization logic could be added here if strictly necessary,
        # but for SR, direct feature extraction is often sufficient.
        sr_features = self.feature_extractor(sr)
        hr_features = self.feature_extractor(hr)
        return self.criterion(sr_features, hr_features)

# B. Combined Loss (Pixel + Perceptual)
class SRCombinedLoss(nn.Module):
    def __init__(self, pixel_weight=1.0, feature_weight=0.006):
        super(SRCombinedLoss, self).__init__()
        self.pixel_weight = pixel_weight
        self.feature_weight = feature_weight
        
        self.pixel_loss = nn.L1Loss() # Sharper than MSE
        self.feature_loss = VGGLoss()

    def forward(self, sr, hr):
        l1 = self.pixel_loss(sr, hr)
        vgg = self.feature_loss(sr, hr)
        
        # Weighted sum
        total_loss = (self.pixel_weight * l1) + (self.feature_weight * vgg)
        return total_loss, l1, vgg

# ==========================================
# 4. MODEL DEFINITION (SRCNN)
# ==========================================
class SRCNN(nn.Module):
    def __init__(self, num_channels=3):
        super(SRCNN, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=9, padding=4) # Padding fixed to 4 for 9x9
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1, padding=0) # 1x1 conv typically has 0 padding
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(32, num_channels, kernel_size=5, padding=2)
        
    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.conv3(x)
        return x

# ==========================================
# 5. DATASET CLASS
# ==========================================
class SRCNNDataset(Dataset):
    def __init__(self, root_dir, upscale_factor=2):
        self.root_dir = root_dir
        self.image_files = [f for f in os.listdir(root_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.upscale_factor = upscale_factor
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_files[idx])
        try:
            image = Image.open(img_name).convert('RGB')
        except:
            return self.__getitem__(0)
        
        # Crop 256x256 patch
        if image.width > 256 and image.height > 256:
            transform_crop = transforms.RandomCrop(256)
            image_hr = transform_crop(image)
        else:
            image_hr = image 

        w, h = image_hr.size
        new_w = w - (w % self.upscale_factor)
        new_h = h - (h % self.upscale_factor)
        image_hr = image_hr.crop((0, 0, new_w, new_h))
        
        image_lr = image_hr.resize((new_w // self.upscale_factor, new_h // self.upscale_factor), resample=Image.BICUBIC)
        image_lr_upscaled = image_lr.resize((new_w, new_h), resample=Image.BICUBIC)
        
        return self.to_tensor(image_lr_upscaled), self.to_tensor(image_hr)

# ==========================================
# 6. TRAINING LOOP (UPDATED)
# ==========================================
# Load Data
full_dataset = SRCNNDataset(root_dir=DATASET_PATH, upscale_factor=2)
train_size = int(0.9 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# Initialize Model & Loss
model = SRCNN(num_channels=3).to(DEVICE)

# --- USING NEW LOSS FUNCTION ---
criterion = SRCombinedLoss(pixel_weight=1.0, feature_weight=0.01).to(DEVICE) # increased weight slightly for SRCNN
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

print(f"Starting training with Perceptual Loss for {NUM_EPOCHS} epochs...")

for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    running_l1 = 0.0
    running_vgg = 0.0
    
    for i, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        
        # Calculate Combined Loss
        loss, l1_loss, vgg_loss = criterion(outputs, targets)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        running_l1 += l1_loss.item()
        running_vgg += vgg_loss.item()
        
    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs = model(inputs)
            loss, _, _ = criterion(outputs, targets)
            val_loss += loss.item()
            
    avg_train_loss = running_loss / len(train_loader)
    avg_l1 = running_l1 / len(train_loader)
    avg_vgg = running_vgg / len(train_loader)
    avg_val_loss = val_loss / len(val_loader)
    
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] Total: {avg_train_loss:.4f} | L1: {avg_l1:.4f} | VGG: {avg_vgg:.4f} | Val: {avg_val_loss:.4f}")
    
    # Save best model
    if epoch % 10 == 0:
        torch.save(model.state_dict(), f'srcnn_perceptual_epoch_{epoch}.pth')
    
torch.save(model.state_dict(), 'srcnn_final_perceptual.pth')
print("🎉 Training Finished!")
