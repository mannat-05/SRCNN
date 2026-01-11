"""
Evaluate SRCNN model on images in `sample/`
- Loads `srcnn_final.pth` (created earlier) into SRCNN model
- For each image in `sample/`:
  - Load image, convert to YCbCr
  - Create LR input by downsampling by factor 2 and upsampling back using bicubic
  - Run model on Y channel to produce SR Y
  - Combine SR Y with original CbCr and save SR RGB image
  - Compute PSNR and SSIM between SR Y and original Y
- Save results to `Outputs/eval/` and write `eval_results.csv`
"""

import torch
import numpy as np
from PIL import Image
from pathlib import Path
from srcnn_model import SRCNN
from config_utils import MetricCalculator, ImageUtils
import csv


def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)


def load_image_y(path):
    img = Image.open(path).convert('YCbCr')
    y, cb, cr = img.split()
    y_np = np.array(y).astype(np.float32) / 255.0
    cb_np = np.array(cb).astype(np.uint8)
    cr_np = np.array(cr).astype(np.uint8)
    return y_np, cb_np, cr_np


def upsample_bicubic(image_np, factor=2):
    img = Image.fromarray((image_np * 255.0).astype(np.uint8))
    small = img.resize((img.width // factor, img.height // factor), Image.BICUBIC)
    up = small.resize((img.width, img.height), Image.BICUBIC)
    up_np = np.array(up).astype(np.float32) / 255.0
    return up_np


def save_sr_rgb(y_sr, cb, cr, out_path):
    # y_sr: numpy float in [0,1]
    y_uint8 = (y_sr * 255.0).clip(0, 255).astype(np.uint8)
    y_img = Image.fromarray(y_uint8)
    cb_img = Image.fromarray(cb)
    cr_img = Image.fromarray(cr)
    img = Image.merge('YCbCr', (y_img, cb_img, cr_img)).convert('RGB')
    img.save(out_path)


def rgb_to_y(img_rgb_np):
    """Convert RGB numpy (HxWx3 in [0,1] or [0,255]) to Y channel in [0,1]"""
    if img_rgb_np.max() <= 1.0:
        img = img_rgb_np * 255.0
    else:
        img = img_rgb_np
    r = img[..., 0]
    g = img[..., 1]
    b = img[..., 2]
    y = (0.299 * r + 0.587 * g + 0.114 * b) / 255.0
    return y


def down_up_rgb(img_rgb_np, factor=2):
    """Downsample then upsample RGB image using bicubic"""
    img = Image.fromarray((img_rgb_np * 255.0).astype(np.uint8))
    small = img.resize((img.width // factor, img.height // factor), Image.BICUBIC)
    up = small.resize((img.width, img.height), Image.BICUBIC)
    up_np = np.array(up).astype(np.float32) / 255.0
    return up_np


def evaluate(model_path='srcnn_final.pth', sample_dir='sample', output_dir='Outputs/eval'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Use 3-channel model if checkpoint has 3-channel weights
    model = SRCNN(num_channels=3)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    sample_dir = Path(sample_dir)
    files = sorted([p for ext in ('*.jpg','*.jpeg','*.png') for p in sample_dir.glob(ext)])
    if not files:
        print(f"No images found in {sample_dir}")
        return


    ensure_dir(output_dir)
    csv_path = Path(output_dir) / 'eval_results.csv'

    rows = []

    for p in files:
        # Load original RGB image
        img = Image.open(p).convert('RGB')
        img_np = np.array(img).astype(np.float32) / 255.0  # HxWx3

        # Create LR by down/upsampling on RGB
        lr_rgb = down_up_rgb(img_np, factor=2)

        # Save LR image for later comparison
        lr_dir = Path(output_dir) / 'lr'
        lr_dir.mkdir(parents=True, exist_ok=True)
        lr_img_path = lr_dir / f'{p.stem}_lr.png'
        lr_uint8 = (lr_rgb * 255.0).clip(0, 255).astype(np.uint8)
        Image.fromarray(lr_uint8).save(lr_img_path)

        # Model input: CxHxW
        inp = torch.from_numpy(lr_rgb.transpose(2, 0, 1)).unsqueeze(0).float().to(device)

        with torch.no_grad():
            out = model(inp)
        out_np = out.squeeze(0).cpu().numpy().transpose(1, 2, 0)  # HxWx3
        out_np = np.clip(out_np, 0.0, 1.0)

        # Compute Y channels
        y_true = rgb_to_y(img_np)
        y_sr = rgb_to_y(out_np)

        psnr_val = MetricCalculator.psnr(y_true, y_sr, max_val=1.0)
        ssim_val = MetricCalculator.ssim(y_true, y_sr)

        # Save SR RGB image (use original YCbCr chroma to preserve colors)
        # For simplicity, convert SR RGB to YCbCr by merging channels
        sr_rgb_uint8 = (out_np * 255.0).clip(0, 255).astype(np.uint8)
        sr_img = Image.fromarray(sr_rgb_uint8)
        out_img_path = Path(output_dir) / f'{p.stem}_sr.png'
        sr_img.save(out_img_path)

        rows.append((p.name, float(psnr_val), float(ssim_val), str(out_img_path), str(lr_img_path)))
        print(f'Processed {p.name}: PSNR={psnr_val:.4f} dB, SSIM={ssim_val:.4f}')

    # Save CSV (include LR path)
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['image', 'psnr', 'ssim', 'sr_path', 'lr_path'])
        writer.writerows(rows)

    print(f'Evaluation finished. Results saved to {csv_path}')


if __name__ == '__main__':
    evaluate()
