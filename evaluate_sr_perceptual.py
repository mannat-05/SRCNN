"""
Evaluate SRCNN model using perceptual weights and save results to Outputs/eval_perceptual
"""
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from srcnn_model import SRCNN
from config_utils import MetricCalculator
import csv


def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)


def down_up_rgb(img_rgb_np, factor=2):
    img = Image.fromarray((img_rgb_np * 255.0).astype(np.uint8))
    small = img.resize((img.width // factor, img.height // factor), Image.BICUBIC)
    up = small.resize((img.width, img.height), Image.BICUBIC)
    up_np = np.array(up).astype(np.float32) / 255.0
    return up_np


def rgb_to_y(img_rgb_np):
    if img_rgb_np.max() <= 1.0:
        img = img_rgb_np * 255.0
    else:
        img = img_rgb_np
    r = img[..., 0]
    g = img[..., 1]
    b = img[..., 2]
    y = (0.299 * r + 0.587 * g + 0.114 * b) / 255.0
    return y


def evaluate_perceptual(model_path='srcnn_final_perceptual.pth', sample_dir='sample', output_dir='Outputs/eval_perceptual'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Create model with kernel sizes matching perceptual checkpoint
    model = SRCNN(num_channels=3, kernel_size_1=9, kernel_size_2=1, kernel_size_3=5)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    sample_dir = Path(sample_dir)
    files = sorted([p for ext in ('*.jpg','*.jpeg','*.png') for p in sample_dir.glob(ext)])
    if not files:
        print(f"No images found in {sample_dir}")
        return

    ensure_dir(output_dir)
    csv_path = Path(output_dir) / 'eval_results_perceptual.csv'

    rows = []

    for p in files:
        img = Image.open(p).convert('RGB')
        img_np = np.array(img).astype(np.float32) / 255.0

        lr_rgb = down_up_rgb(img_np, factor=2)

        lr_dir = Path(output_dir) / 'lr'
        lr_dir.mkdir(parents=True, exist_ok=True)
        lr_img_path = lr_dir / f'{p.stem}_perceptual_lr.png'
        lr_uint8 = (lr_rgb * 255.0).clip(0, 255).astype(np.uint8)
        Image.fromarray(lr_uint8).save(lr_img_path)

        inp = torch.from_numpy(lr_rgb.transpose(2, 0, 1)).unsqueeze(0).float().to(device)

        with torch.no_grad():
            out = model(inp)
        out_np = out.squeeze(0).cpu().numpy().transpose(1, 2, 0)
        out_np = np.clip(out_np, 0.0, 1.0)

        y_true = rgb_to_y(img_np)
        y_sr = rgb_to_y(out_np)

        psnr_val = MetricCalculator.psnr(y_true, y_sr, max_val=1.0)
        ssim_val = MetricCalculator.ssim(y_true, y_sr)

        sr_path = Path(output_dir) / f'{p.stem}_perceptual_sr.png'
        sr_rgb_uint8 = (out_np * 255.0).clip(0, 255).astype(np.uint8)
        Image.fromarray(sr_rgb_uint8).save(sr_path)

        rows.append((p.name, float(psnr_val), float(ssim_val), str(sr_path), str(lr_img_path)))
        print(f'Processed {p.name}: PSNR={psnr_val:.4f} dB, SSIM={ssim_val:.4f}')

    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['image', 'psnr', 'ssim', 'sr_path', 'lr_path'])
        writer.writerows(rows)

    print(f'Perceptual evaluation finished. Results saved to {csv_path}')


if __name__ == '__main__':
    evaluate_perceptual()
