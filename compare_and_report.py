"""
Generate comparison images and summary plots for SR evaluation results.
- Reads `Outputs/eval/eval_results.csv`
- For each image: regenerates LR, loads SR and GT, creates a labeled grid: LR | SR | GT
- Saves compare images to `Outputs/eval/compare_<image>.png`
- Plots PSNR and SSIM per image and overall average; saves `psnr_ssim_summary.png`
- Writes a short markdown report `Outputs/eval/report.md`
"""

import csv
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import matplotlib.pyplot as plt
from evaluate_sr import down_up_rgb


def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)


def make_grid(lr_img, sr_img, gt_img, label_font=None):
    # lr_img, sr_img, gt_img are PIL Images (RGB)
    # Resize them to same height (choose GT size)
    w, h = gt_img.size
    lr = lr_img.resize((w, h), Image.BICUBIC)
    sr = sr_img.resize((w, h), Image.BICUBIC)

    # Make a new image triple wide
    grid = Image.new('RGB', (w * 3, h + 40), (255, 255, 255))
    grid.paste(lr, (0, 0))
    grid.paste(sr, (w, 0))
    grid.paste(gt_img, (w * 2, 0))

    draw = ImageDraw.Draw(grid)
    try:
        font = label_font or ImageFont.load_default()
    except Exception:
        font = ImageFont.load_default()

    # Labels
    draw.text((w // 2 - 20, h + 5), "LR", fill=(0, 0, 0), font=font)
    draw.text((w + w // 2 - 20, h + 5), "SR", fill=(0, 0, 0), font=font)
    draw.text((2 * w + w // 2 - 30, h + 5), "GT", fill=(0, 0, 0), font=font)

    return grid


def run(report_csv='Outputs/eval/eval_results.csv', sample_dir='sample', out_dir='Outputs/eval'):
    report_csv = Path(report_csv)
    out_dir = Path(out_dir)
    ensure_dir(out_dir)

    if not report_csv.exists():
        print(f"CSV not found: {report_csv}")
        return

    rows = []
    with open(report_csv, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    psnrs = []
    ssims = []
    names = []

    for r in rows:
        name = r['image']
        sr_path = Path(r['sr_path'])
        gt_path = Path(sample_dir) / name

        if not sr_path.exists() or not gt_path.exists():
            print(f"Missing files for {name}")
            continue

        # Load GT and SR
        gt = Image.open(gt_path).convert('RGB')
        sr = Image.open(sr_path).convert('RGB')

        # Prefer saved LR if available, otherwise recreate from GT
        lr_path = None
        if 'lr_path' in r and r['lr_path']:
            lr_path = Path(r['lr_path'])
        if lr_path and lr_path.exists():
            lr = Image.open(lr_path).convert('RGB')
        else:
            gt_np = np.array(gt).astype(np.float32) / 255.0
            lr_np = down_up_rgb(gt_np, factor=2)
            lr = Image.fromarray((lr_np * 255).astype(np.uint8))

        # Make grid
        grid = make_grid(lr, sr, gt)
        cmp_path = out_dir / f'compare_{Path(name).stem}.png'
        grid.save(cmp_path)
        print(f"Saved comparison image: {cmp_path}")

        psnrs.append(float(r['psnr']))
        ssims.append(float(r['ssim']))
        names.append(name)

    # Plot PSNR and SSIM
    if names:
        x = np.arange(len(names))
        plt.figure(figsize=(10, 6))
        plt.bar(x - 0.15, psnrs, width=0.3, label='PSNR (dB)')
        plt.bar(x + 0.15, ssims, width=0.3, label='SSIM')
        plt.xticks(x, names, rotation=45)
        plt.legend()
        plt.tight_layout()
        plot_path = out_dir / 'psnr_ssim_summary.png'
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved summary plot: {plot_path}")

        # Write report markdown
        report_md = out_dir / 'report.md'
        with open(report_md, 'w') as mf:
            mf.write('# SRCNN Evaluation Report\n\n')
            mf.write('## Metrics\n\n')
            mf.write('| Image | PSNR (dB) | SSIM | Compare |\n')
            mf.write('|---|---:|---:|---|\n')
            for i, name in enumerate(names):
                cmp_name = f'compare_{Path(name).stem}.png'
                mf.write(f'| {name} | {psnrs[i]:.4f} | {ssims[i]:.4f} | ![]({cmp_name}) |\n')
            mf.write('\n')
            mf.write('## Summary\n\n')
            mf.write(f'* Average PSNR: {np.mean(psnrs):.4f} dB\n')
            mf.write(f'* Average SSIM: {np.mean(ssims):.4f}\n')
            mf.write('\n')
            mf.write(f'![psnr_ssim_summary]({plot_path.name})\n')

        print(f"Wrote report: {report_md}")
    else:
        print("No images processed.")


if __name__ == '__main__':
    run()
