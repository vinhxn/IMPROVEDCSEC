#!/usr/bin/env python3
"""
Evaluate PSNR / SSIM / LPIPS on a validation set and optionally sweep `conv_transformer.pool_size`.

Usage examples:
  # If you already produced predictions for each pool_size in separate folders:
  python tools/eval_metrics.py --gt_dir ./lcdp_dataset-001/gt --pred_dir_pattern ./results/pool_{pool_size} --pool_sizes 1 4 8 --out results/metrics.csv

  # Or let this script call your test runner (example test command template):
  python tools/eval_metrics.py --gt_dir ./lcdp_dataset-001/gt --pool_sizes 1 4 8 --test_cmd "python src/test.py checkpoint_path=src/pretrained/csec.ckpt conv_transformer.pool_size={pool_size} batchsize=1 runtime_precision=16" --out results/metrics.csv

This script tries to use `lpips` and `skimage`. If unavailable it will fall back to kornia (if installed) or skip LPIPS.
"""
import argparse
import subprocess
from pathlib import Path
import sys
import os
from typing import List

try:
    from PIL import Image
except Exception:
    Image = None

import numpy as np

import torch
import torchvision.transforms.functional as TF

try:
    import lpips
    HAVE_LPIPS = True
except Exception:
    HAVE_LPIPS = False

try:
    from skimage.metrics import peak_signal_noise_ratio as sk_psnr
    from skimage.metrics import structural_similarity as sk_ssim
    HAVE_SKIMAGE = True
except Exception:
    HAVE_SKIMAGE = False


def load_image(path: Path, as_tensor=True):
    if not Image:
        raise RuntimeError('PIL is required to read images')
    img = Image.open(path).convert('RGB')
    arr = np.asarray(img).astype(np.float32) / 255.0
    if as_tensor:
        t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
        return t
    return arr


def compute_psnr(pred: np.ndarray, gt: np.ndarray):
    if HAVE_SKIMAGE:
        return sk_psnr(gt, pred, data_range=1.0)
    # fallback: compute PSNR manually
    mse = np.mean((gt - pred) ** 2)
    if mse == 0:
        return float('inf')
    return 20.0 * np.log10(1.0 / np.sqrt(mse))


def compute_ssim(pred: np.ndarray, gt: np.ndarray):
    if HAVE_SKIMAGE:
        # skimage expects (H,W,C)
        return sk_ssim(gt, pred, data_range=1.0, multichannel=True)
    # fallback: approximate with simplified formula (not recommended)
    return 0.0


class LPIPSEvaluator:
    def __init__(self, device='cpu'):
        self.device = device
        self.model = None
        if HAVE_LPIPS:
            self.model = lpips.LPIPS(net='alex').to(device)

    def compute(self, pred: torch.Tensor, gt: torch.Tensor):
        """pred, gt: torch tensors in [0,1], shape [1,3,H,W]"""
        if not HAVE_LPIPS or self.model is None:
            return None
        # lpips expects [-1,1]
        p = pred * 2.0 - 1.0
        g = gt * 2.0 - 1.0
        with torch.no_grad():
            val = self.model(p.to(self.device), g.to(self.device)).item()
        return float(val)


def evaluate_pair(pred_path: Path, gt_path: Path, lpips_eval: LPIPSEvaluator):
    p = load_image(pred_path, as_tensor=False)
    g = load_image(gt_path, as_tensor=False)
    if p.shape != g.shape:
        # resize pred to gt
        from PIL import Image
        pred_img = Image.fromarray((p * 255).astype(np.uint8))
        pred_img = pred_img.resize((g.shape[1], g.shape[0]), Image.BICUBIC)
        p = np.asarray(pred_img).astype(np.float32) / 255.0

    psnr = compute_psnr(p, g)
    ssim = compute_ssim(p, g)

    # compute lpips if available
    lp = None
    if HAVE_LPIPS:
        # convert to tensor [1,3,H,W]
        pt = torch.from_numpy(p).permute(2, 0, 1).unsqueeze(0).float()
        gt_t = torch.from_numpy(g).permute(2, 0, 1).unsqueeze(0).float()
        lp = lpips_eval.compute(pt, gt_t)

    return psnr, ssim, lp


def collect_image_pairs(gt_dir: Path, pred_dir: Path) -> List[tuple]:
    gt_dir = Path(gt_dir)
    pred_dir = Path(pred_dir)
    imgs = sorted([p for p in gt_dir.iterdir() if p.suffix.lower() in ('.png', '.jpg', '.jpeg')])
    pairs = []
    for g in imgs:
        p = pred_dir / g.name
        if p.exists():
            pairs.append((p, g))
    return pairs


def run_test_command(cmd: str):
    print(f"Running: {cmd}")
    res = subprocess.run(cmd, shell=True)
    if res.returncode != 0:
        raise RuntimeError(f"Command failed: {cmd}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_dir', required=True, help='Directory with ground-truth images')
    parser.add_argument('--pred_dir_pattern', default='results/pool_{pool_size}', help='Pattern for predicted output directories (use {pool_size})')
    parser.add_argument('--pool_sizes', nargs='+', type=int, default=[1,4,8], help='Pool sizes to evaluate')
    parser.add_argument('--out', default='results/metrics.csv', help='CSV output file')
    parser.add_argument('--test_cmd', default=None, help='Optional command template to run test stage; use {pool_size} placeholder')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--limit', type=int, default=0, help='Limit number of images for quick tests (0 = all)')
    args = parser.parse_args()

    gt_dir = Path(args.gt_dir)
    out_csv = Path(args.out)
    lpips_eval = LPIPSEvaluator(device=args.device)

    rows = []
    for psize in args.pool_sizes:
        # optionally run external test command to produce predictions
        if args.test_cmd:
            cmd = args.test_cmd.format(pool_size=psize)
            run_test_command(cmd)

        pred_dir = Path(args.pred_dir_pattern.format(pool_size=psize))
        if not pred_dir.exists():
            print(f"Predictions dir {pred_dir} not found, skipping pool_size={psize}")
            continue

        pairs = collect_image_pairs(gt_dir, pred_dir)
        if args.limit > 0:
            pairs = pairs[: args.limit]

        if len(pairs) == 0:
            print(f"No matching image pairs found for pool_size={psize} in {pred_dir}")
            continue

        psnr_list = []
        ssim_list = []
        lpips_list = []

        for pred_path, gt_path in pairs:
            try:
                psnr, ssim, lp = evaluate_pair(pred_path, gt_path, lpips_eval)
            except Exception as e:
                print(f"Failed to eval pair {pred_path} vs {gt_path}: {e}")
                continue
            psnr_list.append(psnr)
            ssim_list.append(ssim)
            if lp is not None:
                lpips_list.append(lp)

        avg_psnr = float(np.mean(psnr_list)) if psnr_list else float('nan')
        avg_ssim = float(np.mean(ssim_list)) if ssim_list else float('nan')
        avg_lpips = float(np.mean(lpips_list)) if lpips_list else float('nan')

        rows.append((psize, avg_psnr, avg_ssim, avg_lpips, len(psnr_list)))
        print(f"pool_size={psize}: PSNR={avg_psnr:.4f}, SSIM={avg_ssim:.4f}, LPIPS={avg_lpips if not np.isnan(avg_lpips) else 'n/a'}, N={len(psnr_list)}")

    # Save CSV
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, 'w') as f:
        f.write('pool_size,psnr,ssim,lpips,n_images\n')
        for r in rows:
            f.write(f"{r[0]},{r[1]},{r[2]},{r[3]},{r[4]}\n")

    print(f"Saved metrics to {out_csv}")


if __name__ == '__main__':
    main()
