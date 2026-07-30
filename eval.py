import argparse
import time
from pathlib import Path

import numpy as np
import torch

from model import build_model

VALID_EXT = (".npy")


def load_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    width = ckpt.get("width", 32)
    model = build_model(width=width).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def load_image(path):
    if path.suffix.lower() == ".npy":
        arr = np.load(path).astype(np.float32)
        arr = np.squeeze(arr)

        if arr.max() > 2.0 or arr.min() < -0.5:
            arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)

        return arr


def save_image(arr, path):
    arr = np.clip(arr, 0.0, 1.0).astype(np.float32)
    np.save(path.with_suffix(".npy"), arr)


def compute_metrics(pred, target):
    mse = np.mean((pred - target) ** 2)
    psnr = 100.0 if mse == 0 else 20 * np.log10(1.0 / np.sqrt(mse))

    try:
        from skimage.metrics import structural_similarity as sk_ssim
        ssim_val = sk_ssim(pred, target, data_range=1.0)
    except ImportError:
        ssim_val = float("nan")

    try:
        import lpips
        if not hasattr(compute_metrics, "_lpips_model"):
            compute_metrics._lpips_model = lpips.LPIPS(net="alex")
        p = torch.from_numpy(pred).float().unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1) * 2 - 1
        t = torch.from_numpy(target).float().unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1) * 2 - 1
        lpips_val = compute_metrics._lpips_model(p, t).item()
    except ImportError:
        lpips_val = float("nan")

    return {"psnr": psnr, "ssim": ssim_val, "lpips": lpips_val}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True,
                         help="Directory of degraded test images")
    parser.add_argument("--output_dir", type=str, required=True,
                         help="Directory to write restored images")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt")
    parser.add_argument("--gt_dir", type=str, default=None,
                         help="Optional: ground-truth dir for self-evaluation metrics")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    model = load_model(args.checkpoint, device)
    print(f"Loaded checkpoint: {args.checkpoint}")

    filenames = sorted(
        f.name for f in input_dir.iterdir() if f.suffix.lower() in VALID_EXT
    )
    if not filenames:
        raise FileNotFoundError(f"No valid images found in {input_dir}")
    print(f"Found {len(filenames)} images to process.")

    use_amp = device.type == "cuda"

    dummy = torch.zeros(1, 1, 128, 128, device=device)
    with torch.inference_mode():
        with torch.autocast(device_type="cuda", enabled=use_amp):
            _ = model(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()

    all_metrics = []
    inference_times = []

    for fname in filenames:
        degraded = load_image(input_dir / fname)
        inp = torch.from_numpy(degraded).unsqueeze(0).unsqueeze(0).float().to(device)

        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        with torch.inference_mode():
            with torch.autocast(device_type="cuda", enabled=use_amp):
                pred = model(inp)

        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        inference_times.append(t1 - t0)

        pred_np = pred.squeeze().float().cpu().numpy()
        save_image(pred_np, output_dir / Path(fname).stem)

        if args.gt_dir is not None:
            gt_path = Path(args.gt_dir) / (Path(fname).stem + ".npy")
            if gt_path.exists():
                target = load_image(gt_path)
                if target.shape != pred_np.shape:
                    print(f"  [WARN] shape mismatch for {fname}, skipping metrics")
                else:
                    m = compute_metrics(pred_np, target)
                    all_metrics.append(m)

    avg_time_ms = np.mean(inference_times) * 1000
    print(f"\nProcessed {len(filenames)} images.")
    print(f"Average inference time: {avg_time_ms:.2f} ms/image "
          f"({1000/avg_time_ms:.1f} images/sec)")

    if all_metrics:
        avg_psnr = np.mean([m["psnr"] for m in all_metrics])
        avg_ssim = np.nanmean([m["ssim"] for m in all_metrics])
        avg_lpips = np.nanmean([m["lpips"] for m in all_metrics])
        print(f"\n--- Self-evaluation metrics (n={len(all_metrics)}) ---")
        print(f"  PSNR:  {avg_psnr:.2f} dB")
        print(f"  SSIM:  {avg_ssim:.4f}")
        print(f"  LPIPS: {avg_lpips:.4f}  (lower is better)")
    else:
        print("(No --gt_dir provided, or no matching ground-truth files — "
              "skipping quality metrics. This is expected on the real blind test set.)")

    print(f"\nRestored images written to: {output_dir}")


if __name__ == "__main__":
    main()