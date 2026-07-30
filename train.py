import argparse
import time
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from model import build_model
from losses import CombinedRestorationLoss
from dataset import build_dataloaders

try:
    from pytorch_msssim import ssim as ssim_metric
    HAS_MSSSIM = True
except ImportError:
    HAS_MSSSIM = False


def compute_psnr(pred, target, max_val=1.0):
    mse = torch.mean((pred - target) ** 2)
    if mse == 0:
        return torch.tensor(100.0)
    return 20 * torch.log10(max_val / torch.sqrt(mse))


@torch.no_grad()
def validate(model, val_loader, device):
    model.eval()
    total_psnr, total_ssim, n = 0.0, 0.0, 0
    for degraded, clean, _ in val_loader:
        degraded, clean = degraded.to(device), clean.to(device)
        pred = model(degraded)
        # Guard against off-by-one size mismatch from odd input dims
        if pred.shape != clean.shape:
            pred = torch.nn.functional.interpolate(pred, size=clean.shape[-2:], mode="bilinear")
        total_psnr += compute_psnr(pred, clean).item() * degraded.size(0)
        if HAS_MSSSIM:
            total_ssim += ssim_metric(pred, clean, data_range=1.0, size_average=True).item() * degraded.size(0)
        n += degraded.size(0)
    model.train()
    return total_psnr / n, (total_ssim / n if HAS_MSSSIM else float("nan"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="data/")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--width", type=int, default=32, help="model width; lower=faster")
    parser.add_argument("--lambda_lpips", type=float, default=0.0,
                         help="LPIPS perceptual loss weight (0 = off). Try 0.1 once "
                              "the basic pipeline trains cleanly — LPIPS is one of "
                              "the three graded metrics.")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--log_every", type=int, default=50)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    model = build_model(width=args.width).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params/1e6:.2f}M")

    criterion = CombinedRestorationLoss(
        lambda_ssim=0.5, lambda_fft=0.1,
        lambda_lpips=args.lambda_lpips, device=device,
    )
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    start_epoch = 0
    best_ssim = -1.0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = ckpt["epoch"] + 1
        best_ssim = ckpt.get("best_ssim", -1.0)
        print(f"Resumed from {args.resume} at epoch {start_epoch}")

    train_loader, val_loader = build_dataloaders(
        args.data_root, batch_size=args.batch_size
    )
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    #scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_start = time.time()
        running_loss = 0.0

        for i, (degraded, clean, fname) in enumerate(train_loader):
            degraded, clean = degraded.to(device), clean.to(device)

            optimizer.zero_grad()
            pred = model(degraded)

            if pred.shape != clean.shape:
                pred = torch.nn.functional.interpolate(
                    pred,
                    size=clean.shape[-2:],
                    mode="bilinear"
                )

            loss, parts = criterion(pred, clean)

            if not torch.isfinite(loss):
                print(f"\nNaN loss detected!")
                print(f"Epoch {epoch+1} Batch {i+1}")
                print(f"Files: {fname}")
                print(f"Prediction range: {pred.min().item():.6f} -> {pred.max().item():.6f}")
                print(f"Target range: {clean.min().item():.6f} -> {clean.max().item():.6f}")
                print(parts)
                break

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            running_loss += loss.item()
            if (i + 1) % args.log_every == 0:
                print(
                    f"Epoch {epoch+1}/{args.epochs} Batch {i+1}/{len(train_loader)} "
                    f"loss={loss.item():.4f} "
                    f"(char={parts['charbonnier']:.4f} ssim_l={parts['ssim_loss']:.4f} "
                    f"fft={parts['fft_loss']:.4f} lpips={parts['lpips_loss']:.4f})"
                )

        scheduler.step()
        val_psnr, val_ssim = validate(model, val_loader, device)
        elapsed = time.time() - epoch_start
        print(
            f"== Epoch {epoch+1} done in {elapsed:.1f}s | "
            f"train_loss={running_loss/len(train_loader):.4f} | "
            f"val_PSNR={val_psnr:.2f}dB val_SSIM={val_ssim:.4f} =="
        )

        state = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "val_psnr": val_psnr,
            "val_ssim": val_ssim,
            "best_ssim": best_ssim,
            "width": args.width,
        }
        torch.save(state, Path(args.checkpoint_dir) / "last_model.pt")

        if val_ssim > best_ssim:
            best_ssim = val_ssim
            state["best_ssim"] = best_ssim
            torch.save(state, Path(args.checkpoint_dir) / "best_model.pt")
            print(f"  -> New best SSIM {best_ssim:.4f}, checkpoint saved.")

    print("Training complete.")


if __name__ == "__main__":
    main()