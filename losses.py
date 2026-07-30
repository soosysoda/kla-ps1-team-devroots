import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from pytorch_msssim import ms_ssim
    HAS_MSSSIM = True
except ImportError:
    HAS_MSSSIM = False

try:
    import lpips as lpips_lib
    HAS_LPIPS = True
except ImportError:
    HAS_LPIPS = False


class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        diff = pred - target
        return torch.mean(torch.sqrt(diff * diff + self.eps * self.eps))


class FFTLoss(nn.Module):
    def forward(self, pred, target):
        pred_fft = torch.fft.rfft2(pred, norm="ortho")
        target_fft = torch.fft.rfft2(target, norm="ortho")
        pred_mag = torch.abs(pred_fft)
        target_mag = torch.abs(target_fft)
        return F.l1_loss(pred_mag, target_mag)


class CombinedRestorationLoss(nn.Module):
    def __init__(self, lambda_ssim=0.5, lambda_fft=0.1, lambda_lpips=0.0, device="cpu"):
        super().__init__()
        self.charbonnier = CharbonnierLoss()
        self.fft = FFTLoss()
        self.lambda_ssim = lambda_ssim
        self.lambda_fft = lambda_fft
        self.lambda_lpips = lambda_lpips
        if not HAS_MSSSIM:
            print(
                "[losses.py] WARNING: pytorch_msssim not installed — "
                "SSIM term will be skipped. Install with:\n"
                "  pip install pytorch-msssim --break-system-packages"
            )
        self._lpips_net = None
        if lambda_lpips > 0:
            if not HAS_LPIPS:
                print(
                    "[losses.py] WARNING: lpips not installed but lambda_lpips>0 — "
                    "LPIPS term will be skipped. Install with:\n"
                    "  pip install lpips --break-system-packages"
                )
            else:
                self._lpips_net = lpips_lib.LPIPS(net="alex").to(device)
                for p in self._lpips_net.parameters():
                    p.requires_grad = False

    def forward(self, pred, target):
        pred = torch.clamp(pred, 0.0, 1.0)
        target = torch.clamp(target, 0.0, 1.0)
        if not torch.isfinite(pred).all():
            raise RuntimeError("Prediction contains NaN or Inf")
        if not torch.isfinite(target).all():
            raise RuntimeError("Target contains NaN or Inf")
        loss_char = self.charbonnier(pred, target)
        loss_fft = self.fft(pred, target)

        if HAS_MSSSIM:
            ssim_val = ms_ssim(pred, target, data_range=1.0, size_average=True)
            if not torch.isfinite(ssim_val):
                print("MS-SSIM became NaN")
                ssim_val = torch.tensor(0.0, device=pred.device)
            loss_ssim = 1.0 - ssim_val
        else:
            loss_ssim = torch.tensor(0.0, device=pred.device)

        total = loss_char + self.lambda_ssim * loss_ssim + self.lambda_fft * loss_fft
        if not torch.isfinite(loss_char):
            print("Charbonnier loss became NaN")
        if not torch.isfinite(loss_fft):
            print("FFT loss became NaN")
        if not torch.isfinite(total):
            print("Total loss became NaN")

        loss_lpips_val = -1.0
        if self._lpips_net is not None:
            pred_3ch = pred.repeat(1, 3, 1, 1) * 2 - 1
            target_3ch = target.repeat(1, 3, 1, 1) * 2 - 1
            loss_lpips = self._lpips_net(pred_3ch, target_3ch).mean()
            total = total + self.lambda_lpips * loss_lpips
            loss_lpips_val = loss_lpips.item()

        return total, {
            "charbonnier": loss_char.item(),
            "ssim_loss": loss_ssim.item() if HAS_MSSSIM else -1.0,
            "fft_loss": loss_fft.item(),
            "lpips_loss": loss_lpips_val,
            "total": total.item(),
        }


if __name__ == "__main__":
    criterion = CombinedRestorationLoss()
    pred = torch.rand(2, 1, 256, 256)
    target = torch.rand(2, 1, 256, 256)
    total, parts = criterion(pred, target)
    print("Loss components:", parts)