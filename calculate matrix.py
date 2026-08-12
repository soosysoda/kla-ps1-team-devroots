import os
import glob
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import pandas as pd
import lpips

# Initialize LPIPS model (VGG network backbone)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
lpips_fn = lpips.LPIPS(net='vgg').to(device)

def load_float_img(path):
    if path.endswith(".npy"):
        img = np.load(path)
    else:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0
    return np.clip(np.nan_to_num(img, nan=0.0), 0.0, 1.0)

def compute_ssim_tensor(t1, t2, window_size=11):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    mu1 = F.avg_pool2d(t1, window_size, stride=1, padding=window_size // 2)
    mu2 = F.avg_pool2d(t2, window_size, stride=1, padding=window_size // 2)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.avg_pool2d(t1 * t1, window_size, stride=1, padding=window_size // 2) - mu1_sq
    sigma2_sq = F.avg_pool2d(t2 * t2, window_size, stride=1, padding=window_size // 2) - mu2_sq
    sigma12 = F.avg_pool2d(t1 * t2, window_size, stride=1, padding=window_size // 2) - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2) + 1e-7)
    return ssim_map.mean().item()

def evaluate_all(restored_dir, gt_dir, output_excel="restoration_metrics.xlsx"):
    valid_exts = ('*.png', '*.jpg', '*.tif', '*.npy')
    restored_paths = []
    for ext in valid_exts:
        restored_paths.extend(glob.glob(os.path.join(restored_dir, ext)))
    restored_paths = sorted(restored_paths)

    records = []
    print(f"Evaluating {len(restored_paths)} image pairs...")

    for r_path in restored_paths:
        fname = os.path.basename(r_path)
        base_name = os.path.splitext(fname)[0]
        
        # Match with GT image
        gt_matches = glob.glob(os.path.join(gt_dir, f"{base_name}.*"))
        if not gt_matches:
            continue
        g_path = gt_matches[0]

        # Load images
        r_img = load_float_img(r_path)
        g_img = load_float_img(g_path)

        # 1. PSNR Calculation
        mse = np.mean((r_img - g_img) ** 2)
        psnr_val = 100.0 if mse == 0 else 20 * np.log10(1.0 / np.sqrt(mse))

        # Tensors for SSIM & LPIPS
        r_tensor = torch.from_numpy(r_img).unsqueeze(0).unsqueeze(0).float().to(device)
        g_tensor = torch.from_numpy(g_img).unsqueeze(0).unsqueeze(0).float().to(device)

        # 2. SSIM Calculation
        ssim_val = compute_ssim_tensor(r_tensor, g_tensor)

        # 3. LPIPS Calculation (Requires 3 channels in range [-1, 1])
        r_lpips_in = r_tensor.repeat(1, 3, 1, 1) * 2.0 - 1.0
        g_lpips_in = g_tensor.repeat(1, 3, 1, 1) * 2.0 - 1.0
        with torch.no_grad():
            lpips_val = lpips_fn(r_lpips_in, g_lpips_in).item()

        records.append({
            "File_Name": fname,
            "PSNR_dB": round(psnr_val, 4),
            "SSIM": round(ssim_val, 4),
            "LPIPS": round(lpips_val, 4)
        })

    df = pd.DataFrame(records)
    
    # Calculate Averages
    avg_row = {
        "File_Name": "AVERAGE",
        "PSNR_dB": round(df["PSNR_dB"].mean(), 4),
        "SSIM": round(df["SSIM"].mean(), 4),
        "LPIPS": round(df["LPIPS"].mean(), 4)
    }
    df_with_avg = pd.concat([df, pd.DataFrame([avg_row])], ignore_index=True)

    # Export to Excel & CSV
    df_with_avg.to_excel(output_excel, index=False)
    df_with_avg.to_csv("restoration_metrics.csv", index=False)

    print("\n--- SUMMARY METRICS FOR SLIDE 6 ---")
    print(f"Average PSNR  : {avg_row['PSNR_dB']} dB")
    print(f"Average SSIM  : {avg_row['SSIM']}")
    print(f"Average LPIPS : {avg_row['LPIPS']}")
    print(f"\nFull breakdown saved to '{output_excel}' and 'restoration_metrics.csv'.")

if __name__ == "__main__":
    RESTORED_FOLDER = r"C:\Users\VICTUS\Downloads\semicon_2k26\restored_outputs"
    GT_FOLDER = r"C:\Users\VICTUS\Downloads\semicon_2k26\train\train\GT"
    
    evaluate_all(RESTORED_FOLDER, GT_FOLDER)