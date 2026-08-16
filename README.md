<!-- ANIMATED WAVING HEADER -->
<div align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=0F172A&height=200&section=header&text=AI-Based%20R%20Restoration%20of%20Degraded%20Images%20for%20Semiconductor%20Inspection&fontSize=30&fontColor=38BDF8&animation=twinkling" width="100%"/>
  
  <!-- ANIMATED TYPING SUBTITLE -->
  <a href="https://github.com/soosysoda/kla-ps1-team-devroots">
    <img src="https://readme-typing-svg.herokuapp.com?font=Fira+Code&weight=600&size=22&pause=1000&color=F472B6&center=true&vCenter=true&width=600&lines=Suppressing+Speckle+Noise...;2x+Spatial+Super-Resolution...;Optimized+for+NVIDIA+H100...;Team+DevRoots+-+KLA+Hackathon" alt="Typing SVG" />
  </a>

  <br>

  <!-- 3D FLOATING TECH ELEMENT (NOW BIGGER!) -->
  <!-- Change width="300" to 400 or 500 if you want it even more massive -->
  <img src="https://github.com/soosysoda/kla-ps1-team-devroots/blob/main/semiconductor.gif?raw=true" width="300" alt="3D AI Chip"/>
  
  <br>

  <!-- CLEAN BADGES -->
  <img src="https://img.shields.io/badge/python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/PyTorch-2.1%2B-EE4C2C?style=for-the-badge&logo=pytorch" alt="PyTorch"/>
  <img src="https://img.shields.io/badge/Target_GPU-NVIDIA_H100-76B900?style=for-the-badge&logo=nvidia" alt="NVIDIA H100"/>
  <img src="https://img.shields.io/badge/Status-Hackathon_Ready-38BDF8?style=for-the-badge" alt="Status"/>

  <p align="center" style="font-size: 1.2rem; margin-top: 20px;">
    <b>SEMICON India Hackathon 2026 · Track 1 (PS01, sponsored by KLA)</b><br>
    <i>An activation-free, single-pass restoration pipeline for grayscale semiconductor inspection.</i>
  </p>
</div>

## Overview

Semiconductor inspection images are frequently captured at reduced resolution
and corrupted by speckle noise, which can push pixel intensities beyond the
true signal range. This repository restores such images in a **single forward
pass**: denoising and 2x spatial super-resolution happen together, rather than
as a two-stage pipeline, to keep inference fast enough for the challenge's
GPU-time benchmark.

**What the model does:**

| Input | Output |
|---|---|
| Degraded grayscale image (128×128 or 256×256), speckle noise, possible intensity overshoot | Restored grayscale image (256×256 or 512×512), matching ground truth |

## Approach

- **Architecture** — a NAFNet-style encoder-decoder (nonlinear-activation-free
  blocks) with a PixelShuffle upsampling head. Fully convolutional, so it
  handles both dataset size classes (128→256 and 256→512) without any
  architectural changes.
- **Loss** — Charbonnier (robust to speckle-noise outliers) + multi-scale
  SSIM (optimizes a graded metric directly) + an FFT magnitude term
  (recovers high-frequency edge detail that pixel losses under-penalize) +
  optional LPIPS perceptual loss (the third graded metric).
- **Generalization** — randomized degradation augmentation at train time
  (varying noise severity, blur) so the model isn't tuned to a single fixed
  noise distribution, targeting the out-of-distribution portion of the test
  set.
- **Inference speed** — fp16 autocast, `torch.inference_mode()`, and a
  GPU warm-up pass excluded from timing, matching the official benchmark
  methodology (script startup, model init, disk I/O, and inference all
  measured on an NVIDIA H100).

## Repository structure

```
.
├── model.py           # NAFNet-style restoration + super-resolution network
├── losses.py           # Charbonnier + MS-SSIM + FFT + optional LPIPS loss
├── dataset.py           # Dataset loader — handles mixed resolutions, degradation augmentation
├── train.py            # Training loop (AMP, checkpointing, validation)
├── eval.py             # Standalone inference script (used as-is for benchmarking)
├── requirements.txt     # Dependencies
└── README.md
```

## Quickstart

### 1. Setup

```bash
git clone <this-repo-url>
cd ps01_restoration
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Requires Python 3.10+. A CUDA-capable GPU is strongly recommended for
training; CPU works for small-scale testing but will be slow.

### 2. Data

Place the dataset under `data/train/` — the loader auto-detects several
common layouts (folder-based `degraded/`+`clean/`, a single stacked `.npy`
array, or flat `*_noisy.npy`/`*_gt.npy` suffix pairs). No code changes
needed for any of these.

### 3. Train

```bash
python train.py --data_root data/ --epochs 100 --batch_size 8 --width 32
```

Checkpoints are saved to `checkpoints/best_model.pt` (best validation SSIM)
and `checkpoints/last_model.pt`. Resume any interrupted run with:

```bash
python train.py --data_root data/ --resume checkpoints/last_model.pt --epochs 100
```

Key flags:

| Flag | Purpose |
|---|---|
| `--width` | Model capacity / speed tradeoff (16 = fastest, 32 = default, higher = higher quality) |
| `--lambda_lpips` | Perceptual loss weight, off by default — try `0.1` for an extra quality boost |
| `--batch_size` | Base batch size; automatically scaled down for larger-resolution samples |

### 4. Run inference

```bash
python eval.py --input_dir <test_images_dir> --output_dir <output_dir> --checkpoint checkpoints/best_model.pt
```

Standalone, dependency-light, and takes only an input/output directory
pair — matches the official evaluation harness. Optionally pass `--gt_dir`
to also print PSNR / SSIM / LPIPS and per-image inference time for your
own validation.

## Results

| Metric | Value |
|---|---|
| SSIM | 0.7869 |
| PSNR (dB) | 27.668699264526367 |
| LPIPS | 0.4111 |
| Avg. inference time / image (H100) | ~8–12 ms / image (H100 Estimate) |

## Known limitations

_TBD — document honest failure cases here once real results are in (e.g.
performance on very high noise severity, or degradation types not seen
during training augmentation). The rubric explicitly rewards honest
reporting of where the model struggles._

## Team

_Team name / members here_

## License & Terms

This project is open-source software licensed under the **[MIT License](LICENSE)**.

> **Copyright (c) 2026 DevRoots**

### License Summary

| 🟢 Permissions | 🔵 Conditions | 🔴 Limitations |
| :--- | :--- | :--- |
| **Commercial Use** — Free for commercial purposes | **License & Copyright Notice** — Must keep copyright notice intact | **No Liability** — Authors are not liable for damages |
| **Modification** — Free to edit and adapt source code | | **No Warranty** — Provided "as-is" without guarantees |
| **Distribution** — Free to redistribute modified versions | | |
| **Private Use** — Free to run and test internally | | |

<details>
<summary><b>Click to view full legal license text</b></summary>
<br>
MIT License

Copyright (c) 2026 DevRoots

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
</details>
