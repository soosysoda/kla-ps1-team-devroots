<!-- ANIMATED WAVING HEADER -->
<div align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=0F172A&height=200&section=header&text=AI%20Image%20Restoration&fontSize=50&fontColor=38BDF8&animation=twinkling" width="100%"/>
  
  <!-- ANIMATED TYPING SUBTITLE -->
  <a href="https://github.com/soosysoda/kla-ps1-team-devroots">
    <img src="https://readme-typing-svg.herokuapp.com?font=Fira+Code&weight=600&size=22&pause=1000&color=F472B6&center=true&vCenter=true&width=600&lines=Suppressing+Speckle+Noise...;2x+Spatial+Super-Resolution...;Optimized+for+NVIDIA+H100...;Team+DevRoots+-+KLA+Hackathon" alt="Typing SVG" />
  </a>

  <br>

  <!-- CLEAN BADGES -->
  <img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch" alt="PyTorch"/>
  <img src="https://img.shields.io/badge/Target_GPU-NVIDIA_H100-76B900?style=for-the-badge&logo=nvidia" alt="NVIDIA H100"/>
  <img src="https://img.shields.io/badge/Status-Hackathon_Ready-38BDF8?style=for-the-badge" alt="Status"/>

  <p align="center" style="font-size: 1.2rem; margin-top: 20px;">
    <b>An activation-free, single-pass restoration pipeline for semiconductor inspection.</b>
  </p>
</div>

---

## 👁️‍🗨️ Visual Results: Before & After
> *Drag the slider in your mind—our AI reconstructs microscopic edges hidden behind severe speckle noise and 2x downsampling.*

<div align="center">
  <table>
    <tr>
      <th align="center"><b>Degraded Input (Noisy & Low Res)</b></th>
      <th align="center"><b>Our AI Restored Output (Clean & High Res)</b></th>
      <th align="center"><b>Original Ground Truth</b></th>
    </tr>
    <tr>
      <!-- Replace these src links with your actual image paths in your repo -->
      <td align="center"><img src="https://placehold.co/256x256/1E293B/FFF?text=Input+Noisy+Image" width="250"/></td>
      <td align="center"><img src="https://placehold.co/256x256/064E3B/FFF?text=AI+Restored+Image" width="250"/></td>
      <td align="center"><img src="https://placehold.co/256x256/334155/FFF?text=Ground+Truth+Image" width="250"/></td>
    </tr>
  </table>
</div>

---

## 🧠 Interactive Architecture Flowchart
> **Interactive:** Click on the **Data Loader**, **NAFBlocks**, or **Loss Function** nodes below to directly view the corresponding code implementation!

```mermaid
graph TD
    %% Custom Colors and Styles
    classDef input fill:#1E293B,stroke:#38BDF8,stroke-width:2px,color:#fff,rx:5px,ry:5px
    classDef process fill:#334155,stroke:#F472B6,stroke-width:2px,color:#fff,rx:5px,ry:5px
    classDef model fill:#0F172A,stroke:#A78BFA,stroke-width:3px,color:#fff,rx:10px,ry:10px
    classDef loss fill:#4C1D95,stroke:#F87171,stroke-width:2px,color:#fff,rx:5px,ry:5px
    classDef output fill:#064E3B,stroke:#34D399,stroke-width:3px,color:#fff,rx:10px,ry:10px

    %% Flowchart Nodes (Wrapped in quotes to fix parse errors)
    A["<b>Degraded Input</b><br>Speckle/Blur/Downsampled<br><i>.npy / .png</i>"]:::input
    B["<b>Robust Data Loader</b><br>Cleans NaNs & Out-of-Range Sensor Spikes"]:::process
    
    subgraph AI Architecture [Modified NAFNet Engine]
        C["<b>Head:</b> 3x3 Convolution"]:::model
        D["<b>Body:</b> 6x Activation-Free NAFBlocks<br><i>(Gated Linear Units)</i>"]:::model
        E["<b>Tail:</b> 2x PixelShuffle Upsampling<br><i>(Super Resolution)</i>"]:::model
    end

    F["<b>GPU Execution</b><br>Mixed Precision <i>(torch.amp.autocast)</i>"]:::process
    G["<b>Hybrid Loss</b><br>SSIM Variance Clamped + Charbonnier"]:::loss
    H["<b>Restored Output</b><br>Clean High-Resolution Signal<br><i>512x512 .png</i>"]:::output

    %% Connections
    A -->|"Loads Image"| B
    B -->|"Float32 Tensor"| C
    C --> D
    D --> E
    E --> F
    
    F -->|"Training Phase"| G
    G -->|"Backpropagate"| D
    
    F -->|"Inference (H100)"| H

    %% INTERACTIVITY: Fixed Clickable Links for your specific repo
    click B "[https://github.com/soosysoda/kla-ps1-team-devroots/blob/main/evaluate.py](https://github.com/soosysoda/kla-ps1-team-devroots/blob/main/evaluate.py)" "Click to view Data Loader code"
    click D "[https://github.com/soosysoda/kla-ps1-team-devroots/blob/main/evaluate.py](https://github.com/soosysoda/kla-ps1-team-devroots/blob/main/evaluate.py)" "Click to view NAFBlock architecture"
    click G "[https://github.com/soosysoda/kla-ps1-team-devroots/blob/main/train.py](https://github.com/soosysoda/kla-ps1-team-devroots/blob/main/train.py)" "Click to view Hybrid Loss function"
