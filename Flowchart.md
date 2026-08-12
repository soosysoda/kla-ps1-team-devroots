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
