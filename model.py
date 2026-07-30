import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class SimplifiedChannelAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(channels, channels, 1, bias=True)

    def forward(self, x):
        return x * self.conv(self.pool(x))


class NAFBlock(nn.Module):
    def __init__(self, channels, expand=2, drop_path=0.0):
        super().__init__()
        hidden = channels * expand

        self.norm1 = nn.GroupNorm(1, channels) 
        self.conv1 = nn.Conv2d(channels, hidden, 1)
        self.dwconv = nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden)
        self.sg1 = SimpleGate()
        self.sca = SimplifiedChannelAttention(hidden // 2)
        self.conv2 = nn.Conv2d(hidden // 2, channels, 1)
        self.beta = nn.Parameter(torch.zeros((1, channels, 1, 1)))

        self.norm2 = nn.GroupNorm(1, channels)
        self.conv3 = nn.Conv2d(channels, hidden, 1)
        self.sg2 = SimpleGate()
        self.conv4 = nn.Conv2d(hidden // 2, channels, 1)
        self.gamma = nn.Parameter(torch.zeros((1, channels, 1, 1)))

        self.drop_path = drop_path

    def forward(self, x):
        y = self.norm1(x)
        y = self.conv1(y)
        y = self.dwconv(y)
        y = self.sg1(y)
        y = self.sca(y)
        y = self.conv2(y)
        x = x + y * self.beta

        y = self.norm2(x)
        y = self.conv3(y)
        y = self.sg2(y)
        y = self.conv4(y)
        x = x + y * self.gamma
        return x


class Downsample(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.op = nn.Conv2d(ch_in, ch_out, 2, stride=2)

    def forward(self, x):
        return self.op(x)


class Upsample(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.op = nn.Sequential(
            nn.Conv2d(ch_in, ch_out * 4, 1, bias=False),
            nn.PixelShuffle(2),
        )

    def forward(self, x):
        return self.op(x)


class RestorationSRNet(nn.Module):
    def __init__(self, in_ch=1, width=32, enc_blocks=(2, 2, 4), middle_blocks=4,
                 dec_blocks=(2, 2, 2)):
        super().__init__()
        self.intro = nn.Conv2d(in_ch, width, 3, padding=1)

        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        ch = width
        for n in enc_blocks:
            self.encoders.append(nn.Sequential(*[NAFBlock(ch) for _ in range(n)]))
            self.downs.append(Downsample(ch, ch * 2))
            ch *= 2

        self.middle = nn.Sequential(*[NAFBlock(ch) for _ in range(middle_blocks)])

        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for n in dec_blocks:
            self.ups.append(Upsample(ch, ch // 2))
            ch //= 2
            self.decoders.append(nn.Sequential(*[NAFBlock(ch) for _ in range(n)]))

        self.sr_head = Upsample(width, width // 2 if width >= 2 else width)
        sr_out_ch = width // 2 if width >= 2 else width
        self.outro = nn.Conv2d(sr_out_ch, in_ch, 3, padding=1)

        self.padder_size = 2 ** len(enc_blocks)

    def _pad_to_multiple(self, x):
        _, _, h, w = x.shape
        pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        return F.pad(x, (0, pad_w, 0, pad_h), mode="reflect"), h, w

    def forward(self, x):
        x, orig_h, orig_w = self._pad_to_multiple(x)
        x = self.intro(x)

        skips = []
        for enc, down in zip(self.encoders, self.downs):
            x = enc(x)
            skips.append(x)
            x = down(x)

        x = self.middle(x)

        for up, dec, skip in zip(self.ups, self.decoders, reversed(skips)):
            x = up(x)
            x = x + skip
            x = dec(x)

        x = self.sr_head(x)
        x = self.outro(x)

        x = x[:, :, : orig_h * 2, : orig_w * 2]
        return torch.sigmoid(x)  # assumes inputs/targets normalized to [0, 1]


def build_model(width=32):
    return RestorationSRNet(in_ch=1, width=width)


if __name__ == "__main__":
    model = build_model(width=32)
    dummy = torch.randn(2, 1, 128, 128)
    out = model(dummy)
    print(f"Input:  {dummy.shape}")
    print(f"Output: {out.shape}  (expected 2x spatial upscale)")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params/1e6:.2f}M")