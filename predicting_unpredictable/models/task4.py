"""Task 4 models."""
import torch
import torch.nn as nn


class SEBlock3D(nn.Module):
    def __init__(self, ch, reduction=8):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(ch, ch // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(ch // reduction, ch),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _, _ = x.shape
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y


class UNet3D_ResSE(nn.Module):
    def __init__(self, in_ch=5, base_ch=16):
        super().__init__()
        Block = ResSEBlock3D

        # encoder
        self.enc1 = Block(in_ch, base_ch)
        # kernel_size=(1,2,2), stride=(1,2,2)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.enc2 = Block(base_ch, base_ch * 2)
        self.pool2 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.enc3 = Block(base_ch * 2, base_ch * 4)
        self.pool3 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        # bottleneck
        self.bottleneck = Block(base_ch * 4, base_ch * 8)

        # decoder: upsample + conv
        self.up3 = nn.ConvTranspose3d(
            base_ch * 8, base_ch * 4,
            kernel_size=(1, 2, 2), stride=(1, 2, 2)
        )
        self.dec3 = Block(base_ch * 8, base_ch * 4)

        self.up2 = nn.ConvTranspose3d(
            base_ch * 4, base_ch * 2,
            kernel_size=(1, 2, 2), stride=(1, 2, 2)
        )
        self.dec2 = Block(base_ch * 4, base_ch * 2)

        self.up1 = nn.ConvTranspose3d(
            base_ch * 2, base_ch,
            kernel_size=(1, 2, 2), stride=(1, 2, 2)
        )
        self.dec1 = Block(base_ch * 2, base_ch)

        self.out_conv = nn.Conv3d(base_ch, 1, kernel_size=1)

    def forward(self, x):  # x: [B,T,C,H,W]
        # [B,C,T,H,W]
        x = x.permute(0, 2, 1, 3, 4)

        # encoder
        e1 = self.enc1(x)     # [B, base_ch, T, H,   W  ]
        p1 = self.pool1(e1)   # [B, base_ch, T, H/2, W/2]

        e2 = self.enc2(p1)    # [B, 2b,      T, H/2, W/2]
        p2 = self.pool2(e2)   # [B, 2b,      T, H/4, W/4]

        e3 = self.enc3(p2)    # [B, 4b,      T, H/4, W/4]
        p3 = self.pool3(e3)   # [B, 4b,      T, H/8, W/8]

        # bottleneck
        b = self.bottleneck(p3)  # [B, 8b, T, H/8, W/8]

        # decoder
        u3 = self.up3(b)          # [B, 4b, T, H/4, W/4]
        # e3  [B, 4b, T, H/4, W/4]， cat
        u3 = torch.cat([u3, e3], dim=1)
        d3 = self.dec3(u3)        # [B, 4b, T, H/4, W/4]

        u2 = self.up2(d3)         # [B, 2b, T, H/2, W/2]
        u2 = torch.cat([u2, e2], dim=1)
        d2 = self.dec2(u2)        # [B, 2b, T, H/2, W/2]

        u1 = self.up1(d2)         # [B, b,  T, H,   W  ]
        u1 = torch.cat([u1, e1], dim=1)
        d1 = self.dec1(u1)        # [B, b,  T, H,   W  ]

        logits = self.out_conv(d1)    # [B,1,T,H,W]
        probs = torch.sigmoid(logits)
        probs = probs.permute(0, 2, 1, 3, 4)  # -> [B,T,1,H,W]
        return probs


class ResSEBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.se = SEBlock3D(out_ch, reduction=8)
        self.skip = nn.Conv3d(
            in_ch, out_ch, kernel_size=1
            ) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        identity = self.skip(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out = out + identity
        out = self.relu(out)
        return out


class UNet3D_TimeBucket(nn.Module):
    """
    :  x [B,36,4,H,W]  [B,4,36,H,W]
    : logits [B,36,C_OUT,H,W],  C_OUT = W_BUCKETS + 1
    """

    T = 36

    FRAME_DT = 300.0

    BUCKET_DT = 10.0
    W_BUCKETS = int(FRAME_DT // BUCKET_DT)
    C_OUT = W_BUCKETS + 1
    H_SMALL, W_SMALL = 384, 384
    H_ORIG, W_ORIG = 384, 384

    def __init__(self, in_ch=4, base_ch=8, T=T, C_out=C_OUT):
        super().__init__()
        self.T = T
        self.C_out = C_out
        Block = ResSEBlock3D

        self.enc1 = Block(in_ch, base_ch)
        self.pool1 = nn.MaxPool3d((1, 2, 2))
        self.enc2 = Block(base_ch, base_ch*2)
        self.pool2 = nn.MaxPool3d((1, 2, 2))
        self.enc3 = Block(base_ch*2, base_ch*4)
        self.pool3 = nn.MaxPool3d((1, 2, 2))

        self.bottleneck = Block(base_ch*4, base_ch*8)

        self.up3 = nn.ConvTranspose3d(
            base_ch*8, base_ch*4, (1, 2, 2), (1, 2, 2))
        self.dec3 = Block(base_ch*8, base_ch*4)
        self.up2 = nn.ConvTranspose3d(
            base_ch*4, base_ch*2, (1, 2, 2), (1, 2, 2))
        self.dec2 = Block(base_ch*4, base_ch*2)
        self.up1 = nn.ConvTranspose3d(
            base_ch*2, base_ch, (1, 2, 2), (1, 2, 2))
        self.dec1 = Block(base_ch*2, base_ch)

        self.out_conv = nn.Conv3d(base_ch, C_out, kernel_size=1)

    def _to_BCTHW(self, x):
        assert x.dim() == 5, f" {x.shape}"
        B, d1, d2, H, W = x.shape
        #  [B, T, C, H, W]
        if d1 == self.T and d2 == 4:
            x = x.permute(0, 2, 1, 3, 4)  # -> [B, C, T, H, W]
        #  [B, C, T, H, W]
        elif d1 == 4 and d2 == self.T:
            pass
        else:
            raise ValueError(f"Invalid input shape: {x.shape}")
        return x

    def forward(self, x):
        x = self._to_BCTHW(x)        # [B,4,T,H,W]

        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        e3 = self.enc3(p2)
        p3 = self.pool3(e3)

        b = self.bottleneck(p3)

        u3 = self.up3(b)
        u3 = torch.cat([u3, e3], dim=1)
        d3 = self.dec3(u3)
        u2 = self.up2(d3)
        u2 = torch.cat([u2, e2], dim=1)
        d2 = self.dec2(u2)
        u1 = self.up1(d2)
        u1 = torch.cat([u1, e1], dim=1)
        d1 = self.dec1(u1)

        logits_chw = self.out_conv(d1)          # [B,C_out,T,H,W]
        logits = logits_chw.permute(0, 2, 1, 3, 4)  # [B,T,C_out,H,W]
        return logits
