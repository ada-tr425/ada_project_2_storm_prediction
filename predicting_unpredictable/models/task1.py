import torch
import torch.nn as nn
import torch.nn.functional as F


class Task1CNNBaseline(nn.Module):
    def __init__(
        self,
        in_frames: int = 12,
        hidden: int = 32,
        out_frames: int = 12,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_frames, hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, out_frames, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(num_groups=8, num_channels=out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(num_groups=8, num_channels=out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x):
        return self.conv(self.pool(x))


class Up(nn.Module):
    """
    x_up: (B, in_ch, h, w) -> upsample -> concat with skip (B, skip_ch, 2h, 2w)
    """

    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(
            scale_factor=2,
            mode="bilinear",
            align_corners=False,
        )
        self.conv = DoubleConv(in_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        diffY = skip.size(2) - x.size(2)
        diffX = skip.size(3) - x.size(3)
        if diffX != 0 or diffY != 0:
            x = F.pad(
                x,
                [
                    diffX // 2,
                    diffX - diffX // 2,
                    diffY // 2,
                    diffY - diffY // 2,
                ],
            )
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.hidden_dim = hidden_dim
        self.conv = nn.Conv2d(
            input_dim + hidden_dim,
            4 * hidden_dim,
            kernel_size=kernel_size,
            padding=padding,
            bias=True,
        )

    def forward(self, x, state):
        h, c = state
        combined = torch.cat([x, h], dim=1)
        gates = self.conv(combined)
        i, f, o, g = torch.chunk(gates, 4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

    def init_state(self, b, h, w, device):
        return (
            torch.zeros(b, self.hidden_dim, h, w, device=device),
            torch.zeros(b, self.hidden_dim, h, w, device=device),
        )


class ConvLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=3):
        super().__init__()
        self.cell = ConvLSTMCell(input_dim, hidden_dim, kernel_size)

    def forward(self, x_seq):
        """
        x_seq: (B, T, C, H, W)
        return: h_T (B, hidden_dim, H, W)
        """
        b, t, c, h, w = x_seq.shape
        device = x_seq.device
        h_t, c_t = self.cell.init_state(b, h, w, device)

        for k in range(t):
            h_t, c_t = self.cell(x_seq[:, k], (h_t, c_t))
        return h_t


class BottleneckConvLSTMUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=12, base=8):
        super().__init__()
        # encoder (per-frame)
        self.inc = DoubleConv(in_channels, base)
        self.down1 = Down(base, base * 2)
        self.down2 = Down(base * 2, base * 4)
        self.down3 = Down(base * 4, base * 8)
        self.down4 = Down(base * 8, base * 16)
        self.convlstm = ConvLSTM(
            input_dim=base * 16,
            hidden_dim=base * 16,
            kernel_size=3,
        )

        # decoder
        self.up1 = Up(base * 16, base * 8, base * 8)
        self.up2 = Up(base * 8, base * 4, base * 4)
        self.up3 = Up(base * 4, base * 2, base * 2)
        self.up4 = Up(base * 2, base, base)
        self.outc = nn.Conv2d(base, out_channels, kernel_size=1)

    def encode_one(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        return x1, x2, x3, x4, x5

    def forward(self, x_seq):

        b, t, c, h, w = x_seq.shape
        x1, x2, x3, x4, _ = self.encode_one(x_seq[:, -1])
        bottlenecks = []
        for k in range(t):
            _, _, _, _, x5 = self.encode_one(x_seq[:, k])
            bottlenecks.append(x5)
        x5_seq = torch.stack(bottlenecks, dim=1)  # (B,T,base*16,24,24)

        hT = self.convlstm(x5_seq)  # (B,base*16,24,24)

        x = self.up1(hT, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)


__all__ = ["Task1CNNBaseline", "BottleneckConvLSTMUNet"]
