import torch
import torch.nn as nn


class ConvBlock(nn.Module):

    def __init__(self, input_size, output_size):
        super().__init__()
        self.conv = nn.Conv2d(
            input_size, output_size, kernel_size=3, padding=1
        )
        # self.bn = nn.BatchNorm2d(output_size)
        g = min(8, output_size)
        while output_size % g != 0:
            g -= 1
        self.norm = nn.GroupNorm(num_groups=g, num_channels=output_size)
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class EncBlock(nn.Module):

    def __init__(self, input_size, output_size):
        super().__init__()
        self.conv_block1 = ConvBlock(input_size, output_size)
        self.conv_block2 = ConvBlock(output_size, output_size)
        self.pool = nn.MaxPool2d((2, 2))

    def forward(self, x):
        h = self.conv_block1(x)
        h = self.conv_block2(h)
        p = self.pool(h)
        return h, p


class DecBlock(nn.Module):

    def __init__(self, input_size, output_size):
        super().__init__()
        self.up = nn.ConvTranspose2d(
            input_size, output_size, kernel_size=2, stride=2, padding=0
        )
        self.conv_block1 = ConvBlock(2*output_size, output_size)
        self.conv_block2 = ConvBlock(output_size, output_size)

    def forward(self, x, skip):
        h = self.up(x)
        h = torch.cat([h, skip], dim=1)
        h = self.conv_block1(h)
        h = self.conv_block2(h)
        return h


class Unet(nn.Module):

    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        # encoder
        self.e1 = EncBlock(in_channels, 32)
        self.e2 = EncBlock(32, 64)
        self.e3 = EncBlock(64, 128)

        # bottleneck
        self.b1 = ConvBlock(128, 256)
        self.b2 = ConvBlock(256, 256)

        # decoder
        self.d3 = DecBlock(256, 128)
        self.d2 = DecBlock(128, 64)
        self.d1 = DecBlock(64, 32)

        self.output = nn.Conv2d(32, out_channels, kernel_size=1, padding=0)

    def forward(self, x):
        s1, p1 = self.e1(x)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)

        x = self.b1(p3)
        x = self.b2(x)

        x = self.d3(x, s3)
        x = self.d2(x, s2)
        x = self.d1(x, s1)
        output = self.output(x)
        return output


# baseline model
class SimpleCNN(nn.Module):
    """
    VERY SIMPLE baseline CNN for Task 2 - Just ONE convolutional layer.
    Input: (batch, 2, 192, 192) - IR069 + IR107
    Output: (batch, 1, 192, 192) - VIL prediction at 192x192
    """
    def __init__(self):
        super(SimpleCNN, self).__init__()

        # Single convolutional layer: 2 input channels -> 1 output channel
        self.conv = nn.Conv2d(2, 1, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()  # Ensure output is in [0, 1]

    def forward(self, x):
        x = self.conv(x)
        x = self.sigmoid(x)
        return x
