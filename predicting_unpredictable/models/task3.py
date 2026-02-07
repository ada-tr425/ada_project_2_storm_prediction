"""
Task 3 Model Architecture: 3 Layer 3D CNN.

This module defines the neural network architecture used for spatiotemporal
classification of storm events.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN3d(nn.Module):
    """
    3 dimensional CNN classifying storm events.

    Input expected: (Batch, Time, Channels, Height, Width)
    Output: (Batch, Num_Classes)
    """

    def __init__(self, num_classes: int = 8):
        super(CNN3d, self).__init__()

        # Block 1: Input 4 channels -> 16 channels
        # Conv3d args: (in_channels, out_channels, kernel_size, padding)
        # Input: (B, 4, T, 384, 384)
        self.conv1 = nn.Conv3d(4, 16, kernel_size=(3, 3, 3), padding=1)
        self.bn1 = nn.BatchNorm3d(16)
        # Pool: Reduce spatial dims by 2, temporal by 2
        self.pool1 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        # Block 2: 16 -> 32 channels
        self.conv2 = nn.Conv3d(16, 32, kernel_size=(3, 3, 3), padding=1)
        self.bn2 = nn.BatchNorm3d(32)
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        # Block 3: 32 -> 64 channels
        self.conv3 = nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=1)
        self.bn3 = nn.BatchNorm3d(64)
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        # Global Pooling: Collapses all remaining spatial/temporal dimensions
        # Output becomes (B, 64, 1, 1, 1)
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))

        # Classifier: 64 features -> num_classes
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (Batch, Time, Channel, Height, Width).
               Note: The standard PyTorch Conv3d expects
               (Batch, Channel, Time, Height, Width), so we permute inside.
        """
        # 1. Permute Dimensions
        # Input: (B, T, C, H, W) -> Needed: (B, C, T, H, W)
        x = x.permute(0, 2, 1, 3, 4)

        # 2. Convolution Blocks
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))

        # 3. Global Pooling & Flatten
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)  # Flatten to (Batch, 64)

        # 4. Classification
        logits = self.fc(x)

        return logits
