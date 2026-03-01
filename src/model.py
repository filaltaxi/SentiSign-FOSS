"""
Emotion detection CNN model architecture.

This module contains the CNN_NeuralNet class used for facial emotion recognition.
"""

import torch
import torch.nn as nn


class CNN_NeuralNet(nn.Module):
    """ResNet-inspired CNN for emotion classification.

    Architecture includes:
    - 4 convolutional blocks (64 -> 128 -> 256 -> 512 channels)
    - 2 residual blocks with skip connections
    - MaxPooling for spatial reduction
    - Fully connected classifier layer

    Args:
        in_channels: Number of input channels (1 for grayscale, 3 for RGB)
        num_classes: Number of emotion classes to predict
        linear_in_features: Number of features after flattening (depends on input size)
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 7,
        linear_in_features: int = 2048
    ):
        super().__init__()

        def conv_block(in_ch, out_ch, pool=False):
            """Create a convolutional block with BatchNorm and ReLU."""
            layers = [
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            ]
            if pool:
                layers.append(nn.MaxPool2d(2))
            return nn.Sequential(*layers)

        # Convolutional layers
        self.conv1 = conv_block(in_channels, 64)
        self.conv2 = conv_block(64, 128, pool=True)
        self.res1 = nn.Sequential(conv_block(128, 128), conv_block(128, 128))
        self.conv3 = conv_block(128, 256, pool=True)
        self.conv4 = conv_block(256, 512, pool=True)
        self.res2 = nn.Sequential(conv_block(512, 512), conv_block(512, 512))

        # Classifier head
        self.classifier = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(linear_in_features, num_classes)
        )

    def forward(self, x):
        """Forward pass through the network.

        Args:
            x: Input tensor of shape (batch, channels, height, width)

        Returns:
            Logits tensor of shape (batch, num_classes)
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.res1(x) + x  # Residual connection
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.res2(x) + x  # Residual connection
        return self.classifier(x)
