# Libraries
import torch
import torch.nn as nn
from torch import Tensor


class ATCNet(nn.Module):
    def __init__(self, n_classes: int) -> None:
        """
        Initializes the architecture of the ATCNet model.

        Args:
            n_classes (int): Number of classes for classification.
        """
        super(ATCNet, self).__init__()

        # Define the layers of the model
        self.conv1 = self._conv_block(1, 64)
        self.conv2 = self._conv_block(64, 64)
        self.sepconv1 = self._separable_conv_block(64, 128)
        self.sepconv2 = self._separable_conv_block(128, 128)
        self.sepconv3 = self._separable_conv_block(128, 256)
        self.sepconv4 = self._separable_conv_block(256, 256)
        self.exit_block = self._conv_block(256, 512, last=True)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.15)
        self.fc = nn.Linear(512, n_classes)


    def _conv_block(self, in_channels: int, out_channels: int, last: bool = False) -> nn.Sequential:
        """
        Create a convolutional block with BatchNorm and ReLU.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            last (bool): Whether this is the last block (no pooling if True).

        Returns:
            nn.Sequential: A sequential container of layers forming the convolutional block.
        """
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        if not last:
            layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        return nn.Sequential(*layers)


    def _separable_conv_block(self, in_channels: int, out_channels: int) -> nn.Sequential:
        """
        Create a separable convolution block.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.

        Returns:
            nn.Sequential: A sequential container of layers forming the separable convolution block.
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False, groups=in_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False, groups=out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )


    def forward(self, x: Tensor) -> Tensor:
        """
        Defines the forward pass of the model.

        Args:
            x (Tensor): Input tensor with shape (batch_size, 1, height, width).

        Returns:
            Tensor: Model logits with shape (batch_size, n_classes).
        """
        # Apply first convolutional layer, batch norm, ReLU, and pooling
        x = self.conv1(x)
        # Apply second convolutional layer, batch norm, ReLU, and pooling
        x = self.conv2(x)
        # Pass through the first separable convolution block
        x = self.sepconv1(x)
        # Pass through the second separable convolution block
        x = self.sepconv2(x)
        # Pass through the third separable convolution block
        x = self.sepconv3(x)
        # Pass through the fourth separable convolution block
        x = self.sepconv4(x)
        # Pass through the exit block
        x = self.exit_block(x)
        # Apply global average pooling
        x = self.global_avg_pool(x)
        # Flatten the tensor
        x = torch.flatten(x, 1)
        # Apply dropout for regularization
        x = self.dropout(x)
        # Pass through the fully connected layer to get logits
        x = self.fc(x)
        return x


if __name__ == "__main__":
    model = ATCNet(n_classes=10)
    dummy_input = torch.randn(8, 1, 224, 224)
    output = model(dummy_input)
    print("Output shape:", output.shape)
