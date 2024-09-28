# Libraries
import torch
from torch import nn
from model_blocks import FirstBlock, InnerBlock, FinalBlock


class Glyphnet(nn.Module):
    """
    PyTorch implementation of the GlyphNet classifier for hieroglyphs.

    This class defines a convolutional neural network (CNN) model for the classification
    of hieroglyphs. The network architecture includes several convolutional blocks, with configurable
    hyperparameters for the number of filters and dropout rate.

    Attributes:
        first_block (nn.Module): The initial convolutional block of the network.
        inner_blocks (nn.ModuleList): A list of intermediate convolutional blocks.
        final_block (nn.Module): The final convolutional block that produces the output.
    """
    def __init__(self, in_channels: int = 1,
                 num_classes: int = 171,
                 first_conv_out: int = 64,
                 last_sconv_out: int = 512,
                 sconv_seq_outs: tuple = (128, 128, 256, 256),
                 dropout_rate: float = 0.15):
        """
        Initializes the GlyphNet model with the specified hyperparameters.

        Args:
            in_channels (int): Number of input channels (e.g., 1 for grayscale images).
            num_classes (int): Number of classes for final classification.
            first_conv_out (int): Number of output channels in the first convolutional block.
            last_sconv_out (int): Number of output channels in the final block.
            sconv_seq_outs (tuple): Output channels for each sequential convolutional block.
            dropout_rate (float): Dropout rate for regularization in the final block.
        """
        super(Glyphnet, self).__init__()

        # Input validation
        assert in_channels > 0, "Input channels must be positive."
        assert num_classes > 0, "Number of classes must be positive."
        assert first_conv_out > 0, "First convolution output channels must be positive."
        assert last_sconv_out > 0, "Last convolution output channels must be positive."
        assert 0 <= dropout_rate <= 1, "Dropout rate must be between 0 and 1."

        # Definition of the first block of the network
        self.first_block = FirstBlock(in_channels, first_conv_out)

        # List of input sizes for inner blocks
        # Definition of the sequential intermediate blocks
        in_channels_sizes = [first_conv_out] + list(sconv_seq_outs)
        self.inner_blocks = nn.ModuleList([InnerBlock(in_channels=i, sconv_out=o) 
                                            for i, o in zip(in_channels_sizes, sconv_seq_outs)])

        # Definition of the final block that produces the output
        self.final_block = FinalBlock(in_channels=in_channels_sizes[-1], out_size=num_classes,
                                      sconv_out=last_sconv_out, dropout_rate=dropout_rate)


    def forward(self, image_input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the model.

        Args:
            image_input_tensor (torch.Tensor): The input tensor representing the image.

        Returns:
            torch.Tensor: The output of the network, corresponding to class predictions.
        """
        # Pass through the first block
        x = self.first_block(image_input_tensor)

        # Pass through the sequential intermediate blocks
        for block in self.inner_blocks:
            x = block(x)

        # Pass through the final block
        x = self.final_block(x)

        return x


def create_dummy_input(batch_size: int = 256, channels: int = 1, height: int = 100, width: int = 100) -> torch.Tensor:
    """
    Create a dummy input tensor for testing the model.

    Args:
        batch_size (int): The size of the batch.
        channels (int): The number of channels (e.g., 1 for grayscale).
        height (int): Height of the input image.
        width (int): Width of the input image.

    Returns:
        torch.Tensor: A dummy input tensor with the specified dimensions.
    """
    # (batch, channels, height, width)
    return torch.zeros((batch_size, channels, height, width))0


if __name__ == "__main__":
    """
    Main entry point for model instantiation and evaluation.

    This script initializes the Glyphnet model, prints information about
    the model's parameters, creates a dummy input tensor, performs a 
    forward pass through the model, and prints the output shape.
    """
    # Instantiate the Glyphnet model
    model = Glyphnet()

    # Print information about the number of parameters in the model
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"The proposed network has a total of {total_params:,} parameters, "
          f"of which {trainable_params:,} are trainable.")

    # Create a dummy input tensor
    # default is (256, 1, 100, 100)
    dummy_input = create_dummy_input()

    # Forward pass through the model
    result = model(dummy_input)

    # Print the shape of the output tensor
    print("Output shape:", result.shape)
