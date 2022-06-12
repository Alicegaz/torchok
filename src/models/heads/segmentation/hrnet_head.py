"""
This HRNet implementation is modified from the following repository:
https://github.com/HRNet/HRNet-Semantic-Segmentation
"""
import torch.nn as nn
from torch import Tensor

from src.constructor import HEADS
from src.models.heads.base import AbstractHead


@HEADS.register_class
class HRNetSegmentationHead(AbstractHead):
    """HRNet head for segmentation tasks."""
    def __init__(self, in_features: int, num_classes: int, kernel_size: int = 1):
        """Init HRNetSegmentationHead.

        Args:
            in_features: Size of each input sample.
            num_classes: Number of classes.
            kernel_size: Kernel size.
        """
        super().__init__(in_features, num_classes)
        self.num_classes = num_classes
        self.final_conv_layer = nn.Conv2d(in_channels=in_features,
                                          out_channels=num_classes,
                                          kernel_size=kernel_size,
                                          stride=1,
                                          padding=1 if kernel_size == 3 else 0)

    def forward(self, x: Tensor) -> Tensor:
        """Forward method"""
        x = self.final_conv_layer(x)
        return x
