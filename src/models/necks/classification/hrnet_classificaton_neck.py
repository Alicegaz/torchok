from typing import List, Union

import torch.nn as nn
from torch import Tensor

from src.constructor import NECKS
from src.models.base_model import BaseModel
from src.models.modules.bricks.convbnact import ConvBnAct
from src.models.modules.blocks.bottleneck import Bottleneck


@NECKS.register_class
class HRNetClassificationNeck(BaseModel):
    """HRNet neck for classification task."""
    def __init__(self, in_channels):
        """Init HRNetClassificationNeck.

        Args:
            in_channels: Input channels.
        """
        super().__init__()
        self.num_features = 2048
        self.incre_modules, self.downsamp_modules, self.final_layer = self.__make_neck(in_channels)

    def __make_neck(self, in_channels):
        head_block = Bottleneck
        self.head_channels = [32, 64, 128, 256]

        # Increasing the #channels on each resolution
        # from C, 2C, 4C, 8C to 128, 256, 512, 1024
        incre_modules = []
        for i, channels in enumerate(in_channels):
            incre_modules.append(self.__make_layer(head_block, channels, self.head_channels[i], 1, stride=1))
        incre_modules = nn.ModuleList(incre_modules)

        # downsampling modules
        downsamp_modules = []
        for i in range(len(in_channels) - 1):
            in_channels = self.head_channels[i] * head_block.expansion
            out_channels = self.head_channels[i + 1] * head_block.expansion
            downsamp_module = ConvBnAct(in_channels=in_channels,
                                        out_channels=out_channels,
                                        kernel_size=3,
                                        padding=1,
                                        stride=2)
            downsamp_modules.append(downsamp_module)
        downsamp_modules = nn.ModuleList(downsamp_modules)

        final_layer = ConvBnAct(in_channels=self.head_channels[3] * head_block.expansion,
                                out_channels=self.num_features,
                                kernel_size=1,
                                padding=0,
                                stride=1)

        return incre_modules, downsamp_modules, final_layer

    def __make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None

        if stride != 1 or inplanes != planes * block.expansion:
            downsample = ConvBnAct(in_channels=inplanes,
                                   out_channels=planes * block.expansion,
                                   kernel_size=1,
                                   padding=0,
                                   stride=stride,
                                   bias=False,
                                   act_layer=None)

        layers = [block(inplanes, planes, stride, downsample)]
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x: List[Tensor]) -> Tensor:
        """Forward method."""
        y = self.incre_modules[0](x[0])
        for i in range(len(self.downsamp_modules)):
            y = self.downsamp_modules[i](y)
            if i + 1 < len(x):
                y = self.incre_modules[i + 1](x[i + 1])
        y = self.final_layer(y)
        return y

    def get_forward_output_channels(self) -> Union[int, List[int]]:
        """Return number of output channels."""
        return self.num_features
