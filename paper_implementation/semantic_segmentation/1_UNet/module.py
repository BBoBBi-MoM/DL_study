import torch
import torch.nn as nn
from block import *

class DownSampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(2, 2)):
        super().__init__()

        self.downsample_block = nn.Sequential(
            DoubleConv(in_channels=in_channels, out_channels=out_channels),
            nn.MaxPool2d(kernel_size=kernel_size)
        )

    def forward(self, x):
        return self.downsample_block(x)
    
class UpSampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, 
                 kernel_size=(2, 2), stride=2):
        self.upsample_block = nn.Sequential(
            ConvTransposeBlock(in_channels=in_channels, out_channels=out_channels,
                               kernel_size=kernel_size, stride=stride),
            DoubleConv(in_channels=in_channels, out_channels=out_channels)
        )

    def forward(self, x):
        return self.upsample_block(x)