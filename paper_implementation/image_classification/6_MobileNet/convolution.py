import torch
import torch.nn as nn

class DeepwiseConv(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.deepwise_conv = nn.Conv2d(
            in_channels=in_dim,
            out_channels=out_dim,
            kernel_size=3,
            padding=1,
            groups=in_dim
        )

    def forward(self, x):
        out = self.deepwise_conv(x)
        return out
    
class PointwiseConv(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.pointwise_conv = nn.Conv2d(
            in_channels=in_dim,
            out_channels=out_dim,
            kernel_size=1,
        )
    def forward(self, x):
        out = self.pointwise_conv(x)
        return out
