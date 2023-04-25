import torch
import torch.nn as nn

class Inception(nn.Module):
    def __init__(self,in_dim,out_dim1,mid_dim2,out_dim2,mid_dim3,out_dim3,out_dim4):
        super().__init__()

        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_dim,out_dim1,kernel_size=1,stride=1),
            nn.ReLU()
        )
        self.conv3x3 = nn.Sequential(
            nn.Conv2d(in_dim,mid_dim2,kernel_size=1,stride=1),
            nn.ReLU(),
            nn.Conv2d(mid_dim2,out_dim2,kernel_size=3,stride=1,padding=1),
            nn.ReLU()
        )
        self.conv5x5 = nn.Sequential(
            nn.Conv2d(in_dim,mid_dim3,kernel_size=1,stride=1),
            nn.ReLU(),
            nn.Conv2d(mid_dim3,out_dim3,kernel_size=5,stride=1,padding=2),
            nn.ReLU()
        )
        self.max3x3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3,stride=1,padding=1),
            nn.Conv2d(in_dim,out_dim4,kernel_size=1,stride=1),
            nn.ReLU()
        )
    
    def forward(self,x):
        out1x1 = self.conv1x1(x)
        out3x3 = self.conv3x3(x)
        out5x5 = self.conv5x5(x)
        out_sub = self.max3x3(x)
        output = torch.cat([out1x1,out3x3,out5x5,out_sub],1)
        return output