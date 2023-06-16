import torch
import torch.nn as nn
from block import *

class UNet(nn.Module):
    def __init__(self, num_channels, num_classes):
        super().__init__()
        
        self.convblock1 = DoubleConv(in_channels=num_channels, out_channels=64)
        self.downsampling1 = nn.MaxPool2d(kernel_size=(2, 2))

        self.convblock2 = DoubleConv(in_channels=64, out_channels=128)
        self.downsampling2 = nn.MaxPool2d(kernel_size=(2, 2))

        self.convblock3 = DoubleConv(in_channels=128, out_channels=256)
        self.downsampling3 = nn.MaxPool2d(kernel_size=(2, 2))

        self.convblock4 = DoubleConv(in_channels=256, out_channels=512)
        self.downsampling4 = nn.MaxPool2d(kernel_size=(2, 2))

        self.convblock5 = DoubleConv(in_channels=512, out_channels=1024)

        self.upsampling1 = ConvTransposeBlock(in_channels=1024, out_channels=512, 
                                              kernel_size=2, stride=2)
        self.convblock6 = DoubleConv(in_channels=1024, out_channels=512)

        self.upsampling2 = ConvTransposeBlock(in_channels=512, out_channels=256, 
                                              kernel_size=2, stride=2)
        self.convblock7 = DoubleConv(in_channels=512, out_channels=256)

        self.upsampling3 = ConvTransposeBlock(in_channels=256, out_channels=128, 
                                              kernel_size=2, stride=2)
        self.convblock8 = DoubleConv(in_channels=256, out_channels=128)

        self.upsampling4 = ConvTransposeBlock(in_channels=128, out_channels=64, 
                                              kernel_size=2, stride=2)
        self.convblock9 = DoubleConv(in_channels=128, out_channels=64)

        self.output_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        encoded1 = self.convblock1(x)
        downsampling1 = self.downsampling1(encoded1)

        encoded2 = self.convblock2(downsampling1)
        downsampling2 = self.downsampling2(encoded2)
    
        encoded3 = self.convblock3(downsampling2)
        downsampling3 = self.downsampling2(encoded3)
    
        encoded4 = self.convblock4(downsampling3)
        downsampling4 = self.downsampling2(encoded4)

        encoded5 = self.convblock5(downsampling4)

        upsampling1 = self.upsampling1(encoded5)
        skip_connection1 = torch.cat(tensors=[self.crop_tensor(encoded4, upsampling1),
                                              upsampling1],
                                              dim=1)
        decoded1 = self.convblock6(skip_connection1)

        upsampling2 = self.upsampling2(decoded1)
        skip_connection2 = torch.cat(tensors=[self.crop_tensor(encoded3, upsampling2),
                                              upsampling2],
                                              dim=1)
        decoded2 = self.convblock7(skip_connection2)
        
        upsampling3 = self.upsampling3(decoded2)
        skip_connection3 = torch.cat(tensors=[self.crop_tensor(encoded2, upsampling3),
                                              upsampling3],
                                              dim=1)
        decoded3 = self.convblock8(skip_connection3)
        
        upsampling4 = self.upsampling4(decoded3)
        skip_connection4 = torch.cat(tensors=[self.crop_tensor(encoded1, upsampling4),
                                              upsampling4],
                                              dim=1)
        decoded4 = self.convblock9(skip_connection4)
        
        output = self.output_conv(decoded4)
        return output
    
    def crop_tensor(self, x1, x2)->torch.Tensor:
        '''
        Parameters:
            x1(tensor): 인코딩구간의 텐서
            x2(tensor): 디코딩구간의 텐서
        Returns:
            tensor: x2사이즈로 자른 x1 
        '''
        _, _, h1, w1 = x1.shape
        _, _, h2, w2 = x2.shape
        dh, dw = (h1 - h2)//2, (w1 - w2)//2
        x1_cropped = x1[:, :, dh:dh+h2, dw:dw+w2]
        return x1_cropped