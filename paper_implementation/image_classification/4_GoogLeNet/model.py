import torch
import torch.nn as nn
from block import Inception

class GoogLeNet(nn.Module):
    def __init__(self,classes):
        super().__init__()

        self.conv_layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=64,kernel_size=7,stride=2,padding=3),
            nn.ReLU())
        
        self.sub_sampling_layer1 = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        
        self.norm_layer1 = nn.LocalResponseNorm(2)

        self.conv_layer2=nn.Sequential(
            nn.Conv2d(in_channels=64,out_channels=192,kernel_size=3,stride=1,padding=1),
            nn.ReLU())
        
        self.conv_layer3=nn.Sequential(
            nn.Conv2d(in_channels=192,out_channels=192,kernel_size=3,stride=1,padding=1),
            nn.ReLU())

        self.norm_layer2 = nn.LocalResponseNorm(2)

        self.sub_sampling_layer2 = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)

        self.inception3a = Inception(192,64,96,128,16,32,32)
        self.inception3b = Inception(256,128,128,192,32,96,64)

        self.sub_sampling_layer3 = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)

        self.inception4a = Inception(480,192,96,208,16,48,64)
        self.inception4b = Inception(512,160,112,224,24,64,64)
        self.inception4c = Inception(512,128,128,256,24,64,64)
        self.inception4d = Inception(512,112,144,288,32,64,64)
        self.inception4e = Inception(528,256,160,320,32,128,128)

        self.sub_sampling_layer4 = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)

        self.inception5a = Inception(832,256,160,320,32,128,128)
        self.inception5b = Inception(832,384,192,384,48,128,128)

        self.avg_pooling_layer = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout2d(0.4)
        self.output_layer = nn.Linear(1024,classes)

    def forward(self,x):
        x = self.conv_layer1(x)
        x = self.sub_sampling_layer1(x)
        x = self.norm_layer1(x)
        x = self.conv_layer2(x)
        x = self.conv_layer3(x)
        x = self.norm_layer2(x)
        x = self.sub_sampling_layer2(x)
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.sub_sampling_layer3(x)
        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)
        x = self.sub_sampling_layer4(x)
        x = self.inception5a(x)
        x = self.inception5b(x)
        x = self.avg_pooling_layer(x)
        x = self.dropout(x)
        x = x.view(x.shape[0],-1)
        x = self.output_layer(x)
        return x