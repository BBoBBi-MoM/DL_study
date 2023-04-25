import torch
import torch.nn as nn 
from block import BasicBlock, Bottleneck


# ResNet18
class ResNet18(nn.Module):
    def __init__(self,classes):
        super().__init__()
        
        self.conv_layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3 ,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
            )
        
        self.pooling_layer = nn.MaxPool2d(kernel_size=3, stride=2,padding=1)

        self.basic_block1 = nn.Sequential(
            BasicBlock(in_channels=64,out_channels=64,stride=1,downsampling=False),
            BasicBlock(in_channels=64,out_channels=64,stride=1,downsampling=False)
            )
        
        self.basic_block2 = nn.Sequential(
            BasicBlock(in_channels=64,out_channels=128,stride=2,downsampling=True),
            BasicBlock(in_channels=128,out_channels=128,stride=1,downsampling=False)
        )

        self.basic_block3 = nn.Sequential(
            BasicBlock(in_channels=128,out_channels=256,stride=2,downsampling=True),
            BasicBlock(in_channels=256,out_channels=256,stride=1,downsampling=False)
        )

        self.basic_block4 = nn.Sequential(
            BasicBlock(in_channels=256,out_channels=512,stride=2,downsampling=True),
            BasicBlock(in_channels=512,out_channels=512,stride=1,downsampling=False)
        )

        self.avg_pooling_layer = nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.output_layer = nn.Linear(512,classes)
    
    def forward(self,x):
        x = self.conv_layer1(x)
        x = self.pooling_layer(x)
        x = self.basic_block1(x)
        x = self.basic_block2(x)
        x = self.basic_block3(x)
        x = self.basic_block4(x)
        x = self.avg_pooling_layer(x)
        x = x.view(x.shape[0],-1)
        x = self.output_layer(x)
        return x

# ResNet34
class ResNet34(nn.Module):
    def __init__(self,classes):
        super().__init__()
        
        self.conv_layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3 ,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
            )
        
        self.pooling_layer = nn.MaxPool2d(kernel_size=3, stride=2,padding=1)

        self.basic_block1 = nn.Sequential(
            BasicBlock(in_channels=64,out_channels=64,stride=1,downsampling=False),
            BasicBlock(in_channels=64,out_channels=64,stride=1,downsampling=False),
            BasicBlock(in_channels=64,out_channels=64,stride=1,downsampling=False)
            )
        
        self.basic_block2 = nn.Sequential(
            BasicBlock(in_channels=64,out_channels=128,stride=2,downsampling=True),
            BasicBlock(in_channels=128,out_channels=128,stride=1,downsampling=False),
            BasicBlock(in_channels=128,out_channels=128,stride=1,downsampling=False),
            BasicBlock(in_channels=128,out_channels=128,stride=1,downsampling=False)
        )

        self.basic_block3 = nn.Sequential(
            BasicBlock(in_channels=128,out_channels=256,stride=2,downsampling=True),
            BasicBlock(in_channels=256,out_channels=256,stride=1,downsampling=False),
            BasicBlock(in_channels=256,out_channels=256,stride=1,downsampling=False),
            BasicBlock(in_channels=256,out_channels=256,stride=1,downsampling=False),
            BasicBlock(in_channels=256,out_channels=256,stride=1,downsampling=False),
            BasicBlock(in_channels=256,out_channels=256,stride=1,downsampling=False)
        )

        self.basic_block4 = nn.Sequential(
            BasicBlock(in_channels=256,out_channels=512,stride=2,downsampling=True),
            BasicBlock(in_channels=512,out_channels=512,stride=1,downsampling=False),
            BasicBlock(in_channels=512,out_channels=512,stride=1,downsampling=False),
        )

        self.avg_pooling_layer = nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.output_layer = nn.Linear(512,classes)
    
    def forward(self,x):
        x = self.conv_layer1(x)
        x = self.pooling_layer(x)
        x = self.basic_block1(x)
        x = self.basic_block2(x)
        x = self.basic_block3(x)
        x = self.basic_block4(x)
        x = self.avg_pooling_layer(x)
        x = x.view(x.shape[0],-1)
        x = self.output_layer(x)
        return x

# ResNet50
class ResNet50(nn.Module):
    def __init__(self,classes):
        super().__init__()
        
        self.conv_layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3 ,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
            )
        
        self.pooling_layer = nn.MaxPool2d(kernel_size=3, stride=2,padding=1)

        self.bottleneck_block1 = nn.Sequential(
            Bottleneck(in_channels=64,out_channels=64,stride=1,downsampling=True),
            Bottleneck(in_channels=256,out_channels=64,stride=1,downsampling=False),
            Bottleneck(in_channels=256,out_channels=64,stride=1,downsampling=False)
            )
        
        self.bottleneck_block2 = nn.Sequential(
            Bottleneck(in_channels=256,out_channels=128,stride=2,downsampling=True),
            Bottleneck(in_channels=512,out_channels=128,stride=1,downsampling=False),
            Bottleneck(in_channels=512,out_channels=128,stride=1,downsampling=False),
            Bottleneck(in_channels=512,out_channels=128,stride=1,downsampling=False)
        )

        self.bottleneck_block3 = nn.Sequential(
            Bottleneck(in_channels=512,out_channels=256,stride=2,downsampling=True),
            Bottleneck(in_channels=1024,out_channels=256,stride=1,downsampling=False),
            Bottleneck(in_channels=1024,out_channels=256,stride=1,downsampling=False),
            Bottleneck(in_channels=1024,out_channels=256,stride=1,downsampling=False),
            Bottleneck(in_channels=1024,out_channels=256,stride=1,downsampling=False),
            Bottleneck(in_channels=1024,out_channels=256,stride=1,downsampling=False),
        )

        self.bottleneck_block4 = nn.Sequential(
            Bottleneck(in_channels=1024,out_channels=512,stride=2,downsampling=True),
            Bottleneck(in_channels=2048,out_channels=512,stride=1,downsampling=False),
            Bottleneck(in_channels=2048,out_channels=512,stride=1,downsampling=False)
        )

        self.avg_pooling_layer = nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.output_layer = nn.Linear(2048,classes)
    
    def forward(self,x):
        x = self.conv_layer1(x)
        x = self.pooling_layer(x)
        x = self.bottleneck_block1(x)
        x = self.bottleneck_block2(x)
        x = self.bottleneck_block3(x)
        x = self.bottleneck_block4(x)
        x = self.avg_pooling_layer(x)
        x = x.view(x.shape[0],-1)
        x = self.output_layer(x)
        return x

# ResNet101
class ResNet101(nn.Module):
    def __init__(self,classes):
        super().__init__()
        
        self.conv_layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3 ,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
            )
        
        self.pooling_layer = nn.MaxPool2d(kernel_size=3, stride=2,padding=1)

        self.bottleneck_block1 = nn.Sequential(
            Bottleneck(in_channels=64,out_channels=64,stride=1,downsampling=True),
            Bottleneck(in_channels=256,out_channels=64,stride=1,downsampling=False),
            Bottleneck(in_channels=256,out_channels=64,stride=1,downsampling=False)
            )
        
        self.bottleneck_block2 = nn.Sequential(
            Bottleneck(in_channels=256,out_channels=128,stride=2,downsampling=True),
            Bottleneck(in_channels=512,out_channels=128,stride=1,downsampling=False),
            Bottleneck(in_channels=512,out_channels=128,stride=1,downsampling=False),
            Bottleneck(in_channels=512,out_channels=128,stride=1,downsampling=False)
        )

        block3_configuration = list()
        for iteration in range(23):
            if iteration == 0:
                block3_configuration.append(Bottleneck(in_channels=512,out_channels=256,stride=2,downsampling=True))
            else:
                block3_configuration.append(Bottleneck(in_channels=1024,out_channels=256,stride=1,downsampling=False))

        self.bottleneck_block3 = nn.Sequential(*block3_configuration)

        self.bottleneck_block4 = nn.Sequential(
            Bottleneck(in_channels=1024,out_channels=512,stride=2,downsampling=True),
            Bottleneck(in_channels=2048,out_channels=512,stride=1,downsampling=False),
            Bottleneck(in_channels=2048,out_channels=512,stride=1,downsampling=False)
        )

        self.avg_pooling_layer = nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.output_layer = nn.Linear(2048,classes)
    
    def forward(self,x):
        x = self.conv_layer1(x)
        x = self.pooling_layer(x)
        x = self.bottleneck_block1(x)
        x = self.bottleneck_block2(x)
        x = self.bottleneck_block3(x)
        x = self.bottleneck_block4(x)
        x = self.avg_pooling_layer(x)
        x = x.view(x.shape[0],-1)
        x = self.output_layer(x)
        return x

# ResNet152
class ResNet152(nn.Module):
    def __init__(self,classes):
        super().__init__()
        
        self.conv_layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3 ,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
            )
        
        self.pooling_layer = nn.MaxPool2d(kernel_size=3, stride=2,padding=1)

        self.bottleneck_block1 = nn.Sequential(
            Bottleneck(in_channels=64,out_channels=64,stride=1,downsampling=True),
            Bottleneck(in_channels=256,out_channels=64,stride=1,downsampling=False),
            Bottleneck(in_channels=256,out_channels=64,stride=1,downsampling=False)
            )
        
        self.bottleneck_block2 = nn.Sequential(
            Bottleneck(in_channels=256,out_channels=128,stride=2,downsampling=True),
            Bottleneck(in_channels=512,out_channels=128,stride=1,downsampling=False),
            Bottleneck(in_channels=512,out_channels=128,stride=1,downsampling=False),
            Bottleneck(in_channels=512,out_channels=128,stride=1,downsampling=False),
            Bottleneck(in_channels=512,out_channels=128,stride=1,downsampling=False),
            Bottleneck(in_channels=512,out_channels=128,stride=1,downsampling=False),
            Bottleneck(in_channels=512,out_channels=128,stride=1,downsampling=False),
            Bottleneck(in_channels=512,out_channels=128,stride=1,downsampling=False)
        )

        block3_configuration = list()
        for iteration in range(36):
            if iteration == 0:
                block3_configuration.append(Bottleneck(in_channels=512,out_channels=256,stride=2,downsampling=True))
            else:
                block3_configuration.append(Bottleneck(in_channels=1024,out_channels=256,stride=1,downsampling=False))

        self.bottleneck_block3 = nn.Sequential(*block3_configuration)


        self.bottleneck_block4 = nn.Sequential(
            Bottleneck(in_channels=1024,out_channels=512,stride=2,downsampling=True),
            Bottleneck(in_channels=2048,out_channels=512,stride=1,downsampling=False),
            Bottleneck(in_channels=2048,out_channels=512,stride=1,downsampling=False)
        )

        self.avg_pooling_layer = nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.output_layer = nn.Linear(2048,classes)
    
    def forward(self,x):
        x = self.conv_layer1(x)
        x = self.pooling_layer(x)
        x = self.bottleneck_block1(x)
        x = self.bottleneck_block2(x)
        x = self.bottleneck_block3(x)
        x = self.bottleneck_block4(x)
        x = self.avg_pooling_layer(x)
        x = x.view(x.shape[0],-1)
        x = self.output_layer(x)
        return x
