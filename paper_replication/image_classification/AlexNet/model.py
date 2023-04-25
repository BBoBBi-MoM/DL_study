import torch
import torch.nn as nn

class AlexNet(nn.Module):
    def __init__(self,classes):
        super().__init__()

        self.conv_layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=96,kernel_size=11,stride=4),
            nn.LocalResponseNorm(size=5,k=2),
            nn.ReLU()
        )
        
        self.sub_sampling_layer1 = nn.MaxPool2d(kernel_size=3,stride=2)

        self.conv_layer2 = nn.Sequential(
            nn.Conv2d(in_channels=96,out_channels=256,kernel_size=5,stride=1,padding=2),
            nn.LocalResponseNorm(size=5,k=2),
            nn.ReLU()
        )

        self.sub_sampling_layer2 = nn.MaxPool2d(kernel_size=3,stride=2)

        self.conv_layer3 = nn.Sequential(
            nn.Conv2d(in_channels=256,out_channels=384,kernel_size=3,stride=1,padding=1),
            nn.LocalResponseNorm(size=5,k=2),
            nn.ReLU()
        )

        self.conv_layer4 = nn.Sequential(
            nn.Conv2d(in_channels=384,out_channels=384,kernel_size=3,stride=1,padding=1),
            nn.LocalResponseNorm(size=5,k=2),
            nn.ReLU()
        )

        self.conv_layer5 = nn.Sequential(
            nn.Conv2d(in_channels=384,out_channels=256,kernel_size=3,stride=1,padding=1),
            nn.LocalResponseNorm(size=5,k=2),
            nn.ReLU()
        )

        self.sub_sampling_layer3 = nn.MaxPool2d(kernel_size=3,stride=2)
        
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=6)
        
        self.flatten = nn.Flatten()
        
        self.fc_layer1 = nn.Sequential(
            nn.Linear(256*6*6,4096),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        self.fc_layer2 = nn.Sequential(
            nn.Linear(4096,4096),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        self.fc_layer3 = nn.Sequential(
            nn.Linear(4096,classes),
            nn.Softmax(dim=-1)
        )

    def forward(self,x):
        x = self.conv_layer1(x)
        x = self.sub_sampling_layer1(x)
        x = self.conv_layer2(x)
        x = self.sub_sampling_layer2(x)
        x = self.conv_layer3(x)
        x = self.conv_layer4(x)
        x = self.conv_layer5(x)
        x = self.sub_sampling_layer3(x)
        x = self.avg_pool(x)
        x = self.flatten(x)
        x = self.fc_layer1(x)
        x = self.fc_layer2(x)
        x = self.fc_layer3(x)
        return x
