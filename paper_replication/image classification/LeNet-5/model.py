import torch
import torch.nn as nn

class LeNet(nn.Module):
    def __init__(self,classes):
        super(LeNet,self).__init__()

        self.conv_layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=6,kernel_size=5,stride=1),
            nn.Sigmoid()
        )

        self.sub_sampling_layer1 = nn.AvgPool2d(kernel_size=2,stride=2)

        self.conv_layer2 = nn.Sequential(
            nn.Conv2d(in_channels=6,out_channels=16,kernel_size=5,stride=1),
            nn.Sigmoid()
        )

        self.sub_sampling_layer2 = nn.AvgPool2d(kernel_size=2,stride=2)

        self.conv_layer3 = nn.Sequential(
            nn.Conv2d(in_channels=16,out_channels=120,kernel_size=5,stride=1),
            nn.Sigmoid()
        )

        self.flatten = nn.Flatten()

        
        self.fc_layer1 = nn.Sequential(
            nn.Linear(120,84),
            nn.Tanh()
            )
        
        self.output_layer = nn.Linear(84,classes)
            

    def forward(self,x):
        x = self.conv_layer1(x)
        x = self.sub_sampling_layer1(x)
        x = self.conv_layer2(x)
        x = self.sub_sampling_layer2(x)
        x = self.conv_layer3(x)
        x = self.flatten(x)
        x = self.fc_layer1(x)
        x = self.output_layer(x)
        return x