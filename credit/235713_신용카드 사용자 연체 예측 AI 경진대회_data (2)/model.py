from credit_dataset import CreditDataSet
import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearBlock(nn.Module):
    def __init__(self, in_channels, out_channels, act=True, p=0.2) -> None:
        super().__init__()

        self.linear = nn.Linear(in_channels, out_channels)
        self.act = nn.ReLU() if act else nn.Identity()
        self.dropout = nn.Dropout(p=p)

        nn.init.xavier_uniform_(self.linear.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.constant_(self.linear.bias, 0)
        self.linear = nn.utils.weight_norm(self.linear)

    def forward(self, x):
        return self.dropout(self.act(self.linear(x)))

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()

        self.fc = LinearBlock(in_channels, out_channels, act=False)
    
    def forward(self, x):
        return F.relu(self.fc(x) + x)
    
class Model(nn.Module):
    def __init__(self, col_dims=None) -> None:
        super().__init__()
        
        if col_dims is not None:
            self.embed_list = nn.ModuleList(
                [nn.Embedding(dim, 256) for dim in col_dims]
                )
        
        self.first_fc = LinearBlock(54, 128)
        self.linear_layers = nn.Sequential(
            LinearBlock(128, 256),
            ResidualBlock(256, 256),
            ResidualBlock(256, 256),

            LinearBlock(256, 128),
            ResidualBlock(128, 128),
            ResidualBlock(128, 128),
            
            LinearBlock(128, 64),
        )

        self.last_fc = nn.Linear(64, 3)

    def forward(self, x):
        x = self.first_fc(x)
        x = self.linear_layers(x)
        x = self.last_fc(x)
        return x
        

if __name__ == '__main__':
    train_dataset = CreditDataSet('train.csv')
    test_dataset = CreditDataSet('test.csv', train_dataset.means, train_dataset.stds, train_dataset.categories, data_type='test')
    model = Model()
    y = model(test_dataset[0])
    print(y.shape)