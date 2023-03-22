import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import pandas as pd

from model import Model
from credit_dataset import CreditDataSet
from utils import kfold

LR = 0.01
EPOCHS = 1000
BATCH_SIZE = 128

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = CreditDataSet('train.csv')
k_fold_datasets = kfold(dataset, k=5)

model_list = []
for k_num, (train_dataset, val_dataset, split_chunk) in enumerate(k_fold_datasets):
    model = Model().to(device)
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=LR)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=split_chunk, shuffle=False)

    best_loss = float('inf')
    no_improvement = 0
    for epoch in range(EPOCHS):
        total_loss = 0
        total_acc = 0

        ### train ###
        model.train()
        for i, (x, y) in enumerate(train_loader):
            x = x.to(device)
            y = y.long().to(device)

            output = model(x)
            output = F.log_softmax(output, dim=-1)
            loss = F.nll_loss(output, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_acc += (output.argmax(dim=-1) == y).sum().item()

        train_loss = total_loss / i
        train_accuracy = total_acc / (BATCH_SIZE * i)

        ### validation ###
        model.eval()
        total_loss = 0
        for x, y in val_loader:
            x = x.to(device)
            y = y.long().to(device)

            with torch.no_grad():
                output = model(x)
                output = F.log_softmax(output, dim=-1)
        
        test_loss = F.nll_loss(output, y).item()
        test_accuracy = (output.argmax(dim=-1) == y).sum().item() / split_chunk
        
        ### save and early stopping ###
        torch.save(model.state_dict(), f'./checkpoint/model{k_num+1}_last.pt')
        if test_loss < best_loss:
            best_loss = test_loss
            no_improvement = 0
            torch.save(model.state_dict(), f'./checkpoint/model{k_num+1}_best.pt')
        else:
            no_improvement += 1
            if no_improvement >= 1000:
                break

        print(f'epoch {epoch}')
        print('train:', train_loss, train_accuracy)
        print('test:', test_loss, test_accuracy)
        print()
                
    model_list.append(model)

sub_df = pd.read_csv('sample_submission.csv')
test_dataset = CreditDataSet('test.csv', dataset.means, dataset.stds, dataset.categories, data_type='test')
test_dataset = torch.cat([data.unsqueeze(0) for data in test_dataset])
test_dataset = test_dataset.to(device)

Y = 0
for i, model in enumerate(model_list):
    with torch.no_grad():
        y = model(test_dataset)
        y = F.softmax(y)
        y = y.cpu().numpy()
        Y += y

    sub_df.loc[:, 1:] = y
    sub_df.to_csv(f'submmision_model{i+1}.csv', index=False)

sub_df.loc[:, 1:] = Y/5
sub_df.to_csv('submmision.csv', index=False)