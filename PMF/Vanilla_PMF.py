import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import argparse
import os
import sys
from pathlib import Path

print(os.getcwd())
sys.path.append(os.getcwd())

parser = argparse.ArgumentParser()
parser.add_argument("-lam_u", "--lambda_u", type=float, default=0.005)
parser.add_argument("-lam_v", "--lambda_v", type=float, default=0.005)

args = parser.parse_args()

# loader Dataset
col_names = ['user_id', 'movie_id', 'rating', 'timestamp']
df = pd.read_csv('../Data/MovieLens/ml-100k/u.data', sep='\t', names=col_names)
train_df = pd.read_csv('../Data/MovieLens/ml-100k/ua.base', sep='\t', names=col_names)
test_df = pd.read_csv('../Data/MovieLens/ml-100k/ua.test', sep='\t', names=col_names)

n_users = len(df.loc[:,'user_id'].unique())
n_items = len(df.loc[:,'movie_id'].unique())

# R matrix
R = torch.zeros((n_users, n_items))
for user_id, movie_id, rating, timestamp in train_df.values:
    R[user_id-1, movie_id-1] = rating

# R_test Matrix
R_test = torch.zeros((n_users, n_items))
for user_id, movie_id, rating, timestamp in test_df.values:
    R_test[user_id-1, movie_id-1] = rating

class PandasDataset(Dataset):
    
    def __init__(self, dataset):
        super(PandasDataset, self).__init__()
        self.X = dataset.iloc[ : , [0,1]]
        self.y = dataset.iloc[ : , 2]
        self.X_value, self.y_value = self.X.values, self.y.values
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return{
            'X' : torch.from_numpy(self.X_value)[idx],
            'y' : torch.from_numpy(self.y_value)[idx]
        }

batch_size = 1000

train_dataset = PandasDataset(train_df)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# device
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
torch.cuda.is_available()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# number of latnet factor
k = 10

class PMF(nn.Module):
    def __init__(self):
        super(PMF, self).__init__()
        
        self.U = nn.Parameter(nn.init.normal_(torch.zeros(k, n_users), std=1.0/k), requires_grad = True) # k x n_users
        self.V = nn.Parameter(nn.init.normal_(torch.zeros(k, n_items), std=1.0/k), requires_grad = True) # k x n_items
    
    def forward(self):
        
        output = torch.mm(self.U.T, self.V) # n_users x n_items
        reg_user = torch.norm(self.U)
        reg_items = torch.norm(self.V)
    
        return output, reg_user, reg_items

model = PMF()
model = model.to(device)

epochs = 100
lambda_u = args.lambda_u
lambda_v = args.lambda_v

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(epochs):
    total_loss = 0
    model.train()
    for data in train_loader: # batch x 2
        
        # predict
        R_hat = model()[0]
        reg_user = model()[1]
        reg_items = model()[2]
        
        # loss
        loss = torch.norm(R[R!=0].to(device)-R_hat[R!=0]) + (lambda_u/2)*reg_user + (lambda_u/2)*reg_items
        
        # initialize
        optimizer.zero_grad()
        
        # calculate gradient
        loss.backward()
        
        # update
        optimizer.step()
        
        total_loss += loss.item()
        
    obj = total_loss / len(train_loader)
    
    #evaluation
    model.eval()
    
    R_hat = model()[0]
    SSE = torch.norm(R_test[R_test!=0].to(device) - R_hat[R_test!=0])
    RMSE = torch.sqrt(SSE / R_test.shape[0])
    
    if (epoch+1) % 10 == 0:
        print("epoch : {}, obj : {}, RMSE : {}".format(epoch+1, obj, RMSE))