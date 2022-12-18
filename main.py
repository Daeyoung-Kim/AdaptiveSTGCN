import logging
import os
import argparse
import math
import random
import tqdm
import numpy as np
import pandas as pd
from sklearn import preprocessing

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils as utils


from torch.utils.data import DataLoader

from script import dataloader, utility, earlystopping
from model import models

def set_env(seed):
    # Set available CUDA devices
    # This option is crucial for an multi-GPU device
    os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    # os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['PYTHONHASHSEED']=str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)

# set arg

def set_parameters():

    args = dict()

    args['enable_cuda'] = True
    args['seed'] = 42
    args['n_his'] = 7
    args['n_pred'] = 7
    args['Kt'] = 2
    args['act_func'] = 'glu'
    args['Ks'] = 3
    args['graph_conv_type'] = 'cheb_graph_conv'
    args['gso_type'] = 'sym_norm_lap'
    args['enable_bias'] = True
    args['droprate'] = 0.5
    args['lr'] = 0.001
    args['weight_decay_rate'] = 0.0005
    args['batch_size'] = 5
    args['epochs'] = 400
    args['opt'] = 'adam'
    args['step_size'] = 10
    args['gamma'] = 0.95
    args['patience'] = 30

    set_env(args['seed'])

    # Running in Nvidia GPU (CUDA) or CPU
    if args['enable_cuda'] and torch.cuda.is_available():
        # Set available CUDA devices
        # This option is crucial for multiple GPUs
        # 'cuda' ≡ 'cuda:0'
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    Ko = args['n_his']- (args['Kt'] - 1) * 2 * 2

    # blocks: settings of channel size in st_conv_blocks and output layer,
    # using the bottleneck design in st_conv_blocks
    blocks = []
    blocks.append([1])
    for l in range(2):
        blocks.append([64, 16, 64])
    blocks.append([32]) # for adaptive layer
    blocks.append([64]) # for adaptive layer
    if Ko == 0:
        blocks.append([128])
    elif Ko > 0:
        blocks.append([128, 128])
    blocks.append([args['n_pred']])
    
    return args, device, blocks

args, device, blocks = set_parameters()

# Get data and preprocess
adj4 = np.array(pd.read_csv('./Data/Matrix_base/AdjMat04.csv',
                            index_col = 0).values).astype(dtype=np.float32)
adj5 = np.array(pd.read_csv('./Data/Matrix_base/AdjMat05.csv',
                            index_col = 0).values).astype(dtype=np.float32)
adj6 = np.array(pd.read_csv('./Data/Matrix_base/AdjMat06.csv',
                            index_col = 0).values).astype(dtype=np.float32)

n_vertex = adj5.shape[1]

def to_gso(adj):
    gso = utility.calc_chebynet_gso(adj)
    gso = gso.toarray()
    gso = gso.astype(dtype=np.float32)
    gso = torch.from_numpy(gso).to(device)
    
    return gso

gso4 = to_gso(adj4)
gso5 = to_gso(adj5)
gso6 = to_gso(adj6)

FeatureMat4 = pd.read_csv('./Data/Matrix_base/FeatureMat04.csv').groupby('역명').sum()
FeatureMat5 = pd.read_csv('./Data/Matrix_base/FeatureMat05.csv').groupby('역명').sum()

FM4 = FeatureMat4.transpose().astype(float)
FM5 = FeatureMat5.transpose().astype(float)

train4 = FM4.iloc[:30,:].values
test4 = FM4.iloc[30:,:].values

zscore = preprocessing.StandardScaler()

train4 = zscore.fit_transform(train4)
test4 = zscore.transform(test4)

train5 = FM5.iloc[:31,:].values
test5 = FM5.iloc[31:,:].values

zscore = preprocessing.StandardScaler()

train5 = zscore.fit_transform(train5)
test5 = zscore.transform(test5)

x_train4 = torch.tensor(train4)
x_test4 = torch.tensor(test4)

x_train5 = torch.tensor(train5)
x_test5 = torch.tensor(test5)

# Time step
T = 7 

tensor_train4 = torch.rand(x_train4.shape[0]-T+1,T,x_train4.shape[1])
tensor_test4 = torch.rand(x_test4.shape[0]-T+1,T,x_test4.shape[1])

tensor_train5 = torch.rand(x_train5.shape[0]-T+1,T,x_train5.shape[1])
tensor_test5 = torch.rand(x_test5.shape[0]-T+1,T,x_test5.shape[1])

for i in range(0,x_train4.shape[0]-T+1):
    tensor_train4[i,:,:] = x_train4[i:(i+T),:]
    tensor_test4[i,:,:] = x_test4[i:(i+T),:]
  
for i in range(0,x_train5.shape[0]-T+1):
    tensor_train5[i,:,:] = x_train5[i:(i+T),:]
    tensor_test5[i,:,:] = x_test5[i:(i+T),:]
    
tensor_train4 = torch.reshape(tensor_train4,[24,1,7,276])
tensor_test4 = torch.reshape(tensor_test4,[24,1,7,276])

tensor_train5 = torch.reshape(tensor_train5,[25,1,7,276])
tensor_test5 = torch.reshape(tensor_test5,[25,1,7,276])

train_data4 = utils.data.TensorDataset(tensor_train4, tensor_test4)

train_iter4 = utils.data.DataLoader(dataset=train_data4, batch_size=args['batch_size'], shuffle=False)

# train model

def prepare_model(args, blocks, n_vertex):
    loss = nn.MSELoss()

    if args['graph_conv_type'] == 'cheb_graph_conv':
        model = models.STGCNChebGraphConv(args, blocks, n_vertex).to(device)
    else:
        model = models.STGCNGraphConv(args, blocks, n_vertex).to(device)

    if args['opt'] == "rmsprop":
        optimizer = optim.RMSprop(model.parameters(), lr=args['lr'], weight_decay=args['weight_decay_rate'])
    elif args['opt'] == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args['lr'], weight_decay=args['weight_decay_rate'], amsgrad=False)
    elif args['opt'] == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=args['lr'], weight_decay=args['weight_decay_rate'], amsgrad=False)
    else:
        raise NotImplementedError(f'ERROR: The optimizer is not implemented.')

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args['step_size'], gamma=args['gamma'])

    return loss, model, optimizer, scheduler

def train(loss, args, optimizer, scheduler, model, train_iter, gso, adp_gso):
    for epoch in range(args['epochs']):
        l_sum, n = 0.0, 0  # 'l_sum' is epoch sum loss, 'n' is epoch instance number
        model.train()
        for x, y in tqdm.tqdm(train_iter):
            y_pred = model(x,gso,adp_gso)
            l = loss(y_pred, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            scheduler.step()
            l_sum += l.item() * y.shape[0]
            n += y.shape[0]
    return l_sum

loss, model, optimizer, scheduler = prepare_model(args, blocks, n_vertex)

train(loss, args, optimizer, scheduler, model, train_iter4, gso4, gso5)

# Get prediction

pred = model(tensor_train5,gso5,gso6)

final_pred = torch.rand(test5.shape[0],test5.shape[1])

for i in range(0,pred.shape[0]):
    final_pred[i:(i+T),:] += pred[i,0,:,:]
    
for i in range(7):
    final_pred[i,] = final_pred[i,]/(i+1)
    final_pred[-(i+1),] = final_pred[-(i+1),] / (i+1)
    
final_pred[7:31-7,] = final_pred[7:31-7,] / (7)

final_pred = zscore.inverse_transform(final_pred.detach().numpy())

final_pred = pd.DataFrame(final_pred)

final_pred.columns = FM5.columns
final_pred.index = FM5.iloc[31:,:].index


day = ['월','화','수','목','금','토','일'] * 4
day.append('월')
day.append('화')
day.append('수')

origin = FM5.iloc[:31,:]
real = FM5.iloc[31:,:]

origin['요일'] = day
real['요일'] = day
final_pred['요일'] = day

before = np.empty((276,7))
for i,d in enumerate(np.array(['월','화','수','목','금','토','일'])):
    before[:,i] = origin.loc[origin['요일'] == d, origin.columns != '요일'].mean(axis = 0)

before = pd.DataFrame(before)
before.index = origin.columns[:-1]
before.columns = ['월','화','수','목','금','토','일']

real_1 = np.empty((276,7))
for i,d in enumerate(np.array(['월','화','수','목','금','토','일'])):
    real_1[:,i] = real.loc[real['요일'] == d, real.columns != '요일'].mean(axis = 0)

real_1 = pd.DataFrame(real_1)
real_1.index = real.columns[:-1]
real_1.columns = ['월','화','수','목','금','토','일']

pred = np.empty((276,7))
for i,d in enumerate(np.array(['월','화','수','목','금','토','일'])):
    pred[:,i] = final_pred.loc[real['요일'] == d, final_pred.columns != '요일'].mean(axis = 0)

pred = pd.DataFrame(pred)
pred.index = final_pred.columns[:-1]
pred.columns = ['월','화','수','목','금','토','일']

center = ['석촌','올림픽공원','종합운동장']
hop_1 = ['잠실','송파', '둔촌동','방이','잠실새내','삼성','선정릉']

before_df = before.loc[center + hop_1,]
real_df = real_1.loc[center + hop_1,]
pred_df = pred.loc[center + hop_1,]

before_df.to_csv('./Result/before_df.csv')
real_df.to_csv('./Result/real_df.csv')
pred_df.to_csv('./Result/pred_df.csv')

# 평일 비교

before_weekday_mae = abs(real_df.loc[center+hop_1,['월','화','수','목','금']].mean(axis = 1) - before_df.loc[center+hop_1,['월','화','수','목','금']].mean(axis = 1))

pred_weekday_mae = abs(real_df.loc[center+hop_1,['월','화','수','목','금']].mean(axis = 1) - pred_df.loc[center+hop_1,['월','화','수','목','금']].mean(axis = 1))

pd.DataFrame(before_weekday_mae).to_csv('./Result/before_weekday_mae.csv')
pd.DataFrame(pred_weekday_mae).to_csv('./Result/pred_weekday_mae.csv')

# 주말 비교

before_weekend_mae = abs(real_df.loc[center+hop_1,['토','일']].mean(axis = 1) - before_df.loc[center+hop_1,['토','일']].mean(axis = 1))

pred_weekend_mae = abs(real_df.loc[center+hop_1,['토','일']].mean(axis = 1) - pred_df.loc[center+hop_1,['토','일']].mean(axis = 1))

pd.DataFrame(before_weekend_mae).to_csv('./Result/before_weekend_mae.csv')
pd.DataFrame(pred_weekend_mae).to_csv('./Result/pred_weekend_mae.csv')
