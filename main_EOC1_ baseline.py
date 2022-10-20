# -*- coding: utf-8 -*-
"""

@author: owner
Jae-Ho Choi, POSTECH, EE
`Modified: `200402

Language: Python 3.6, Pytorch
SAR-ATR based on Deep Learning

"""

# =============================================================================
# In[1]
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split # for train,test,split
from sklearn.preprocessing import minmax_scale # for minmax normalization
from sklearn.metrics import confusion_matrix # for making confusion matrix
from skimage.transform import resize # for data image resize

from scipy.interpolate import interp1d
import scipy.signal as ssignal
import scipy.ndimage as ndi

import itertools # for plot confusion matrix
import random # for random shuffle for minibatch sampling
import os
import datetime

import cv2

import torch
from torch.autograd import Variable
from torch.utils.data import Dataset, TensorDataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler

import torchvision
import torchvision.transforms as transforms
import time

import sys
import copy
import pickle
import shutil
import math

#sub function
from sub_preprocessing import prepro_DB
from sub_preprocessing import prepro_targetseg
from sub_preprocessing import prepro_shadowseg
from sub_preprocessing import prepro_crop
from sub_preprocessing import plot_segmentation

from sub_dataaugmentation import aug_rotate
from sub_dataaugmentation import aug_scaling
from sub_dataaugmentation import aug_randomerase

import sub_preprocessing
import sub_dataaugmentation
import sub_custom_dataset
import sub_network_architectures
import sub_resnet
import sub_utils
# =============================================================================
# In[2] Initialize, Parameter
# =============================================================================
torch.cuda.empty_cache()

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
print()
#Additional Info when using cuda
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print()
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')


# Hyper-parameters
mode = 'fusion'
# preprocessing
is_dB = False
SAR_data_type = 'EOC1'
prepro_size = 64
use_scaling = True      # False: no scaling, True: training-centric scaling
cfg_total = dict(
                norm_mode = 5   # 1: norm in all training, 2: template-wise, 5: inv. normalization, 6: conventional normalization
)
# network backbone
backbone = 'AConvnet'   # AConvnet, LMBNCNN, ESENet, Resnet
# network training
Num_epoch = 250
Learning_rate = 0.001
Batch_size = 128
# other
flag_saveresult = True
result_path = '/workspace/Result'    # Save path for result
# =============================================================================
# In[3.1] Data Load & Preprocessing
# =============================================================================
"""
Data Load, no preprocessing
Input: file load
Output: 
"""

file_path = '/data'

if mode=='fusion':
    SAR_mode = 2
elif mode=='target':
    SAR_mode = 0
elif mode=='shadow':
    SAR_mode = 1

if SAR_data_type=='SOC':
    data_load = np.load(file_path+'/MSTAR_SOC+sdw_v8.1.npz')
    train_deg = 17
    test_deg = 15
    scale_sdw = np.sin(math.radians(test_deg))/np.sin((math.radians(train_deg)))
elif SAR_data_type=='EOC1':
    data_load = np.load(file_path+'/MSTAR_EOC1+sdw_v8.1.npz')
    train_deg = 17
    test_deg = 30
    scale_sdw = np.sin(math.radians(test_deg))/np.sin((math.radians(train_deg)))


X_train_tgt_cart = data_load['X_train_tgt_cart']
X_train_sdw_cart = data_load['X_train_sdw_cart']
X_test_tgt_cart = data_load['X_test_tgt_cart']
X_test_sdw_cart = data_load['X_test_sdw_cart']
y_train = data_load['Y_train']
y_test = data_load['Y_test']
list_target = data_load['list_class']

COM_train_tgt = data_load['COM_train_tgt']
COM_train_sdw = data_load['COM_train_sdw']
COM_test_tgt = data_load['COM_test_tgt']
COM_test_sdw = data_load['COM_test_sdw']

"""
# Problematic index
# test
- BTR60: 1343~1345, 1381~1382, 1407, 1443~1447, 1487~1488, 1490, 1489
"""
X_train_tgt_adj = X_train_tgt_cart
X_train_sdw_adj = X_train_sdw_cart
y_train_adj = y_train
COM_train_tgt_adj = COM_train_tgt
COM_train_sdw_adj = COM_train_sdw
X_test_tgt_adj = X_test_tgt_cart
X_test_sdw_adj = X_test_sdw_cart
y_test_adj = y_test
COM_test_tgt_adj = COM_test_tgt
COM_test_sdw_adj = COM_test_sdw
# # Eliminate problematic index (train)
# X_train_tgt_adj = X_train_tgt_cart
# X_train_sdw_adj = X_train_sdw_cart
# y_train_adj = y_train
# COM_train_tgt_adj = COM_train_tgt
# COM_train_sdw_adj = COM_train_sdw
# # Eliminate problematic index (test)
# test_del_idx = [1106]
# X_test_tgt_adj = np.delete(X_test_tgt_cart,test_del_idx, axis=2)
# X_test_sdw_adj = np.delete(X_test_sdw_cart,test_del_idx, axis=2)
# y_test_adj = np.delete(y_test, test_del_idx, axis=0)
# COM_test_tgt_adj = np.delete(COM_test_tgt, test_del_idx, axis=1)
# COM_test_sdw_adj = np.delete(COM_test_sdw, test_del_idx, axis=1)

X_train_cart = np.zeros((2,prepro_size,prepro_size,np.shape(X_train_tgt_adj)[2]), dtype='float')
X_test_cart = np.zeros((2,prepro_size,prepro_size,np.shape(X_test_tgt_adj)[2]), dtype='float')

# Cart crop
#tgt
for i in range(X_train_tgt_adj.shape[2]):
#    X_train_cart[0,:,:,i] = prepro_crop(X_train_tgt_cart[:,:,i], COM_train_tgt[:,i].astype('int'), [60,60])
    X_train_cart[0,:,:,i] = prepro_crop(X_train_tgt_adj[:,:,i], [64,64], [prepro_size,prepro_size])
for i in range(X_test_tgt_adj.shape[2]):
#    X_test_cart[0,:,:,i] = prepro_crop(X_test_tgt_cart[:,:,i], COM_test_tgt[:,i].astype('int'), [60,60])
    X_test_cart[0,:,:,i] = prepro_crop(X_test_tgt_adj[:,:,i], [64,64], [prepro_size,prepro_size])
#sdw
for i in range(X_train_tgt_adj.shape[2]):
    X_temp = X_train_sdw_adj[:,:,i]
    X_adj = np.zeros(X_temp.shape, dtype=float)
    X_adj[-96:,:] = X_temp[1:97,:] 
    X_adj[0:32,:] = X_temp[-32:,:]
    COM_sdw = COM_train_sdw_adj[:,i].astype('int')
    COM_sdw[0] = COM_sdw[0]+32
    X_train_cart[1,:,:,i] = prepro_crop(X_adj, COM_sdw, [prepro_size,prepro_size])
for i in range(X_test_tgt_adj.shape[2]):
    X_temp = X_test_sdw_adj[:,:,i]
    X_adj = np.zeros(X_temp.shape, dtype=float)
    X_adj[-96:,:] = X_temp[1:97,:] 
    X_adj[0:32,:] = X_temp[-32:,:]
    COM_sdw = COM_test_sdw_adj[:,i].astype('int')
    COM_sdw[0] = COM_sdw[0]+32
    X_test_cart[1,:,:,i] = prepro_crop(X_adj, COM_sdw, [prepro_size,prepro_size])



# =============================================================================
# In[5] Normalization & Custom dataset
# =============================================================================
tgt_data = X_train_cart[0,:,:,:].copy()
sdw_data = X_train_cart[1,:,:,:].copy()
tgt_mean = np.mean(tgt_data[np.where(tgt_data!=0)])
tgt_std = np.std(tgt_data[np.where(tgt_data!=0)])
sdw_mean = np.mean(sdw_data[np.where(tgt_data!=0)])
sdw_std = np.std(sdw_data[np.where(tgt_data!=0)])
cfg_total['norm_param'] = [tgt_mean,tgt_std,sdw_mean,sdw_std]

if use_scaling==False:
    tgt_transform_train  = transforms.Compose([])
    tgt_transform_test  = transforms.Compose([])
    sdw_transform_train  = transforms.Compose([])
    sdw_transform_test  = transforms.Compose([])
elif use_scaling==True:
    tgt_transform_train  = transforms.Compose([])
    tgt_transform_test  = transforms.Compose([])
    sdw_transform_train  = transforms.Compose([
                                                sub_dataaugmentation.Aug_RandomErasing(),
                                                sub_dataaugmentation.Aug_RandomElasticDistortion(10,50,5,5,1,1.5)])
    sdw_transform_test = transforms.Compose([
                                            sub_dataaugmentation.Aug_RandomScaling(1,1,scale_sdw,scale_sdw, flag_save=False)])

D_train = sub_custom_dataset.custom_dataset_baseline(dat_train=X_train_cart,
                                                    dat_test=X_test_cart,
                                                    label_train=y_train_adj,
                                                    label_test=y_test_adj,
                                                    label_list=list_target,
                                                    train=True,
                                                    tgt_transform = tgt_transform_train,
                                                    sdw_transform = sdw_transform_train,
                                                    use_sdw = SAR_mode, **cfg_total)

D_test = sub_custom_dataset.custom_dataset_baseline(dat_train=X_train_cart,
                                                    dat_test=X_test_cart,
                                                    label_train=y_train_adj,
                                                    label_test=y_test_adj,
                                                    label_list=list_target,
                                                    train=False,
                                                    tgt_transform = tgt_transform_test,
                                                    sdw_transform = sdw_transform_test,
                                                    use_sdw = SAR_mode, **cfg_total)

train_loader = DataLoader(D_train, batch_size = Batch_size, shuffle = True)
test_loader = DataLoader(D_test, batch_size =  Batch_size, shuffle = False)
# =============================================================================
# In[3] Network Architecture Design
# =============================================================================
#model = sub_network_architectures.AConvNet_60_DisReg_SE_v2(n_classes=len(list_target)).to(device)
if backbone=='AConvnet':
    if mode=='fusion':
        model = sub_network_architectures.AConvNet_64_GRIF(n_classes=len(list_target)).to(device)
    else:
        model = sub_network_architectures.AConvNet_60(n_classes=len(list_target)).to(device)
elif backbone[:6]=='resnet':
    model = sub_resnet.__dict__[backbone](flag_abs=True, num_classes=len(list_target)).to(device)
#model = sub_resnet.resnet18(flag_abs=True, num_classes=len(list_target)).to(device)


# =============================================================================
# In[4] Optimizer, Loss function
# =============================================================================

optimizer = optim.Adam(model.parameters(), lr=Learning_rate)
# optim_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)

criterion = nn.CrossEntropyLoss().to(device)


# =============================================================================
# In[5] Network Training
# =============================================================================
print("Model's state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())

print("Optimizer's state_dict:")
for var_name in optimizer.state_dict():
    print(var_name, "\t", optimizer.state_dict()[var_name]) 

print("=============Learning Started=============")
train_loss_list = []
test_loss_list = []
train_acc_list = []
test_acc_list = []

best_model_weight = copy.deepcopy(model.state_dict())
best_loss = 100
best_acc = 0.0

for epoch in range(Num_epoch):
    train_loss_temp = 0
    test_loss_temp = 0
    train_correct = 0
    test_correct = 0
    train_total = 0
    test_total = 0
    
    ts = time.time()    # start time

    for i, data in enumerate(train_loader):
        X, Y = data
        X, Y = Variable(X.float().to(device)), Variable(Y.long().to(device))
        
        model.train()
        
        # Forward
        Y_logit = model(X)
        loss = criterion(Y_logit, Y)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss_temp += loss.data
        _, pred = torch.max(Y_logit.data, 1)
        
        train_total += Y.size(0)
        train_correct += (pred == Y).sum()
        

    pred_list = np.array([], dtype=int)
    target_list = np.array([], dtype=int)    
    for i, data in enumerate(test_loader):
        X, Y = data
        X, Y = Variable(X.float().to(device)), Variable(Y.long().to(device))
        
        model.eval()
        
        # Forward
        Y_logit = model(X)
        loss = criterion(Y_logit, Y)
        
        _, pred = torch.max(Y_logit, 1)
        
        test_loss_temp += loss.data
        test_total += Y.size(0)
        test_correct += (pred == Y).sum()
        pred_list = np.concatenate([pred_list,np.array(pred.to('cpu'))])
        target_list = np.concatenate([target_list,np.array(Y.to('cpu'))])
        
    te = time.time()    # end time
    
    train_loss = np.array(train_loss_temp.to('cpu')/len(train_loader))
    train_acc = 100*np.array(train_correct.to('cpu'))/np.array(train_total)
    test_loss = np.array(test_loss_temp.to('cpu')/len(test_loader))
    test_acc = 100*np.array(test_correct.to('cpu'))/np.array(test_total)
    train_loss_list.append(train_loss)
    test_loss_list.append(test_loss)
    train_acc_list.append(train_acc)
    test_acc_list.append(test_acc)

    if flag_saveresult:
        if test_acc > best_acc:
            best_acc = test_acc
            best_state={
            'Epoch': epoch+1,
            'State_dict': copy.deepcopy(model.state_dict()),
            'Optimizer': copy.deepcopy(optimizer.state_dict()),
            'Pred': pred_list,
            'Target': target_list
            }

    print('Epoch {}, Acc(train/test): {:2.2f}/{:2.2f}, Loss(train/test) {:2.2f}/{:2.2f}. Took {:2.2f} sec'.format(epoch+1, train_acc, test_acc, train_loss, test_loss, te-ts))
    
print("=============Learning Finished=============")
# =============================================================================
# In[6] Model Save
# =============================================================================
if flag_saveresult:
    result_path_adj = os.path.join(result_path,mode,SAR_data_type,backbone)
    result_folder_path = sub_utils.make_savefolder(result_path=result_path_adj)
result = {}
result['train_losses'] = train_loss_list
result['test_losses'] = test_loss_list
result['train_accs'] = train_acc_list
result['test_accs'] = test_acc_list
if flag_saveresult:
    # Save Best model weight
    best_model_name = 'Best.pth'
    best_model_name_full = os.path.join(result_folder_path,best_model_name)
    torch.save(best_state, best_model_name_full)
    # Save Result dictionary
    result_name = 'Result_summary.pickle'
    result_name_full = os.path.join(result_folder_path,result_name)
    with open(result_name_full, 'wb') as f:
        pickle.dump(result, f, pickle.HIGHEST_PROTOCOL)
    # Save code script
    import datetime

    current_file = os.path.realpath(__file__)
    current_time = datetime.datetime.now()
    current_time_path = current_time.strftime('%y%m%d%H%M')
    shutil.copy(current_file,result_folder_path)
    os.rename(os.path.join(result_folder_path,os.path.basename(__file__)),
            os.path.join(result_folder_path,'['+SAR_data_type+'_'+current_time_path+']'+os.path.basename(__file__)))

a = 1