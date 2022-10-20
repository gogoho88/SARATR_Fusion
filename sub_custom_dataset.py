# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 23:23:42 2020

@author: owner
"""
import numpy as np

import torch
from torch.autograd import Variable
from torch.utils.data import Dataset, TensorDataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

class custom_dataset_baseline(Dataset):
    """
    - dat_train: train dataset (C, W, H, B) 꼴, C0은 target, C1은 shadow
    - label_list: 표적 label 이름 list
    - use_sdw: shadow도 사용할지 결정(2:둘다 사용, 1: sdw만 사용, 0: tgt만 사용)
    """
    def __init__(self, dat_train, dat_test, label_train, label_test, label_list, 
                train=True, tgt_transform=None, sdw_transform=None, use_sdw=False, **kwargs):
        
        self.train = train
        self.tgt_transform = tgt_transform
        self.sdw_transform = sdw_transform
        self.use_sdw = use_sdw
        self.mode_norm = kwargs['norm_mode']
        self.norm_param = kwargs['norm_param']
        
        if self.train:
            if self.use_sdw==2:         # 둘 다 사용
                self.list_X = dat_train
            elif self.use_sdw==1:       # sdw만 사용
                self.list_X = dat_train[1,:,:,:]
                self.list_X = np.expand_dims(self.list_X, axis=0)
            else:                       # tgt만 사용
                self.list_X = dat_train[0,:,:,:]
                self.list_X = np.expand_dims(self.list_X, axis=0)
            self.list_y = label_train
        else:
            if self.use_sdw==2:
                self.list_X = dat_test
            elif self.use_sdw==1:
                self.list_X = dat_test[1,:,:,:]
                self.list_X = np.expand_dims(self.list_X, axis=0)
            else:
                self.list_X = dat_test[0,:,:,:]
                self.list_X = np.expand_dims(self.list_X, axis=0)
            self.list_y = label_test
            
        self.list_X = self.list_X.transpose((3,1,2,0))  # BxWxHxC 형태로 변경
        
        #self.list_y 조정
        for i in range(len(label_list)):
            self.list_y[self.list_y == label_list[i]] = i
        self.list_y = np.array(self.list_y, dtype=int)
        
    # 원래는 여기가 전처리
    def __len__(self):
        return len(self.list_X)
    
    def __getitem__(self, index):
        X, y = self.list_X[index], self.list_y[index]
        
        if self.use_sdw==2:
            X_tgt = X[:,:,0]
            X_sdw = X[:,:,1]
            
            if self.tgt_transform is not None:
                X_tgt = self.tgt_transform(X_tgt)
            if self.sdw_transform is not None:
                X_sdw = self.sdw_transform(X_sdw)
            
            # (W,H) -> (C,W,H)
            X_tgt = np.expand_dims(X_tgt, axis=0)
            X_sdw = np.expand_dims(X_sdw, axis=0)
            
            X_ad = np.concatenate([X_tgt,X_sdw], axis=0)         
        elif self.use_sdw==1:
            X_sdw = X[:,:,0]
            
            if self.sdw_transform is not None:
                X_sdw = self.sdw_transform(X_sdw)
            
            # (W,H) -> (C,W,H)
            X_sdw = np.expand_dims(X_sdw, axis=0)
            
            X_ad = X_sdw            
        else:
            X_tgt = X[:,:,0]
            
            if self.tgt_transform is not None:
                X_tgt = self.tgt_transform(X_tgt)
            
            # (W,H) -> (C,W,H)
            X_tgt = np.expand_dims(X_tgt, axis=0)
            
            X_ad = X_tgt
        
        # tensor 변환
        X_f = torch.from_numpy(X_ad)
        
        # Normalization
#        if torch.max(X_f) !=0:
#        if self.use_sdw==2:
#            X_f = 0 # 이부분 나중에 수정 필요
#        elif self.use_sdw==1:
#        X_f = (X_f-torch.min(X_f))/(torch.max(X_f)-torch.min(X_f))
#        X_f = (-X_f)+1
#        X_f = torch.where(X_f==1.,torch.tensor(0., dtype=float),X_f)
        if self.mode_norm==1:
            if self.use_sdw==2:
                X_f[0,:,:] = torch.where(X_f[0,:,:]!=0, (X_f[0,:,:]-self.norm_param[0])/self.norm_param[1], torch.zeros(X_f[0,:,:].shape[0],X_f[0,:,:].shape[1],dtype=float))
                X_f[1,:,:] = torch.where(X_f[1,:,:]!=0, (X_f[1,:,:]-self.norm_param[2])/self.norm_param[3], torch.zeros(X_f[0,:,:].shape[0],X_f[0,:,:].shape[1],dtype=float))
            elif self.use_sdw==1:
                X_f = torch.where(X_f!=0, (X_f-self.norm_param[2])/self.norm_param[3], torch.zeros(X_f.shape[0],X_f.shape[1],dtype=float))
            elif self.use_sdw==0:
                X_f = torch.where(X_f!=0, (X_f-self.norm_param[0])/self.norm_param[1], torch.zeros(X_f.shape[0],X_f.shape[1],dtype=float))
        elif self.mode_norm==2:
            if self.use_sdw==2:
                X_f[0,:,:] = X_f[0,:,:]/torch.max(X_f[0,:,:])
                X_f[1,:,:] = X_f[1,:,:]/torch.max(X_f[1,:,:])
            else:
                X_f = X_f/torch.max(X_f)
        elif self.mode_norm==3:     # 이건 inv norm이 어려울듯
            if self.use_sdw==2:
                X_f[0,:,:] = (X_f[0,:,:]-torch.mean(X_f))/torch.std(X_f[0,:,:])
                X_f[1,:,:] = (X_f[1,:,:]-torch.mean(X_f))/torch.std(X_f[1,:,:])
            else:
                X_f = (X_f-torch.mean(X_f))/torch.std(X_f)
        elif self.mode_norm==4:
            if self.use_sdw==2:
                X_f[0,:,:] = (X_f[0,:,:]-self.norm_param[0])/self.norm_param[1]
                X_f[1,:,:] = (X_f[1,:,:]-self.norm_param[2])/self.norm_param[3]
            elif self.use_sdw==1:
                X_f = (X_f-self.norm_param[2])/self.norm_param[3]
            elif self.use_sdw==0:
                X_f = (X_f-self.norm_param[0])/self.norm_param[1]
                
        elif self.mode_norm==5:         # inv norm.
            if self.use_sdw==2:
                X_f[0,:,:] = torch.where(X_f[0,:,:]!=0, (X_f[0,:,:]-self.norm_param[0])/self.norm_param[1], 
                                            torch.min((X_f[0,:,:]-self.norm_param[0])/self.norm_param[1])*torch.ones(X_f[0,:,:].shape[0],X_f[0,:,:].shape[1],dtype=float))
                X_f[1,:,:] = torch.where(X_f[1,:,:]!=0, -(X_f[1,:,:]-self.norm_param[2])/self.norm_param[3], 
                                            torch.min(-((X_f[1,:,:]-self.norm_param[2])/self.norm_param[3]))*torch.ones(X_f[1,:,:].shape[0],X_f[1,:,:].shape[1],dtype=float))
            elif self.use_sdw==1:
                X_f = torch.where(X_f!=0, -((X_f-self.norm_param[2])/self.norm_param[3]), 
                                            torch.min(-((X_f-self.norm_param[2])/self.norm_param[3]))*torch.ones(X_f.shape[0],X_f.shape[1],dtype=float))
            elif self.use_sdw==0:
                X_f = torch.where(X_f!=0, (X_f-self.norm_param[0])/self.norm_param[1],
                                            torch.min((X_f-self.norm_param[0])/self.norm_param[1])*torch.ones(X_f.shape[0],X_f.shape[1],dtype=float))  

        elif self.mode_norm==6:         # Conventional norm.
            if self.use_sdw==2:
                X_f[0,:,:] = torch.where(X_f[0,:,:]!=0, (X_f[0,:,:]-self.norm_param[0])/self.norm_param[1], 
                                            torch.min((X_f[0,:,:]-self.norm_param[0])/self.norm_param[1])*torch.ones(X_f[0,:,:].shape[0],X_f[0,:,:].shape[1],dtype=float))
                X_f[1,:,:] = torch.where(X_f[1,:,:]!=0, (X_f[1,:,:]-self.norm_param[2])/self.norm_param[3], 
                                            torch.min((X_f[1,:,:]-self.norm_param[2])/self.norm_param[3])*torch.ones(X_f[1,:,:].shape[0],X_f[1,:,:].shape[1],dtype=float))
            elif self.use_sdw==1:
                X_f = torch.where(X_f!=0, (X_f-self.norm_param[2])/self.norm_param[3], 
                                            torch.min(((X_f-self.norm_param[2])/self.norm_param[3]))*torch.ones(X_f.shape[0],X_f.shape[1],dtype=float))
            elif self.use_sdw==0:
                X_f = torch.where(X_f!=0, (X_f-self.norm_param[0])/self.norm_param[1], 
                                            torch.min((X_f-self.norm_param[0])/self.norm_param[1])*torch.ones(X_f.shape[0],X_f.shape[1],dtype=float))
        else:
            X_f = X_f

        return X_f, y