# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 00:11:46 2020

@author: owner
"""
# =============================================================================
# In[1]
# =============================================================================
import numpy as np

import torch
from torch.autograd import Variable
from torch.utils.data import Dataset, TensorDataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sub_lsoftmax import LSoftmaxLinear
# =============================================================================
# In[1] AConvNet 계열
# =============================================================================
# AConvNet
class AConvNet_OGN(nn.Module):
    """
    - Original AConvNet Architecture
    - 참조: S. Chen, H. Wang, F. Xu, and Y. Q. Jin, “Target Classification Using the Deep Convolutional Networks for SAR Images,” 
    IEEE Trans. Geosci. Remote Sens., vol. 54, no. 8, pp. 4806–4817, 2016.
    
    - Nx1x88x88 size Input
    """
    def __init__(self, n_classes, H_size=[16,32,64,128]):
        super(AConvNet_OGN, self).__init__()
        
        self.c1 = nn.Conv2d(1, H_size[0], kernel_size=(5,5), stride=1, padding=0) # 88x88 -> 84x84
        self.p1 = nn.MaxPool2d(kernel_size=(2,2), stride=2)     # 84x84 -> 42x42
        
        self.c2 = nn.Conv2d(H_size[0], H_size[1], kernel_size=(5,5), stride=1, padding=0) # 42x42 -> 38x38
        self.p2 = nn.MaxPool2d(kernel_size=(2,2), stride=2)     # 38x38 -> 19x19
        
        self.c3 = nn.Conv2d(H_size[1], H_size[2], kernel_size=(6,6), stride=1, padding=0) # 19x19 -> 14x14
        self.p3 = nn.MaxPool2d(kernel_size=(2,2), stride=2)     # 14x14 -> 7x7    
        
        self.c4 = nn.Conv2d(H_size[2], H_size[3], kernel_size=(5,5), stride=1, padding=0) # 7x7 -> 3x3
        self.c4_drop = nn.Dropout2d(p=0.5)
        
        self.c5 = nn.Conv2d(H_size[3], n_classes, kernel_size=(3,3), stride=1, padding=0) # 3x3 -> 1x1
               
    def forward(self, X):
        X = F.relu(self.c1(X))
        X = self.p1(X)
        
        X = F.relu(self.c2(X))
        X = self.p2(X)
        
        X = F.relu(self.c3(X))
        X = self.p3(X)
        
        X = F.relu(self.c4(X))
        X = self.c4_drop(X)
        
        X = self.c5(X)
        logit = X.view(-1,self.num_flat_features(X))
        
        return logit
        
    def num_flat_features(self, X):
        size = X.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

class AConvNet_64(nn.Module):
    """
    - Modified AConvNet Architecture -> 64x64 input을 받도록 kernel size 약간 수정
    - 참조: S. Chen, H. Wang, F. Xu, and Y. Q. Jin, “Target Classification Using the Deep Convolutional Networks for SAR Images,” 
    IEEE Trans. Geosci. Remote Sens., vol. 54, no. 8, pp. 4806–4817, 2016.
    
    - Nx1x64x64 size Input
    """
    def __init__(self, n_classes, H_size=[16,32,64,128]):
        super(AConvNet_64, self).__init__()
        
        self.c1 = nn.Conv2d(1, H_size[0], kernel_size=(5,5), stride=1, padding=0) # 64x64 -> 60x60
        self.p1 = nn.MaxPool2d(kernel_size=(2,2), stride=2)     # 60x60 -> 30x30
        
        self.c2 = nn.Conv2d(H_size[0], H_size[1], kernel_size=(5,5), stride=1, padding=0) # 30x30 -> 26x26
        self.p2 = nn.MaxPool2d(kernel_size=(2,2), stride=2)     # 26x26 -> 13x13
        
        self.c3 = nn.Conv2d(H_size[1], H_size[2], kernel_size=(4,4), stride=1, padding=0) # 13x13 -> 10x10
        self.p3 = nn.MaxPool2d(kernel_size=(2,2), stride=2)     # 10x10 -> 5x5    
        
        self.c4 = nn.Conv2d(H_size[2], H_size[3], kernel_size=(3,3), stride=1, padding=0) # 5x5 -> 3x3
        self.c4_drop = nn.Dropout2d(p=0.5)
        
        self.c5 = nn.Conv2d(H_size[3], n_classes, kernel_size=(3,3), stride=1, padding=0) # 3x3 -> 1x1
               
    def forward(self, X):
        X = F.relu(self.c1(X))
        X = self.p1(X)
        
        X = F.relu(self.c2(X))
        X = self.p2(X)
        
        X = F.relu(self.c3(X))
        X = self.p3(X)
        
        X = F.relu(self.c4(X))
        X = self.c4_drop(X)
        
        X = self.c5(X)
        logit = X.view(-1,self.num_flat_features(X))
        
        return logit
        
    def num_flat_features(self, X):
        size = X.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features      

class AConvNet_60(nn.Module):
    """
    - Modified AConvNet Architecture -> 64x64 input을 받도록 kernel size 약간 수정
    - 참조: S. Chen, H. Wang, F. Xu, and Y. Q. Jin, “Target Classification Using the Deep Convolutional Networks for SAR Images,” 
    IEEE Trans. Geosci. Remote Sens., vol. 54, no. 8, pp. 4806–4817, 2016.
    
    - Nx1x60x60 size Input
    """
    def __init__(self, n_classes, H_size=[16,32,64,128]):
        super(AConvNet_60, self).__init__()
        
        self.c1 = nn.Conv2d(1, H_size[0], kernel_size=(5,5), stride=1, padding=0) # 60x60 -> 56x56
        self.p1 = nn.MaxPool2d(kernel_size=(2,2), stride=2)     # 56x56 -> 28x28
        
        self.c2 = nn.Conv2d(H_size[0], H_size[1], kernel_size=(5,5), stride=1, padding=0) # 28x28 -> 24x24
        self.p2 = nn.MaxPool2d(kernel_size=(2,2), stride=2)     # 24x24 -> 12x12
        
        self.c3 = nn.Conv2d(H_size[1], H_size[2], kernel_size=(3,3), stride=1, padding=0) # 12x12 -> 10x10
        self.p3 = nn.MaxPool2d(kernel_size=(2,2), stride=2)     # 10x10 -> 5x5    
        
        self.c4 = nn.Conv2d(H_size[2], H_size[3], kernel_size=(3,3), stride=1, padding=0) # 5x5 -> 3x3
        self.c4_drop = nn.Dropout2d(p=0.5)
        
        self.c5 = nn.Conv2d(H_size[3], n_classes, kernel_size=(3,3), stride=1, padding=0) # 3x3 -> 1x1
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))    

    def forward(self, X):
        X = F.relu(self.c1(X))
        X = self.p1(X)
        
        X = F.relu(self.c2(X))
        X = self.p2(X)
        
        X = F.relu(self.c3(X))
        X = self.p3(X)
        
        X = F.relu(self.c4(X))
        X = self.c4_drop(X)
        
        X = self.c5(X)
        X = self.avgpool(X)
        logit = X.view(-1,self.num_flat_features(X))
        
        return logit
        
    def num_flat_features(self, X):
        size = X.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

class AConvNet_128(nn.Module):
    """
    - Modified AConvNet Architecture -> 128x128 input을 받도록 kernel size 약간 수정
    - 참조: S. Chen, H. Wang, F. Xu, and Y. Q. Jin, “Target Classification Using the Deep Convolutional Networks for SAR Images,” 
    IEEE Trans. Geosci. Remote Sens., vol. 54, no. 8, pp. 4806–4817, 2016.
    
    - Nx1x60x60 size Input
    """
    def __init__(self, n_classes, H_size=[16,32,64,128,128]):
        super(AConvNet_128, self).__init__()
        
        self.c1 = nn.Conv2d(1, H_size[0], kernel_size=(5,5), stride=1, padding=0) # 128x128 -> 124x124
        self.p1 = nn.MaxPool2d(kernel_size=(2,2), stride=2)     # 124x124 -> 62x62
        
        self.c2 = nn.Conv2d(H_size[0], H_size[1], kernel_size=(5,5), stride=1, padding=0) # 62x62 -> 58x58
        self.p2 = nn.MaxPool2d(kernel_size=(2,2), stride=2)     # 58x58 -> 29x29
        
        self.c3 = nn.Conv2d(H_size[1], H_size[2], kernel_size=(4,4), stride=1, padding=0) # 29x29 -> 26x26
        self.p3 = nn.MaxPool2d(kernel_size=(2,2), stride=2)     # 26x26 -> 13x13    
        
        self.c4 = nn.Conv2d(H_size[2], H_size[3], kernel_size=(4,4), stride=1, padding=0) # 13x13 -> 10x10
        self.p4 = nn.MaxPool2d(kernel_size=(2,2), stride=2)     # 10x10 -> 5x5 
        
        self.c5 = nn.Conv2d(H_size[3], H_size[4], kernel_size=(3,3), stride=1, padding=0) # 5x5 -> 3x3
        self.c5_drop = nn.Dropout2d(p=0.5)
        
        self.c6 = nn.Conv2d(H_size[4], n_classes, kernel_size=(3,3), stride=1, padding=0) # 3x3 -> 1x1
               
    def forward(self, X):
        X = F.relu(self.c1(X))
        X = self.p1(X)
        
        X = F.relu(self.c2(X))
        X = self.p2(X)
        
        X = F.relu(self.c3(X))
        X = self.p3(X)
        
        X = F.relu(self.c4(X))
        X = self.p4(X)
        
        X = F.relu(self.c5(X))
        X = self.c5_drop(X)
        
        X = self.c6(X)
        logit = X.view(-1,self.num_flat_features(X))
        
        return logit
        
    def num_flat_features(self, X):
        size = X.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

class AConvNet_60_minpool(nn.Module):
    """
    - Modified AConvNet Architecture -> 64x64 input을 받도록 kernel size 약간 수정
    - 참조: S. Chen, H. Wang, F. Xu, and Y. Q. Jin, “Target Classification Using the Deep Convolutional Networks for SAR Images,” 
    IEEE Trans. Geosci. Remote Sens., vol. 54, no. 8, pp. 4806–4817, 2016.
    
    - Nx1x60x60 size Input
    """
    def __init__(self, n_classes, H_size=[16,32,64,128]):
        super(AConvNet_60_minpool, self).__init__()
        
        self.c1 = nn.Conv2d(1, H_size[0], kernel_size=(5,5), stride=1, padding=0) # 60x60 -> 56x56
        self.p1 = nn.MaxPool2d(kernel_size=(2,2), stride=2)     # 56x56 -> 28x28
        
        self.c2 = nn.Conv2d(H_size[0], H_size[1], kernel_size=(5,5), stride=1, padding=0) # 28x28 -> 24x24
        self.p2 = nn.MaxPool2d(kernel_size=(2,2), stride=2)     # 24x24 -> 12x12
        
        self.c3 = nn.Conv2d(H_size[1], H_size[2], kernel_size=(3,3), stride=1, padding=0) # 12x12 -> 10x10
        self.p3 = nn.MaxPool2d(kernel_size=(2,2), stride=2)     # 10x10 -> 5x5    
        
        self.c4 = nn.Conv2d(H_size[2], H_size[3], kernel_size=(3,3), stride=1, padding=0) # 5x5 -> 3x3
        self.c4_drop = nn.Dropout2d(p=0.5)
        
        self.c5 = nn.Conv2d(H_size[3], n_classes, kernel_size=(3,3), stride=1, padding=0) # 3x3 -> 1x1
               
    def forward(self, X):
        X = F.relu(self.c1(X))
        X = -self.p1(-X)
        
        X = F.relu(self.c2(X))
        X = -self.p2(-X)
        
        X = F.relu(self.c3(X))
        X = -self.p3(-X)
        
        X = F.relu(self.c4(X))
        X = self.c4_drop(X)
        
        X = self.c5(X)
        logit = X.view(-1,self.num_flat_features(X))
        
        return logit
        
    def num_flat_features(self, X):
        size = X.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
    

# =============================================================================
# In[1]
# =============================================================================
# ConvSVM
        
    
# =============================================================================
# In[1]
# =============================================================================
# LMBMCNN
class LMBNCNN_OGN(nn.Module):
    """
    - Original LM-BN-CNN Architecture
    - 참조: F. Zhou, L. Wang, X. Bai, and Y. Hui, “SAR ATR of Ground Vehicles Based on LM-BN-CNN,” 
    IEEE Trans. Geosci. Remote Sens., vol. 56, no. 12, pp. 7282–7293, Dec. 2018.
    
    - Nx1x60x60 size Input
    """
    def __init__(self, n_classes, margin, device, H_size=[16,32,64,64]):
        super(LMBNCNN_OGN, self).__init__()
        self.margin = margin
        self.device = device
        
        self.c1 = nn.Conv2d(1, H_size[0], kernel_size=(5,5), stride=1, padding=0) # 60x60 -> 56x56
        self.c1_bn = nn.BatchNorm2d(H_size[0])
        self.p1 =  nn.MaxPool2d(kernel_size=(2,2), stride=2)     # 56x56 -> 28x28
        
        self.c2 = nn.Conv2d(H_size[0], H_size[1], kernel_size=(3,3), stride=1, padding=0) # 28x28 -> 26x26
        self.c2_bn = nn.BatchNorm2d(H_size[1])
        self.p2 =  nn.MaxPool2d(kernel_size=(2,2), stride=2)     # 26x26 -> 13x13
        
        self.c3 = nn.Conv2d(H_size[1], H_size[2], kernel_size=(4,4), stride=1, padding=0) # 13x13 -> 10x10
        self.c3_bn = nn.BatchNorm2d(H_size[2])
        self.p3 =  nn.MaxPool2d(kernel_size=(2,2), stride=2)     # 10x10 -> 5x5
        self.p3_drop = nn.Dropout2d(p=0.5)
        
        self.c4 = nn.Conv2d(H_size[2], H_size[3], kernel_size=(5,5), stride=1, padding=0) # 5x5 -> 1x1
        self.c4_bn = nn.BatchNorm2d(H_size[3])
        
        self.fc = nn.Linear(H_size[3],n_classes)
        
#        self.lsoftmax_linear = LSoftmaxLinear(H_size[4], H_size[4], margin=margin, device=self.device)
#        self.reset_parameters()
        
    def reset_parameters(self):
        self.lsoftmax_linear.reset_parameters()
        
    def forward(self, X, target=None):
        X = F.relu(self.c1_bn(self.c1(X)))
        X = self.p1(X)
        
        X = F.relu(self.c2_bn(self.c2(X)))
        X = self.p2(X) 
        
        X = F.relu(self.c3_bn(self.c3(X)))
        X = self.p3(X) 
        X = self.p3_drop(X)
        
        X = F.relu(self.c4_bn(self.c4(X)))
        
        X = X.view(-1,self.num_flat_features(X))
        
        logit = self.fc(X)
#        logit = self.lsoftmax_linear(input=X, target=target)
        
        return logit
        
    def num_flat_features(self, X):
        size = X.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features  


# =============================================================================
# In[1]
# =============================================================================
# ESENet
class SELayer(nn.Module):
    """
    - 참조: https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py
    """
    def __init__(self, channel, reduction):
        super(SELayer, self).__init__()
        
        self.p_avg = nn.AdaptiveAvgPool2d(1) # NxCxWxH -> NxCx1x1
        self.fc = nn.Sequential(
                nn.Linear(channel, channel//reduction, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(channel//reduction, channel, bias=False),
                nn.Sigmoid()
                )
        
    def forward(self, X):
        B, C, _, _ = X.size()
        y = self.p_avg(X).view(B,C)
        y = self.fc(y).view(B, C, 1, 1)
        return X*y.expand_as(X)

class ESELayer(nn.Module):
    def __init__(self, channel, W_size, H_size):
        super(ESELayer, self).__init__()
        
        self.c = nn.Conv2d(channel, channel, kernel_size=(W_size,H_size), stride=1, padding=0)
        self.fc = nn.Linear(channel, channel)
        
    def forward(self, X):
        B, C, _, _ = X.size()
        y = F.relu(self.c(X)).view(B,C)
        y = torch.sigmoid(self.fc(y))
        y = y*y
        y = y.view(B,C,1,1)
        return X*y.expand_as(X)
        
               
class ESENet_OGN(nn.Module):
    """
    - Original ESENet Architecture
    - 참조: L. Wang, X. Bai, and F. Zhou, “SAR ATR of Ground Vehicles Based on ESENet,” 
    Remote Sens., vol. 11, no. 11, p. 1316, Jun. 2019.
    
    - Nx1x60x60 size Input
    """
    def __init__(self, n_classes, margin, device, H_size=[16,32,64,64], r=16):
        super(ESENet_OGN, self).__init__()
        self.margin = margin
        self.device = device
        
        self.c1 = nn.Conv2d(1, H_size[0], kernel_size=(5,5), stride=1, padding=0) # 60x60 -> 56x56
        self.c1_bn = nn.BatchNorm2d(H_size[0])
        self.p1 =  nn.MaxPool2d(kernel_size=(2,2), stride=2)     # 56x56 -> 28x28
        
        self.c2 = nn.Conv2d(H_size[0], H_size[1], kernel_size=(3,3), stride=1, padding=0) # 28x28 -> 26x26
        self.SE = SELayer(H_size[1], reduction=r)
        self.p2 = nn.MaxPool2d(kernel_size=(2,2), stride=2)     # 26x26 -> 13x13
        
        self.c3 = nn.Conv2d(H_size[1], H_size[2], kernel_size=(4,4), stride=1, padding=0) # 13x13 -> 10x10
        self.c3_drop = nn.Dropout2d(p=0.5)
        
        self.ESE = ESELayer(H_size[2], W_size=10, H_size=10)
        self.p3 = nn.MaxPool2d(kernel_size=(2,2), stride=2)     # 10x10 -> 5x5
        
        self.c4 = nn.Conv2d(H_size[2], H_size[3], kernel_size=(5,5), stride=1, padding=0) # 5x5 -> 1x1
        self.c4_drop = nn.Dropout2d(p=0.25)
        
        self.fc = nn.Linear(H_size[3],n_classes)
        
#        self.lsoftmax_linear = LSoftmaxLinear(H_size[4], H_size[4], margin=margin, device=self.device)
#        self.reset_parameters()
    
    def reset_parameters(self):
        self.lsoftmax_linear.reset_parameters()
    
    def forward(self, X, target=None):
        X = F.relu(self.c1_bn(self.c1(X)))
        X = self.p1(X)
        
        X = F.relu(self.c2(X))
        
        X = self.SE(X)
        X = self.p2(X)
        
        X = F.relu(self.c3(X))
        X = self.c3_drop(X)
        
        X = self.ESE(X)
        X = self.p3(X)
        
        X = F.relu(self.c4(X))
        X = self.c4_drop(X)
        X = X.view(-1,self.num_flat_features(X))
        
        logit = self.fc(X)
#        logit = self.lsoftmax_linear(input=X, target=target)
        
        return logit
        
    def num_flat_features(self, X):
        size = X.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
        
        
class AConvNet_60_CHfuse(nn.Module):
    """
    - Modified AConvNet Architecture -> 64x64 input을 받도록 kernel size 약간 수정
    - 참조: S. Chen, H. Wang, F. Xu, and Y. Q. Jin, “Target Classification Using the Deep Convolutional Networks for SAR Images,” 
    IEEE Trans. Geosci. Remote Sens., vol. 54, no. 8, pp. 4806–4817, 2016.
    
    - Nx1x60x60 size Input
    """
    def __init__(self, n_classes, H_size=[16,32,64,128]):
        super(AConvNet_60_CHfuse, self).__init__()
        
        self.c1 = nn.Conv2d(2, H_size[0], kernel_size=(5,5), stride=1, padding=0) # 60x60 -> 56x56
        self.p1 = nn.MaxPool2d(kernel_size=(2,2), stride=2)     # 56x56 -> 28x28
        
        self.c2 = nn.Conv2d(H_size[0], H_size[1], kernel_size=(5,5), stride=1, padding=0) # 28x28 -> 24x24
        self.p2 = nn.MaxPool2d(kernel_size=(2,2), stride=2)     # 24x24 -> 12x12
        
        self.c3 = nn.Conv2d(H_size[1], H_size[2], kernel_size=(3,3), stride=1, padding=0) # 12x12 -> 10x10
        self.p3 = nn.MaxPool2d(kernel_size=(2,2), stride=2)     # 10x10 -> 5x5    
        
        self.c4 = nn.Conv2d(H_size[2], H_size[3], kernel_size=(3,3), stride=1, padding=0) # 5x5 -> 3x3
        self.c4_drop = nn.Dropout2d(p=0.5)
        
        self.c5 = nn.Conv2d(H_size[3], n_classes, kernel_size=(3,3), stride=1, padding=0) # 3x3 -> 1x1
               
    def forward(self, X):
        X = F.relu(self.c1(X))
        X = self.p1(X)
        
        X = F.relu(self.c2(X))
        X = self.p2(X)
        
        X = F.relu(self.c3(X))
        X = self.p3(X)
        
        X = F.relu(self.c4(X))
        X = self.c4_drop(X)
        
        X = self.c5(X)
        logit = X.view(-1,self.num_flat_features(X))
        
        return logit
        
    def num_flat_features(self, X):
        size = X.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
    
class AConvNet_60_fuse_late(nn.Module):
    """
    - AConvNet 기반 late fusion (10x1 + 10x1 -> 10x1)
    - 마지막에 AconvNet처럼 10channel로 바로 확률vector하는 것 대신 channel 수 넉넉하게 32로 맞춘 후 FC 한번 더 거침
    
    - Nx1x60x60 size Input
    """
    def __init__(self, n_classes, H_size=[16,32,64,128,32]):
        super(AConvNet_60_fuse_late, self).__init__()
        
        self.c1_tgt = nn.Conv2d(1, H_size[0], kernel_size=(5,5), stride=1, padding=0) # 60x60 -> 56x56
        self.c1_bn_tgt = nn.BatchNorm2d(H_size[0])
        self.p1_tgt = nn.MaxPool2d(kernel_size=(2,2), stride=2)     # 56x56 -> 28x28
        
        self.c1_sdw = nn.Conv2d(1, H_size[0], kernel_size=(5,5), stride=1, padding=0) # 60x60 -> 56x56
        self.c1_bn_sdw = nn.BatchNorm2d(H_size[0])
        self.p1_sdw = nn.MaxPool2d(kernel_size=(2,2), stride=2)     # 56x56 -> 28x28
        
        self.c2_tgt = nn.Conv2d(H_size[0], H_size[1], kernel_size=(5,5), stride=1, padding=0) # 28x28 -> 24x24
        self.c2_bn_tgt = nn.BatchNorm2d(H_size[1])
        self.p2_tgt = nn.MaxPool2d(kernel_size=(2,2), stride=2)     # 24x24 -> 12x12
        
        self.c2_sdw = nn.Conv2d(H_size[0], H_size[1], kernel_size=(5,5), stride=1, padding=0) # 28x28 -> 24x24
        self.c2_bn_sdw = nn.BatchNorm2d(H_size[1])
        self.p2_sdw = nn.MaxPool2d(kernel_size=(2,2), stride=2)     # 24x24 -> 12x12
        
        self.c3_tgt = nn.Conv2d(H_size[1], H_size[2], kernel_size=(3,3), stride=1, padding=0) # 12x12 -> 10x10
        self.c3_bn_tgt = nn.BatchNorm2d(H_size[2])
        self.p3_tgt = nn.MaxPool2d(kernel_size=(2,2), stride=2)     # 10x10 -> 5x5
        
        self.c3_sdw = nn.Conv2d(H_size[1], H_size[2], kernel_size=(3,3), stride=1, padding=0) # 12x12 -> 10x10
        self.c3_bn_sdw = nn.BatchNorm2d(H_size[2])
        self.p3_sdw = nn.MaxPool2d(kernel_size=(2,2), stride=2)     # 10x10 -> 5x5 
        
        self.c4_tgt = nn.Conv2d(H_size[2], H_size[3], kernel_size=(3,3), stride=1, padding=0) # 5x5 -> 3x3
        self.c4_bn_tgt = nn.BatchNorm2d(H_size[3])
        self.c4_drop_tgt = nn.Dropout2d(p=0.5)
        
        self.c4_sdw = nn.Conv2d(H_size[2], H_size[3], kernel_size=(3,3), stride=1, padding=0) # 5x5 -> 3x3
        self.c4_bn_sdw = nn.BatchNorm2d(H_size[3])
        self.c4_drop_sdw = nn.Dropout2d(p=0.5)
        
        self.c5_tgt = nn.Conv2d(H_size[3], H_size[4], kernel_size=(3,3), stride=1, padding=0) # 3x3 -> 1x1
        self.c5_bn_tgt = nn.BatchNorm2d(H_size[4])
        
        self.c5_sdw = nn.Conv2d(H_size[3], H_size[4], kernel_size=(3,3), stride=1, padding=0) # 3x3 -> 1x1
        self.c5_bn_sdw = nn.BatchNorm2d(H_size[4])
        
        self.fc = nn.Linear(H_size[4]*2,n_classes) # 32*2x1 -> 10x1

    def forward(self, X):
        X_tgt = X[:,0,:,:]
        X_tgt = torch.unsqueeze(X_tgt, 1)
        X_sdw = X[:,1,:,:]
        X_sdw = torch.unsqueeze(X_sdw, 1)
        
        X_tgt = F.relu(self.c1_bn_tgt(self.c1_tgt(X_tgt)))
        X_tgt = self.p1_tgt(X_tgt)
        
        X_tgt = F.relu(self.c2_bn_tgt(self.c2_tgt(X_tgt)))
        X_tgt = self.p2_tgt(X_tgt)
        
        X_tgt = F.relu(self.c3_bn_tgt(self.c3_tgt(X_tgt)))
        X_tgt = self.p3_tgt(X_tgt)
        
        X_tgt = F.relu(self.c4_bn_tgt(self.c4_tgt(X_tgt)))
        X_tgt = self.c4_drop_tgt(X_tgt)
        
        X_tgt = F.relu(self.c5_bn_tgt(self.c5_tgt(X_tgt)))
        
        X_sdw = F.relu(self.c1_bn_sdw(self.c1_sdw(X_sdw)))
        X_sdw = -self.p1_sdw(-X_sdw)
        
        X_sdw = F.relu(self.c2_bn_sdw(self.c2_sdw(X_sdw)))
        X_sdw = -self.p2_sdw(-X_sdw)
        
        X_sdw = F.relu(self.c3_bn_sdw(self.c3_sdw(X_sdw)))
        X_sdw = -self.p3_sdw(-X_sdw)
        
        X_sdw = F.relu(self.c4_bn_sdw(self.c4_sdw(X_sdw)))
        X_sdw = self.c4_drop_sdw(X_sdw)
        
        X_sdw = F.relu(self.c5_bn_sdw(self.c5_sdw(X_sdw)))
        
        X_tgt = X_tgt.view(-1,self.num_flat_features(X_tgt))
        X_sdw = X_sdw.view(-1,self.num_flat_features(X_sdw))
        
        X_fuse = torch.cat((X_tgt,X_sdw), dim=1)
        
        logit = self.fc(X_fuse)
        
        return logit
        
    def num_flat_features(self, X):
        size = X.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
    
class AConvNet_60_DisReg(nn.Module):
    """
    - 마지막에 AconvNet처럼 10channel로 바로 확률vector하는 것 대신 channel 수 넉넉하게 32로 맞춘 후 FC 한번 더 거침
    - Nx1x60x60 size Input
    """
    def __init__(self, n_classes, H_size=[16,32,64,128,32]):
        super(AConvNet_60_DisReg, self).__init__()
        
        # L1
        self.c1_tgt = nn.Conv2d(1, H_size[0], kernel_size=(5,5), stride=1, padding=0) # 60x60 -> 56x56
        self.c1_bn_tgt = nn.BatchNorm2d(H_size[0])
        self.p1_tgt = nn.MaxPool2d(kernel_size=(2,2), stride=2)     # 56x56 -> 28x28
        
        self.c1_sdw = nn.Conv2d(1, H_size[0], kernel_size=(5,5), stride=1, padding=0) # 60x60 -> 56x56
        self.c1_bn_sdw = nn.BatchNorm2d(H_size[0])
        self.p1_sdw = nn.MaxPool2d(kernel_size=(2,2), stride=2)     # 56x56 -> 28x28
        
        self.c1_fus = nn.Conv2d(2, H_size[0], kernel_size=(5,5), stride=1, padding=0) # 60x60 -> 56x56
        self.c1_bn_fus = nn.BatchNorm2d(H_size[0])
        self.p1_fus = nn.MaxPool2d(kernel_size=(2,2), stride=2)     # 56x56 -> 28x28

        # L2
        self.c2_tgt = nn.Conv2d(H_size[0], H_size[1], kernel_size=(5,5), stride=1, padding=0) # 28x28 -> 24x24
        self.c2_bn_tgt = nn.BatchNorm2d(H_size[1])
        self.p2_tgt = nn.MaxPool2d(kernel_size=(2,2), stride=2)     # 24x24 -> 12x12
        
        self.c2_sdw = nn.Conv2d(H_size[0], H_size[1], kernel_size=(5,5), stride=1, padding=0) # 28x28 -> 24x24
        self.c2_bn_sdw = nn.BatchNorm2d(H_size[1])
        self.p2_sdw = nn.MaxPool2d(kernel_size=(2,2), stride=2)     # 24x24 -> 12x12
        
        self.c2_fus = nn.Conv2d(H_size[0]*3, H_size[1], kernel_size=(5,5), stride=1, padding=0) # 28x28 -> 24x24
        self.c2_bn_fus = nn.BatchNorm2d(H_size[1])
        self.p2_fus = nn.MaxPool2d(kernel_size=(2,2), stride=2)     # 24x24 -> 12x12

        # L3
        self.c3_tgt = nn.Conv2d(H_size[1], H_size[2], kernel_size=(3,3), stride=1, padding=0) # 12x12 -> 10x10
        self.c3_bn_tgt = nn.BatchNorm2d(H_size[2])
        self.p3_tgt = nn.MaxPool2d(kernel_size=(2,2), stride=2)     # 10x10 -> 5x5
        
        self.c3_sdw = nn.Conv2d(H_size[1], H_size[2], kernel_size=(3,3), stride=1, padding=0) # 12x12 -> 10x10
        self.c3_bn_sdw = nn.BatchNorm2d(H_size[2])
        self.p3_sdw = nn.MaxPool2d(kernel_size=(2,2), stride=2)     # 10x10 -> 5x5 
        
        self.c3_fus = nn.Conv2d(H_size[1]*3, H_size[2], kernel_size=(3,3), stride=1, padding=0) # 12x12 -> 10x10
        self.c3_bn_fus = nn.BatchNorm2d(H_size[2])
        self.p3_fus = nn.MaxPool2d(kernel_size=(2,2), stride=2)     # 10x10 -> 5x5 
        
        
        # L4
        self.c4_tgt = nn.Conv2d(H_size[2], H_size[3], kernel_size=(3,3), stride=1, padding=0) # 5x5 -> 3x3
        self.c4_bn_tgt = nn.BatchNorm2d(H_size[3])
        self.c4_drop_tgt = nn.Dropout2d(p=0.5)
        
        self.c4_sdw = nn.Conv2d(H_size[2], H_size[3], kernel_size=(3,3), stride=1, padding=0) # 5x5 -> 3x3
        self.c4_bn_sdw = nn.BatchNorm2d(H_size[3])
        self.c4_drop_sdw = nn.Dropout2d(p=0.5)
        
        self.c4_fus = nn.Conv2d(H_size[2]*3, H_size[3], kernel_size=(3,3), stride=1, padding=0) # 5x5 -> 3x3
        self.c4_bn_fus = nn.BatchNorm2d(H_size[3])
        self.c4_drop_fus = nn.Dropout2d(p=0.5)
        
        # L5
        self.c5_tgt = nn.Conv2d(H_size[3], H_size[4], kernel_size=(3,3), stride=1, padding=0) # 3x3 -> 1x1
        self.c5_bn_tgt = nn.BatchNorm2d(H_size[4])
        
        self.c5_sdw = nn.Conv2d(H_size[3], H_size[4], kernel_size=(3,3), stride=1, padding=0) # 3x3 -> 1x1
        self.c5_bn_sdw = nn.BatchNorm2d(H_size[4])
        
        self.c5_fus = nn.Conv2d(H_size[3]*3, H_size[4], kernel_size=(3,3), stride=1, padding=0) # 3x3 -> 1x1
        self.c5_bn_fus = nn.BatchNorm2d(H_size[4])
        
        # L6
        self.fc = nn.Linear(H_size[4]*3,n_classes) # 32*3x1 -> 10x1

    def forward(self, X):
        X_tgt = X[:,0,:,:]
        X_tgt = torch.unsqueeze(X_tgt, 1)
        X_sdw = X[:,1,:,:]
        X_sdw = torch.unsqueeze(X_sdw, 1)
        
        # L1
        X_fus = torch.cat((X_tgt,X_sdw), dim=1)
        X_fus = F.relu(self.c1_bn_fus(self.c1_fus(X_fus)))
        X_fus = self.p1_fus(X_fus)
        X_tgt = F.relu(self.c1_bn_tgt(self.c1_tgt(X_tgt)))
        X_tgt = self.p1_tgt(X_tgt)
        X_sdw = F.relu(self.c1_bn_sdw(self.c1_sdw(X_sdw)))
        X_sdw = -self.p1_sdw(-X_sdw)
        
        # L2
        X_fus = torch.cat((X_fus,X_tgt,X_sdw), dim=1)
        X_fus = F.relu(self.c2_bn_fus(self.c2_fus(X_fus)))
        X_fus = self.p2_fus(X_fus)
        X_tgt = F.relu(self.c2_bn_tgt(self.c2_tgt(X_tgt)))
        X_tgt = self.p2_tgt(X_tgt)
        X_sdw = F.relu(self.c2_bn_sdw(self.c2_sdw(X_sdw)))
        X_sdw = -self.p2_sdw(-X_sdw)
        
        # L3
        X_fus = torch.cat((X_fus,X_tgt,X_sdw), dim=1)
        X_fus = F.relu(self.c3_bn_fus(self.c3_fus(X_fus)))
        X_fus = self.p3_fus(X_fus)
        X_tgt = F.relu(self.c3_bn_tgt(self.c3_tgt(X_tgt)))
        X_tgt = self.p3_tgt(X_tgt)
        X_sdw = F.relu(self.c3_bn_sdw(self.c3_sdw(X_sdw)))
        X_sdw = -self.p3_sdw(-X_sdw)
        
        # L4
        X_fus = torch.cat((X_fus,X_tgt,X_sdw), dim=1)
        X_fus = F.relu(self.c4_bn_fus(self.c4_fus(X_fus)))
        X_fus = self.c4_drop_fus(X_fus)
        X_tgt = F.relu(self.c4_bn_tgt(self.c4_tgt(X_tgt)))
        X_tgt = self.c4_drop_tgt(X_tgt)
        X_sdw = F.relu(self.c4_bn_sdw(self.c4_sdw(X_sdw)))
        X_sdw = self.c4_drop_sdw(X_sdw)
        
        # L5
        X_fus = torch.cat((X_fus,X_tgt,X_sdw), dim=1)
        X_fus = F.relu(self.c5_bn_fus(self.c5_fus(X_fus)))       
        X_tgt = F.relu(self.c5_bn_tgt(self.c5_tgt(X_tgt)))       
        X_sdw = F.relu(self.c5_bn_sdw(self.c5_sdw(X_sdw)))
        
        X_fus = X_fus.view(-1,self.num_flat_features(X_fus))
        X_tgt = X_tgt.view(-1,self.num_flat_features(X_tgt))
        X_sdw = X_sdw.view(-1,self.num_flat_features(X_sdw))
        
        # L6
        X_fus = torch.cat((X_fus,X_tgt,X_sdw), dim=1)        
        logit = self.fc(X_fus)
        
        return logit
        
    def num_flat_features(self, X):
        size = X.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
    
    
class AConvNet_60_DisReg_SE_v1(nn.Module):
    """
    - 마지막에 AconvNet처럼 10channel로 바로 확률vector하는 것 대신 channel 수 넉넉하게 32로 맞춘 후 FC 한번 더 거침
    - Nx1x60x60 size Input
    """
    def __init__(self, n_classes, H_size=[16,32,64,128,32], r=4):
        super(AConvNet_60_DisReg_SE_v1, self).__init__()
        
        # L1
        self.c1_tgt = nn.Conv2d(1, H_size[0], kernel_size=(5,5), stride=1, padding=0) # 60x60 -> 56x56
        self.c1_bn_tgt = nn.BatchNorm2d(H_size[0])
        self.p1_tgt = nn.MaxPool2d(kernel_size=(2,2), stride=2)     # 56x56 -> 28x28
        
        self.c1_sdw = nn.Conv2d(1, H_size[0], kernel_size=(5,5), stride=1, padding=0) # 60x60 -> 56x56
        self.c1_bn_sdw = nn.BatchNorm2d(H_size[0])
        self.p1_sdw = nn.MaxPool2d(kernel_size=(2,2), stride=2)     # 56x56 -> 28x28
        
        self.c1_fus = nn.Conv2d(2, H_size[0], kernel_size=(5,5), stride=1, padding=0) # 60x60 -> 56x56
        self.c1_bn_fus = nn.BatchNorm2d(H_size[0])
        self.p1_fus = nn.MaxPool2d(kernel_size=(2,2), stride=2)     # 56x56 -> 28x28

        # L2
        self.c2_tgt = nn.Conv2d(H_size[0], H_size[1], kernel_size=(5,5), stride=1, padding=0) # 28x28 -> 24x24
        self.c2_bn_tgt = nn.BatchNorm2d(H_size[1])
        self.p2_tgt = nn.MaxPool2d(kernel_size=(2,2), stride=2)     # 24x24 -> 12x12
        
        self.c2_sdw = nn.Conv2d(H_size[0], H_size[1], kernel_size=(5,5), stride=1, padding=0) # 28x28 -> 24x24
        self.c2_bn_sdw = nn.BatchNorm2d(H_size[1])
        self.p2_sdw = nn.MaxPool2d(kernel_size=(2,2), stride=2)     # 24x24 -> 12x12
        
        self.SE1 = SELayer(H_size[0]*3, reduction=r)
        self.c2_fus = nn.Conv2d(H_size[0]*3, H_size[1], kernel_size=(5,5), stride=1, padding=0) # 28x28 -> 24x24
        self.c2_bn_fus = nn.BatchNorm2d(H_size[1])
        self.p2_fus = nn.MaxPool2d(kernel_size=(2,2), stride=2)     # 24x24 -> 12x12
 
        # L3
        self.c3_tgt = nn.Conv2d(H_size[1], H_size[2], kernel_size=(3,3), stride=1, padding=0) # 12x12 -> 10x10
        self.c3_bn_tgt = nn.BatchNorm2d(H_size[2])
        self.p3_tgt = nn.MaxPool2d(kernel_size=(2,2), stride=2)     # 10x10 -> 5x5
        
        self.c3_sdw = nn.Conv2d(H_size[1], H_size[2], kernel_size=(3,3), stride=1, padding=0) # 12x12 -> 10x10
        self.c3_bn_sdw = nn.BatchNorm2d(H_size[2])
        self.p3_sdw = nn.MaxPool2d(kernel_size=(2,2), stride=2)     # 10x10 -> 5x5 
        
        self.SE2 = SELayer(H_size[1]*3, reduction=r)
        self.c3_fus = nn.Conv2d(H_size[1]*3, H_size[2], kernel_size=(3,3), stride=1, padding=0) # 12x12 -> 10x10
        self.c3_bn_fus = nn.BatchNorm2d(H_size[2])
        self.p3_fus = nn.MaxPool2d(kernel_size=(2,2), stride=2)     # 10x10 -> 5x5 
        
        
        # L4
        self.c4_tgt = nn.Conv2d(H_size[2], H_size[3], kernel_size=(3,3), stride=1, padding=0) # 5x5 -> 3x3
        self.c4_bn_tgt = nn.BatchNorm2d(H_size[3])
        self.c4_drop_tgt = nn.Dropout2d(p=0.5)
        
        self.c4_sdw = nn.Conv2d(H_size[2], H_size[3], kernel_size=(3,3), stride=1, padding=0) # 5x5 -> 3x3
        self.c4_bn_sdw = nn.BatchNorm2d(H_size[3])
        self.c4_drop_sdw = nn.Dropout2d(p=0.5)
        
        self.SE3 = SELayer(H_size[2]*3, reduction=r)
        self.c4_fus = nn.Conv2d(H_size[2]*3, H_size[3], kernel_size=(3,3), stride=1, padding=0) # 5x5 -> 3x3
        self.c4_bn_fus = nn.BatchNorm2d(H_size[3])
        self.c4_drop_fus = nn.Dropout2d(p=0.5)
        
        # L5
        self.c5_tgt = nn.Conv2d(H_size[3], H_size[4], kernel_size=(3,3), stride=1, padding=0) # 3x3 -> 1x1
        self.c5_bn_tgt = nn.BatchNorm2d(H_size[4])
        
        self.c5_sdw = nn.Conv2d(H_size[3], H_size[4], kernel_size=(3,3), stride=1, padding=0) # 3x3 -> 1x1
        self.c5_bn_sdw = nn.BatchNorm2d(H_size[4])
        
        self.SE4 = SELayer(H_size[3]*3, reduction=r)
        self.c5_fus = nn.Conv2d(H_size[3]*3, H_size[4], kernel_size=(3,3), stride=1, padding=0) # 3x3 -> 1x1
        self.c5_bn_fus = nn.BatchNorm2d(H_size[4])
        
        # L6
        self.fc = nn.Linear(H_size[4]*3,n_classes) # 32*3x1 -> 10x1

    def forward(self, X):
        X_tgt = X[:,0,:,:]
        X_tgt = torch.unsqueeze(X_tgt, 1)
        X_sdw = X[:,1,:,:]
        X_sdw = torch.unsqueeze(X_sdw, 1)
        
        # L1
        X_fus = torch.cat((X_tgt,X_sdw), dim=1)
        X_fus = F.relu(self.c1_bn_fus(self.c1_fus(X_fus)))
        X_fus = self.p1_fus(X_fus)
        X_tgt = F.relu(self.c1_bn_tgt(self.c1_tgt(X_tgt)))
        X_tgt = self.p1_tgt(X_tgt)
        X_sdw = F.relu(self.c1_bn_sdw(self.c1_sdw(X_sdw)))
        X_sdw = -self.p1_sdw(-X_sdw)
        
        # L2
        X_fus = torch.cat((X_fus,X_tgt,X_sdw), dim=1)
        X_fus = self.SE1(X_fus)
        X_fus = F.relu(self.c2_bn_fus(self.c2_fus(X_fus)))
        X_fus = self.p2_fus(X_fus)
        X_tgt = F.relu(self.c2_bn_tgt(self.c2_tgt(X_tgt)))
        X_tgt = self.p2_tgt(X_tgt)
        X_sdw = F.relu(self.c2_bn_sdw(self.c2_sdw(X_sdw)))
        X_sdw = -self.p2_sdw(-X_sdw)
        
        # L3
        X_fus = torch.cat((X_fus,X_tgt,X_sdw), dim=1)
        X_fus = self.SE2(X_fus)
        X_fus = F.relu(self.c3_bn_fus(self.c3_fus(X_fus)))
        X_fus = self.p3_fus(X_fus)
        X_tgt = F.relu(self.c3_bn_tgt(self.c3_tgt(X_tgt)))
        X_tgt = self.p3_tgt(X_tgt)
        X_sdw = F.relu(self.c3_bn_sdw(self.c3_sdw(X_sdw)))
        X_sdw = -self.p3_sdw(-X_sdw)
        
        # L4
        X_fus = torch.cat((X_fus,X_tgt,X_sdw), dim=1)
        X_fus = self.SE3(X_fus)
        X_fus = F.relu(self.c4_bn_fus(self.c4_fus(X_fus)))
        X_fus = self.c4_drop_fus(X_fus)
        X_tgt = F.relu(self.c4_bn_tgt(self.c4_tgt(X_tgt)))
        X_tgt = self.c4_drop_tgt(X_tgt)
        X_sdw = F.relu(self.c4_bn_sdw(self.c4_sdw(X_sdw)))
        X_sdw = self.c4_drop_sdw(X_sdw)
        
        # L5
        X_fus = torch.cat((X_fus,X_tgt,X_sdw), dim=1)
        X_fus = self.SE4(X_fus)
        X_fus = F.relu(self.c5_bn_fus(self.c5_fus(X_fus)))       
        X_tgt = F.relu(self.c5_bn_tgt(self.c5_tgt(X_tgt)))       
        X_sdw = F.relu(self.c5_bn_sdw(self.c5_sdw(X_sdw)))
        
        X_fus = X_fus.view(-1,self.num_flat_features(X_fus))
        X_tgt = X_tgt.view(-1,self.num_flat_features(X_tgt))
        X_sdw = X_sdw.view(-1,self.num_flat_features(X_sdw))
        
        # L6
        X_fus = torch.cat((X_fus,X_tgt,X_sdw), dim=1)        
        logit = self.fc(X_fus)
        
        return logit
        
    def num_flat_features(self, X):
        size = X.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
    
class AConvNet_60_DisReg_SE_v2(nn.Module):
    """
    - 마지막에 AconvNet처럼 10channel로 바로 확률vector하는 것 대신 channel 수 넉넉하게 32로 맞춘 후 FC 한번 더 거침
    - Nx1x60x60 size Input
    - ADD SE layer on the ealiest layer
    """
    def __init__(self, n_classes, H_size=[16,32,64,128,32], r=4):
        super(AConvNet_60_DisReg_SE_v2, self).__init__()
        
        # L1
        self.c1_tgt = nn.Conv2d(1, H_size[0], kernel_size=(5,5), stride=1, padding=0) # 60x60 -> 56x56
        self.c1_bn_tgt = nn.BatchNorm2d(H_size[0])
        self.p1_tgt = nn.MaxPool2d(kernel_size=(2,2), stride=2)     # 56x56 -> 28x28
        
        self.c1_sdw = nn.Conv2d(1, H_size[0], kernel_size=(5,5), stride=1, padding=0) # 60x60 -> 56x56
        self.c1_bn_sdw = nn.BatchNorm2d(H_size[0])
        self.p1_sdw = nn.MaxPool2d(kernel_size=(2,2), stride=2)     # 56x56 -> 28x28
        
        self.SE0 = SELayer(2, reduction=2)
        self.c1_fus = nn.Conv2d(2, H_size[0], kernel_size=(5,5), stride=1, padding=0) # 60x60 -> 56x56
        self.c1_bn_fus = nn.BatchNorm2d(H_size[0])
        self.p1_fus = nn.MaxPool2d(kernel_size=(2,2), stride=2)     # 56x56 -> 28x28

        # L2
        self.c2_tgt = nn.Conv2d(H_size[0], H_size[1], kernel_size=(5,5), stride=1, padding=0) # 28x28 -> 24x24
        self.c2_bn_tgt = nn.BatchNorm2d(H_size[1])
        self.p2_tgt = nn.MaxPool2d(kernel_size=(2,2), stride=2)     # 24x24 -> 12x12
        
        self.c2_sdw = nn.Conv2d(H_size[0], H_size[1], kernel_size=(5,5), stride=1, padding=0) # 28x28 -> 24x24
        self.c2_bn_sdw = nn.BatchNorm2d(H_size[1])
        self.p2_sdw = nn.MaxPool2d(kernel_size=(2,2), stride=2)     # 24x24 -> 12x12
        
        self.SE1 = SELayer(H_size[0]*3, reduction=r)
        self.c2_fus = nn.Conv2d(H_size[0]*3, H_size[1], kernel_size=(5,5), stride=1, padding=0) # 28x28 -> 24x24
        self.c2_bn_fus = nn.BatchNorm2d(H_size[1])
        self.p2_fus = nn.MaxPool2d(kernel_size=(2,2), stride=2)     # 24x24 -> 12x12
 
        # L3
        self.c3_tgt = nn.Conv2d(H_size[1], H_size[2], kernel_size=(3,3), stride=1, padding=0) # 12x12 -> 10x10
        self.c3_bn_tgt = nn.BatchNorm2d(H_size[2])
        self.p3_tgt = nn.MaxPool2d(kernel_size=(2,2), stride=2)     # 10x10 -> 5x5
        
        self.c3_sdw = nn.Conv2d(H_size[1], H_size[2], kernel_size=(3,3), stride=1, padding=0) # 12x12 -> 10x10
        self.c3_bn_sdw = nn.BatchNorm2d(H_size[2])
        self.p3_sdw = nn.MaxPool2d(kernel_size=(2,2), stride=2)     # 10x10 -> 5x5 
        
        self.SE2 = SELayer(H_size[1]*3, reduction=r)
        self.c3_fus = nn.Conv2d(H_size[1]*3, H_size[2], kernel_size=(3,3), stride=1, padding=0) # 12x12 -> 10x10
        self.c3_bn_fus = nn.BatchNorm2d(H_size[2])
        self.p3_fus = nn.MaxPool2d(kernel_size=(2,2), stride=2)     # 10x10 -> 5x5 
        
        
        # L4
        self.c4_tgt = nn.Conv2d(H_size[2], H_size[3], kernel_size=(3,3), stride=1, padding=0) # 5x5 -> 3x3
        self.c4_bn_tgt = nn.BatchNorm2d(H_size[3])
        self.c4_drop_tgt = nn.Dropout2d(p=0.5)
        
        self.c4_sdw = nn.Conv2d(H_size[2], H_size[3], kernel_size=(3,3), stride=1, padding=0) # 5x5 -> 3x3
        self.c4_bn_sdw = nn.BatchNorm2d(H_size[3])
        self.c4_drop_sdw = nn.Dropout2d(p=0.5)
        
        self.SE3 = SELayer(H_size[2]*3, reduction=r)
        self.c4_fus = nn.Conv2d(H_size[2]*3, H_size[3], kernel_size=(3,3), stride=1, padding=0) # 5x5 -> 3x3
        self.c4_bn_fus = nn.BatchNorm2d(H_size[3])
        self.c4_drop_fus = nn.Dropout2d(p=0.5)
        
        # L5
        self.c5_tgt = nn.Conv2d(H_size[3], H_size[4], kernel_size=(3,3), stride=1, padding=0) # 3x3 -> 1x1
        self.c5_bn_tgt = nn.BatchNorm2d(H_size[4])
        
        self.c5_sdw = nn.Conv2d(H_size[3], H_size[4], kernel_size=(3,3), stride=1, padding=0) # 3x3 -> 1x1
        self.c5_bn_sdw = nn.BatchNorm2d(H_size[4])
        
        self.SE4 = SELayer(H_size[3]*3, reduction=r)
        self.c5_fus = nn.Conv2d(H_size[3]*3, H_size[4], kernel_size=(3,3), stride=1, padding=0) # 3x3 -> 1x1
        self.c5_bn_fus = nn.BatchNorm2d(H_size[4])
        
        # L6
        self.fc = nn.Linear(H_size[4]*3,n_classes) # 32*3x1 -> 10x1

    def forward(self, X):
        X_tgt = X[:,0,:,:]
        X_tgt = torch.unsqueeze(X_tgt, 1)
        X_sdw = X[:,1,:,:]
        X_sdw = torch.unsqueeze(X_sdw, 1)
        
        # L1
        X_fus = torch.cat((X_tgt,X_sdw), dim=1)
        X_fus = self.SE0(X_fus)
        X_fus = F.relu(self.c1_bn_fus(self.c1_fus(X_fus)))
        X_fus = self.p1_fus(X_fus)
        X_tgt = F.relu(self.c1_bn_tgt(self.c1_tgt(X_tgt)))
        X_tgt = self.p1_tgt(X_tgt)
        X_sdw = F.relu(self.c1_bn_sdw(self.c1_sdw(X_sdw)))
        X_sdw = -self.p1_sdw(-X_sdw)
        
        # L2
        X_fus = torch.cat((X_fus,X_tgt,X_sdw), dim=1)
        X_fus = self.SE1(X_fus)
        X_fus = F.relu(self.c2_bn_fus(self.c2_fus(X_fus)))
        X_fus = self.p2_fus(X_fus)
        X_tgt = F.relu(self.c2_bn_tgt(self.c2_tgt(X_tgt)))
        X_tgt = self.p2_tgt(X_tgt)
        X_sdw = F.relu(self.c2_bn_sdw(self.c2_sdw(X_sdw)))
        X_sdw = -self.p2_sdw(-X_sdw)
        
        # L3
        X_fus = torch.cat((X_fus,X_tgt,X_sdw), dim=1)
        X_fus = self.SE2(X_fus)
        X_fus = F.relu(self.c3_bn_fus(self.c3_fus(X_fus)))
        X_fus = self.p3_fus(X_fus)
        X_tgt = F.relu(self.c3_bn_tgt(self.c3_tgt(X_tgt)))
        X_tgt = self.p3_tgt(X_tgt)
        X_sdw = F.relu(self.c3_bn_sdw(self.c3_sdw(X_sdw)))
        X_sdw = -self.p3_sdw(-X_sdw)
        
        # L4
        X_fus = torch.cat((X_fus,X_tgt,X_sdw), dim=1)
        X_fus = self.SE3(X_fus)
        X_fus = F.relu(self.c4_bn_fus(self.c4_fus(X_fus)))
        X_fus = self.c4_drop_fus(X_fus)
        X_tgt = F.relu(self.c4_bn_tgt(self.c4_tgt(X_tgt)))
        X_tgt = self.c4_drop_tgt(X_tgt)
        X_sdw = F.relu(self.c4_bn_sdw(self.c4_sdw(X_sdw)))
        X_sdw = self.c4_drop_sdw(X_sdw)
        
        # L5
        X_fus = torch.cat((X_fus,X_tgt,X_sdw), dim=1)
        X_fus = self.SE4(X_fus)
        X_fus = F.relu(self.c5_bn_fus(self.c5_fus(X_fus)))       
        X_tgt = F.relu(self.c5_bn_tgt(self.c5_tgt(X_tgt)))       
        X_sdw = F.relu(self.c5_bn_sdw(self.c5_sdw(X_sdw)))
        
        X_fus = X_fus.view(-1,self.num_flat_features(X_fus))
        X_tgt = X_tgt.view(-1,self.num_flat_features(X_tgt))
        X_sdw = X_sdw.view(-1,self.num_flat_features(X_sdw))
        
        # L6
        X_fus = torch.cat((X_fus,X_tgt,X_sdw), dim=1)        
        logit = self.fc(X_fus)
        
        return logit
        
    def num_flat_features(self, X):
        size = X.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
    
    
class AConvNet_60_DisReg_SE_v3(nn.Module):
    """
    - 마지막에 AconvNet처럼 10channel로 바로 확률vector하는 것 대신 channel 수 넉넉하게 32로 맞춘 후 FC 한번 더 거침
    - Nx1x60x60 size Input
    - v2에서 SE->ESE로 변경
    """
    def __init__(self, n_classes, H_size=[16,32,64,128,32], r=4):
        super(AConvNet_60_DisReg_SE_v3, self).__init__()
        
        # L1
        self.c1_tgt = nn.Conv2d(1, H_size[0], kernel_size=(5,5), stride=1, padding=0) # 60x60 -> 56x56
        self.c1_bn_tgt = nn.BatchNorm2d(H_size[0])
        self.p1_tgt = nn.MaxPool2d(kernel_size=(2,2), stride=2)     # 56x56 -> 28x28
        
        self.c1_sdw = nn.Conv2d(1, H_size[0], kernel_size=(5,5), stride=1, padding=0) # 60x60 -> 56x56
        self.c1_bn_sdw = nn.BatchNorm2d(H_size[0])
        self.p1_sdw = nn.MaxPool2d(kernel_size=(2,2), stride=2)     # 56x56 -> 28x28
        
        self.ESE0 = ESELayer(2, W_size=60, H_size=60)
        self.c1_fus = nn.Conv2d(2, H_size[0], kernel_size=(5,5), stride=1, padding=0) # 60x60 -> 56x56
        self.c1_bn_fus = nn.BatchNorm2d(H_size[0])
        self.p1_fus = nn.MaxPool2d(kernel_size=(2,2), stride=2)     # 56x56 -> 28x28

        # L2
        self.c2_tgt = nn.Conv2d(H_size[0], H_size[1], kernel_size=(5,5), stride=1, padding=0) # 28x28 -> 24x24
        self.c2_bn_tgt = nn.BatchNorm2d(H_size[1])
        self.p2_tgt = nn.MaxPool2d(kernel_size=(2,2), stride=2)     # 24x24 -> 12x12
        
        self.c2_sdw = nn.Conv2d(H_size[0], H_size[1], kernel_size=(5,5), stride=1, padding=0) # 28x28 -> 24x24
        self.c2_bn_sdw = nn.BatchNorm2d(H_size[1])
        self.p2_sdw = nn.MaxPool2d(kernel_size=(2,2), stride=2)     # 24x24 -> 12x12
        
        self.ESE1 = ESELayer(H_size[0]*3, W_size=28, H_size=28)
        self.c2_fus = nn.Conv2d(H_size[0]*3, H_size[1], kernel_size=(5,5), stride=1, padding=0) # 28x28 -> 24x24
        self.c2_bn_fus = nn.BatchNorm2d(H_size[1])
        self.p2_fus = nn.MaxPool2d(kernel_size=(2,2), stride=2)     # 24x24 -> 12x12
 
        # L3
        self.c3_tgt = nn.Conv2d(H_size[1], H_size[2], kernel_size=(3,3), stride=1, padding=0) # 12x12 -> 10x10
        self.c3_bn_tgt = nn.BatchNorm2d(H_size[2])
        self.p3_tgt = nn.MaxPool2d(kernel_size=(2,2), stride=2)     # 10x10 -> 5x5
        
        self.c3_sdw = nn.Conv2d(H_size[1], H_size[2], kernel_size=(3,3), stride=1, padding=0) # 12x12 -> 10x10
        self.c3_bn_sdw = nn.BatchNorm2d(H_size[2])
        self.p3_sdw = nn.MaxPool2d(kernel_size=(2,2), stride=2)     # 10x10 -> 5x5 
        
        self.ESE2 = ESELayer(H_size[1]*3, W_size=12, H_size=12)
        self.c3_fus = nn.Conv2d(H_size[1]*3, H_size[2], kernel_size=(3,3), stride=1, padding=0) # 12x12 -> 10x10
        self.c3_bn_fus = nn.BatchNorm2d(H_size[2])
        self.p3_fus = nn.MaxPool2d(kernel_size=(2,2), stride=2)     # 10x10 -> 5x5 
        
        
        # L4
        self.c4_tgt = nn.Conv2d(H_size[2], H_size[3], kernel_size=(3,3), stride=1, padding=0) # 5x5 -> 3x3
        self.c4_bn_tgt = nn.BatchNorm2d(H_size[3])
        self.c4_drop_tgt = nn.Dropout2d(p=0.5)
        
        self.c4_sdw = nn.Conv2d(H_size[2], H_size[3], kernel_size=(3,3), stride=1, padding=0) # 5x5 -> 3x3
        self.c4_bn_sdw = nn.BatchNorm2d(H_size[3])
        self.c4_drop_sdw = nn.Dropout2d(p=0.5)
        
        self.ESE3 = ESELayer(H_size[2]*3, W_size=5, H_size=5)
        self.c4_fus = nn.Conv2d(H_size[2]*3, H_size[3], kernel_size=(3,3), stride=1, padding=0) # 5x5 -> 3x3
        self.c4_bn_fus = nn.BatchNorm2d(H_size[3])
        self.c4_drop_fus = nn.Dropout2d(p=0.5)
        
        # L5
        self.c5_tgt = nn.Conv2d(H_size[3], H_size[4], kernel_size=(3,3), stride=1, padding=0) # 3x3 -> 1x1
        self.c5_bn_tgt = nn.BatchNorm2d(H_size[4])
        
        self.c5_sdw = nn.Conv2d(H_size[3], H_size[4], kernel_size=(3,3), stride=1, padding=0) # 3x3 -> 1x1
        self.c5_bn_sdw = nn.BatchNorm2d(H_size[4])
        
        self.ESE4 = ESELayer(H_size[3]*3, W_size=3, H_size=3)
        self.c5_fus = nn.Conv2d(H_size[3]*3, H_size[4], kernel_size=(3,3), stride=1, padding=0) # 3x3 -> 1x1
        self.c5_bn_fus = nn.BatchNorm2d(H_size[4])
        
        # L6
        self.fc = nn.Linear(H_size[4]*3,n_classes) # 32*3x1 -> 10x1

    def forward(self, X):
        X_tgt = X[:,0,:,:]
        X_tgt = torch.unsqueeze(X_tgt, 1)
        X_sdw = X[:,1,:,:]
        X_sdw = torch.unsqueeze(X_sdw, 1)
        
        # L1
        X_fus = torch.cat((X_tgt,X_sdw), dim=1)
        X_fus = self.ESE0(X_fus)
        X_fus = F.relu(self.c1_bn_fus(self.c1_fus(X_fus)))
        X_fus = self.p1_fus(X_fus)
        X_tgt = F.relu(self.c1_bn_tgt(self.c1_tgt(X_tgt)))
        X_tgt = self.p1_tgt(X_tgt)
        X_sdw = F.relu(self.c1_bn_sdw(self.c1_sdw(X_sdw)))
        X_sdw = self.p1_sdw(X_sdw)
        
        # L2
        X_fus = torch.cat((X_fus,X_tgt,X_sdw), dim=1)
        X_fus = self.ESE1(X_fus)
        X_fus = F.relu(self.c2_bn_fus(self.c2_fus(X_fus)))
        X_fus = self.p2_fus(X_fus)
        X_tgt = F.relu(self.c2_bn_tgt(self.c2_tgt(X_tgt)))
        X_tgt = self.p2_tgt(X_tgt)
        X_sdw = F.relu(self.c2_bn_sdw(self.c2_sdw(X_sdw)))
        X_sdw = self.p2_sdw(X_sdw)
        
        # L3
        X_fus = torch.cat((X_fus,X_tgt,X_sdw), dim=1)
        X_fus = self.ESE2(X_fus)
        X_fus = F.relu(self.c3_bn_fus(self.c3_fus(X_fus)))
        X_fus = self.p3_fus(X_fus)
        X_tgt = F.relu(self.c3_bn_tgt(self.c3_tgt(X_tgt)))
        X_tgt = self.p3_tgt(X_tgt)
        X_sdw = F.relu(self.c3_bn_sdw(self.c3_sdw(X_sdw)))
        X_sdw = self.p3_sdw(X_sdw)
        
        # L4
        X_fus = torch.cat((X_fus,X_tgt,X_sdw), dim=1)
        X_fus = self.ESE3(X_fus)
        X_fus = F.relu(self.c4_bn_fus(self.c4_fus(X_fus)))
        X_fus = self.c4_drop_fus(X_fus)
        X_tgt = F.relu(self.c4_bn_tgt(self.c4_tgt(X_tgt)))
        X_tgt = self.c4_drop_tgt(X_tgt)
        X_sdw = F.relu(self.c4_bn_sdw(self.c4_sdw(X_sdw)))
        X_sdw = self.c4_drop_sdw(X_sdw)
        
        # L5
        X_fus = torch.cat((X_fus,X_tgt,X_sdw), dim=1)
        X_fus = self.ESE4(X_fus)
        X_fus = F.relu(self.c5_bn_fus(self.c5_fus(X_fus)))       
        X_tgt = F.relu(self.c5_bn_tgt(self.c5_tgt(X_tgt)))       
        X_sdw = F.relu(self.c5_bn_sdw(self.c5_sdw(X_sdw)))
        
        X_fus = X_fus.view(-1,self.num_flat_features(X_fus))
        X_tgt = X_tgt.view(-1,self.num_flat_features(X_tgt))
        X_sdw = X_sdw.view(-1,self.num_flat_features(X_sdw))
        
        # L6
        X_fus = torch.cat((X_fus,X_tgt,X_sdw), dim=1)        
        logit = self.fc(X_fus)
        
        return logit
        
    def num_flat_features(self, X):
        size = X.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

class AConvNet_64_GRIF(nn.Module):
    """
    - 마지막에 AconvNet처럼 10channel로 바로 확률vector하는 것 대신 channel 수 넉넉하게 32로 맞춘 후 FC 한번 더 거침
    - Nx1x60x60 size Input
    """
    def __init__(self, n_classes, H_size=[16,32,64,128,64]):
        super(AConvNet_64_GRIF, self).__init__()
        
        # L1
        self.c1_tgt = nn.Conv2d(1, H_size[0], kernel_size=(5,5), stride=1, padding=2) # 64x64 
        self.c1_bn_tgt = nn.BatchNorm2d(H_size[0])
        self.p1_tgt = nn.MaxPool2d(kernel_size=(2,2), stride=2)     # 32x32 
        
        self.c1_sdw = nn.Conv2d(1, H_size[0], kernel_size=(5,5), stride=1, padding=2) # 64x64
        self.c1_bn_sdw = nn.BatchNorm2d(H_size[0])
        self.p1_sdw = nn.MaxPool2d(kernel_size=(2,2), stride=2)     # 32x32 
        
        self.GRIF1 = GRIF(H_size[0]*2, 32)

        # L2
        self.c2_tgt = nn.Conv2d(H_size[0], H_size[1], kernel_size=(5,5), stride=1, padding=2) # 32x32 
        self.c2_bn_tgt = nn.BatchNorm2d(H_size[1])
        self.p2_tgt = nn.MaxPool2d(kernel_size=(2,2), stride=2)     # 16x16
        
        self.c2_sdw = nn.Conv2d(H_size[0], H_size[1], kernel_size=(5,5), stride=1, padding=2) # 32x32 
        self.c2_bn_sdw = nn.BatchNorm2d(H_size[1])
        self.p2_sdw = nn.MaxPool2d(kernel_size=(2,2), stride=2)     # 16x16
        
        self.GRIF2 = GRIF(H_size[1]*2, 16)

        # L3
        self.c3_tgt = nn.Conv2d(H_size[1], H_size[2], kernel_size=(3,3), stride=1, padding=1) # 16x16
        self.c3_bn_tgt = nn.BatchNorm2d(H_size[2])
        self.p3_tgt = nn.MaxPool2d(kernel_size=(2,2), stride=2)     # 8x8
        
        self.c3_sdw = nn.Conv2d(H_size[1], H_size[2], kernel_size=(3,3), stride=1, padding=1) # 16x16
        self.c3_bn_sdw = nn.BatchNorm2d(H_size[2])
        self.p3_sdw = nn.MaxPool2d(kernel_size=(2,2), stride=2)     # 8x8
        
        self.GRIF3 = GRIF(H_size[2]*2, 8)    
        
        # L4
        self.c4_tgt = nn.Conv2d(H_size[2], H_size[3], kernel_size=(3,3), stride=1, padding=1) # 8x8
        self.c4_bn_tgt = nn.BatchNorm2d(H_size[3])
        self.p4_tgt = nn.MaxPool2d(kernel_size=(2,2), stride=2) # 4x4
        
        self.c4_sdw = nn.Conv2d(H_size[2], H_size[3], kernel_size=(3,3), stride=1, padding=1) # 8x8
        self.c4_bn_sdw = nn.BatchNorm2d(H_size[3])
        self.p4_sdw = nn.MaxPool2d(kernel_size=(2,2), stride=2) # 4x4
        
        self.GRIF4 = GRIF(H_size[3]*2, 4)    

        # L5
        self.c5_tgt = nn.Conv2d(H_size[3], H_size[4], kernel_size=(3,3), stride=1, padding=1) # 4x4
        self.c5_bn_tgt = nn.BatchNorm2d(H_size[4])
        self.p5_tgt = nn.MaxPool2d(kernel_size=(2,2), stride=2) # 2x2
        
        self.c5_sdw = nn.Conv2d(H_size[3], H_size[4], kernel_size=(3,3), stride=1, padding=1) # 4x4
        self.c5_bn_sdw = nn.BatchNorm2d(H_size[4])
        self.p5_sdw = nn.MaxPool2d(kernel_size=(2,2), stride=2) # 2x2
        
        self.GRIF5 = GRIF(H_size[4]*2, 2)  

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))     # Cx1x1
        self.drop = nn.Dropout(p=0.5)
        # L6
        self.fc = nn.Linear(H_size[4]*2,n_classes) # 32*3x1 -> 10x1

    def forward(self, X):
        X_tgt = X[:,0,:,:]
        X_tgt = torch.unsqueeze(X_tgt, 1)
        X_sdw = X[:,1,:,:]
        X_sdw = torch.unsqueeze(X_sdw, 1)
        
        # L1
        X_tgt = F.relu(self.c1_bn_tgt(self.c1_tgt(X_tgt)))
        X_tgt = self.p1_tgt(X_tgt)
        X_sdw = F.relu(self.c1_bn_sdw(self.c1_sdw(X_sdw)))
        X_sdw = self.p1_sdw(X_sdw)     
        (X_tgt,X_sdw) = self.GRIF1(X_tgt, X_sdw)

        # L2
        X_tgt = F.relu(self.c2_bn_tgt(self.c2_tgt(X_tgt)))
        X_tgt = self.p2_tgt(X_tgt)
        X_sdw = F.relu(self.c2_bn_sdw(self.c2_sdw(X_sdw)))
        X_sdw = self.p2_sdw(X_sdw)        
        (X_tgt,X_sdw) = self.GRIF2(X_tgt, X_sdw)

        # L3
        X_tgt = F.relu(self.c3_bn_tgt(self.c3_tgt(X_tgt)))
        X_tgt = self.p3_tgt(X_tgt)
        X_sdw = F.relu(self.c3_bn_sdw(self.c3_sdw(X_sdw)))
        X_sdw = self.p3_sdw(X_sdw)
        (X_tgt,X_sdw) = self.GRIF3(X_tgt, X_sdw)

        # L4
        X_tgt = F.relu(self.c4_bn_tgt(self.c4_tgt(X_tgt)))
        X_tgt = self.p4_tgt(X_tgt)
        X_sdw = F.relu(self.c4_bn_sdw(self.c4_sdw(X_sdw)))
        X_sdw = self.p4_sdw(X_sdw)
        (X_tgt,X_sdw) = self.GRIF4(X_tgt, X_sdw)

        # L5
        X_tgt = F.relu(self.c5_bn_tgt(self.c5_tgt(X_tgt)))
        X_tgt = self.p5_tgt(X_tgt)
        X_sdw = F.relu(self.c5_bn_sdw(self.c5_sdw(X_sdw)))
        X_sdw = self.p5_sdw(X_sdw)
        (X_tgt,X_sdw) = self.GRIF5(X_tgt, X_sdw)

        # L5
        X_tgt = self.avgpool(X_tgt)
        X_sdw = self.avgpool(X_sdw)
        X_tgt = X_tgt.view(-1,self.num_flat_features(X_tgt))
        X_sdw = X_sdw.view(-1,self.num_flat_features(X_sdw))

        # FC
        X_fus = torch.cat((X_tgt,X_sdw), dim=1)
        X_fus = self.drop(X_fus)        
        logit = self.fc(X_fus)
        
        return logit
        
    def num_flat_features(self, X):
        size = X.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

class AConvNet_64_GRIF2(nn.Module):
    """
    - 마지막에 AconvNet처럼 10channel로 바로 확률vector하는 것 대신 channel 수 넉넉하게 32로 맞춘 후 FC 한번 더 거침
    - Nx1x60x60 size Input
    """
    def __init__(self, n_classes, H_size=[16,32,64,128,64]):
        super(AConvNet_64_GRIF2, self).__init__()
        
        # L1
        self.c1_tgt = nn.Conv2d(1, H_size[0], kernel_size=(5,5), stride=1, padding=2) # 64x64 
        self.c1_bn_tgt = nn.BatchNorm2d(H_size[0])
        self.p1_tgt = nn.MaxPool2d(kernel_size=(2,2), stride=2)     # 32x32 
        
        self.c1_sdw = nn.Conv2d(1, H_size[0], kernel_size=(5,5), stride=1, padding=2) # 64x64
        self.c1_bn_sdw = nn.BatchNorm2d(H_size[0])
        self.p1_sdw = nn.MaxPool2d(kernel_size=(2,2), stride=2)     # 32x32 
        
        self.GRIF1 = GRIF2(H_size[0]*2, 32)

        # L2
        self.c2_tgt = nn.Conv2d(H_size[0], H_size[1], kernel_size=(5,5), stride=1, padding=2) # 32x32 
        self.c2_bn_tgt = nn.BatchNorm2d(H_size[1])
        self.p2_tgt = nn.MaxPool2d(kernel_size=(2,2), stride=2)     # 16x16
        
        self.c2_sdw = nn.Conv2d(H_size[0], H_size[1], kernel_size=(5,5), stride=1, padding=2) # 32x32 
        self.c2_bn_sdw = nn.BatchNorm2d(H_size[1])
        self.p2_sdw = nn.MaxPool2d(kernel_size=(2,2), stride=2)     # 16x16
        
        self.GRIF2 = GRIF2(H_size[1]*2, 16)

        # L3
        self.c3_tgt = nn.Conv2d(H_size[1], H_size[2], kernel_size=(3,3), stride=1, padding=1) # 16x16
        self.c3_bn_tgt = nn.BatchNorm2d(H_size[2])
        self.p3_tgt = nn.MaxPool2d(kernel_size=(2,2), stride=2)     # 8x8
        
        self.c3_sdw = nn.Conv2d(H_size[1], H_size[2], kernel_size=(3,3), stride=1, padding=1) # 16x16
        self.c3_bn_sdw = nn.BatchNorm2d(H_size[2])
        self.p3_sdw = nn.MaxPool2d(kernel_size=(2,2), stride=2)     # 8x8
        
        self.GRIF3 = GRIF2(H_size[2]*2, 8)    
        
        # L4
        self.c4_tgt = nn.Conv2d(H_size[2], H_size[3], kernel_size=(3,3), stride=1, padding=1) # 8x8
        self.c4_bn_tgt = nn.BatchNorm2d(H_size[3])
        self.p4_tgt = nn.MaxPool2d(kernel_size=(2,2), stride=2) # 4x4
        
        self.c4_sdw = nn.Conv2d(H_size[2], H_size[3], kernel_size=(3,3), stride=1, padding=1) # 8x8
        self.c4_bn_sdw = nn.BatchNorm2d(H_size[3])
        self.p4_sdw = nn.MaxPool2d(kernel_size=(2,2), stride=2) # 4x4
        
        self.GRIF4 = GRIF2(H_size[3]*2, 4)    

        # L5
        self.c5_tgt = nn.Conv2d(H_size[3], H_size[4], kernel_size=(3,3), stride=1, padding=1) # 4x4
        self.c5_bn_tgt = nn.BatchNorm2d(H_size[4])
        self.p5_tgt = nn.MaxPool2d(kernel_size=(2,2), stride=2) # 2x2
        
        self.c5_sdw = nn.Conv2d(H_size[3], H_size[4], kernel_size=(3,3), stride=1, padding=1) # 4x4
        self.c5_bn_sdw = nn.BatchNorm2d(H_size[4])
        self.p5_sdw = nn.MaxPool2d(kernel_size=(2,2), stride=2) # 2x2
        
        self.GRIF5 = GRIF2(H_size[4]*2, 2)  

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))     # Cx1x1
        self.drop = nn.Dropout(p=0.5)
        # L6
        self.fc = nn.Linear(H_size[4]*2,n_classes) # 32*3x1 -> 10x1

    def forward(self, X):
        X_tgt = X[:,0,:,:]
        X_tgt = torch.unsqueeze(X_tgt, 1)
        X_sdw = X[:,1,:,:]
        X_sdw = torch.unsqueeze(X_sdw, 1)
        
        # L1
        X_tgt = F.relu(self.c1_bn_tgt(self.c1_tgt(X_tgt)))
        X_tgt = self.p1_tgt(X_tgt)
        X_sdw = F.relu(self.c1_bn_sdw(self.c1_sdw(X_sdw)))
        X_sdw = self.p1_sdw(X_sdw)     
        (X_tgt,X_sdw) = self.GRIF1(X_tgt, X_sdw)

        # L2
        X_tgt = F.relu(self.c2_bn_tgt(self.c2_tgt(X_tgt)))
        X_tgt = self.p2_tgt(X_tgt)
        X_sdw = F.relu(self.c2_bn_sdw(self.c2_sdw(X_sdw)))
        X_sdw = self.p2_sdw(X_sdw)        
        (X_tgt,X_sdw) = self.GRIF2(X_tgt, X_sdw)

        # L3
        X_tgt = F.relu(self.c3_bn_tgt(self.c3_tgt(X_tgt)))
        X_tgt = self.p3_tgt(X_tgt)
        X_sdw = F.relu(self.c3_bn_sdw(self.c3_sdw(X_sdw)))
        X_sdw = self.p3_sdw(X_sdw)
        (X_tgt,X_sdw) = self.GRIF3(X_tgt, X_sdw)

        # L4
        X_tgt = F.relu(self.c4_bn_tgt(self.c4_tgt(X_tgt)))
        X_tgt = self.p4_tgt(X_tgt)
        X_sdw = F.relu(self.c4_bn_sdw(self.c4_sdw(X_sdw)))
        X_sdw = self.p4_sdw(X_sdw)
        (X_tgt,X_sdw) = self.GRIF4(X_tgt, X_sdw)

        # L5
        X_tgt = F.relu(self.c5_bn_tgt(self.c5_tgt(X_tgt)))
        X_tgt = self.p5_tgt(X_tgt)
        X_sdw = F.relu(self.c5_bn_sdw(self.c5_sdw(X_sdw)))
        X_sdw = self.p5_sdw(X_sdw)
        (X_tgt,X_sdw) = self.GRIF5(X_tgt, X_sdw)

        # L5
        X_tgt = self.avgpool(X_tgt)
        X_sdw = self.avgpool(X_sdw)
        X_tgt = X_tgt.view(-1,self.num_flat_features(X_tgt))
        X_sdw = X_sdw.view(-1,self.num_flat_features(X_sdw))

        # FC
        X_fus = torch.cat((X_tgt,X_sdw), dim=1)
        X_fus = self.drop(X_fus)        
        logit = self.fc(X_fus)
        
        return logit
        
    def num_flat_features(self, X):
        size = X.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


"""
Module
"""
class GRIF(nn.Module):
    def __init__(self, channel_in, kernel_in):
        super(GRIF, self).__init__()
        self.kernel_in = kernel_in

        if self.kernel_in != 1: # conv layer인 경우
            self.module_fuse = nn.Conv2d(channel_in, 2, (self.kernel_in, self.kernel_in), stride=1, padding=0)
        else:
            self.module_fuse = nn.Linear(channel_in, 2)
    
    def forward(self, X_tgt, X_sdw):
        w = self.module_fuse(torch.cat((X_tgt,X_sdw), dim=1))  # out: (N, 2, 1, 1)
        w = F.sigmoid(w)
        w = torch.squeeze(w)     # out: (N, 2)
        w_tgt = torch.div(w[:,0], (torch.sum(w,dim=1)+1e-10))   # out: N
        w_sdw = torch.div(w[:,1], (torch.sum(w,dim=1)+1e-10))   # out: N

        if self.kernel_in != 1:
            w_tgt = torch.unsqueeze(w_tgt, 1)
            w_tgt = torch.unsqueeze(w_tgt, 1)
            w_tgt = torch.unsqueeze(w_tgt, 1)
            w_sdw = torch.unsqueeze(w_sdw, 1)
            w_sdw = torch.unsqueeze(w_sdw, 1)
            w_sdw = torch.unsqueeze(w_sdw, 1)

            X_one = torch.ones_like(X_tgt)
            w_tgt = torch.mul(X_one, w_tgt)
            w_sdw = torch.mul(X_one, w_sdw)

            (X_tgt_weight, X_sdw_weight) = (torch.mul(X_tgt,w_tgt), torch.mul(X_sdw,w_sdw))

            assert torch.sum(torch.isnan(X_tgt_weight))==0 or torch.sum(torch.isnan(X_tgt_weight))==0
        return (X_tgt_weight, X_sdw_weight)

class GRIF2(nn.Module):
    def __init__(self, channel_in, kernel_in):
        super(GRIF2, self).__init__()
        self.kernel_in = kernel_in

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.model_fuse = nn.Linear(channel_in, 2) 
    
    def forward(self, X_tgt, X_sdw):
        w = self.avgpool(torch.cat((X_tgt, X_sdw), dim=1))   # out: (N, 2*C, 1, 1)
        w = torch.squeeze(w)     # out: (N, 2*C)
        w = self.model_fuse(w)   # out: (N, 2)
        w = F.sigmoid(w)
        
        w_tgt = torch.div(w[:,0], (torch.sum(w,dim=1)+1e-10))   # out: N
        w_sdw = torch.div(w[:,1], (torch.sum(w,dim=1)+1e-10))   # out: N

        if self.kernel_in != 1:
            w_tgt = torch.unsqueeze(w_tgt, 1)
            w_tgt = torch.unsqueeze(w_tgt, 1)
            w_tgt = torch.unsqueeze(w_tgt, 1)
            w_sdw = torch.unsqueeze(w_sdw, 1)
            w_sdw = torch.unsqueeze(w_sdw, 1)
            w_sdw = torch.unsqueeze(w_sdw, 1)

            X_one = torch.ones_like(X_tgt)
            w_tgt = torch.mul(X_one, w_tgt)
            w_sdw = torch.mul(X_one, w_sdw)

            (X_tgt_weight, X_sdw_weight) = (torch.mul(X_tgt,w_tgt), torch.mul(X_sdw,w_sdw))

            assert torch.sum(torch.isnan(X_tgt_weight))==0 or torch.sum(torch.isnan(X_tgt_weight))==0
        return (X_tgt_weight, X_sdw_weight)