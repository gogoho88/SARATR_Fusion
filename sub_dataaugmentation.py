# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 16:59:27 2020
Preprocessing

@author: owner
"""
# =============================================================================
# In[1]
# =============================================================================
import scipy.signal as ssignal
from scipy.ndimage.interpolation import map_coordinates

import numpy as np
import matplotlib.pyplot as plt

import random

import cv2

"""
- def(함수)는 test, plot용
- class가 실제로 사용
"""
# =============================================================================
# In[2] Rotation
# =============================================================================
def aug_rotate(d2_in, theta, flag_plot=False):
    """
    - 입력 MSTAR 영상 theta만큼 rotate
    
    d2_in: 입력 MSTAR 영상
    theta: 회전 각도 [degree]
    flag_plot: output 영상 plot 할지
    """
    rot_matrix = cv2.getRotationMatrix2D((64, 64), theta, 1)

    d2_out = cv2.warpAffine(d2_in, rot_matrix, (128,128))
    
    if flag_plot==True:
        fig = plt.figure()
        plt.imshow(d2_in, cmap=plt.get_cmap('gray'))
        
        fig = plt.figure()
        plt.imshow(d2_out, cmap=plt.get_cmap('gray'))
    
    return d2_out


class Aug_Rotate(object):
    """
    - 입력 MSTAR 영상 theta만큼 rotate
    - 입력 영상 128x128
    
    d2_in: 입력 MSTAR 영상 (128x128 형태)
    theta: 회전 각도 [degree]
    
    Out: 회전된 SAR 영상
    """
    def __init__(self, theta):
        self.theta = theta
    def __call__(self, d2_in):
        if d2_in.shape[0] != 128:
            raise NameError('Invalid Input SAR size')
            
        rot_matrix = cv2.getRotationMatrix2D((64, 64), self.theta, 1)

        d2_out = cv2.warpAffine(d2_in, rot_matrix, (128,128))
        
        return d2_out

class Aug_RandomRotate(object):
    """
    - 입력 MSTAR 영상 random theta 범위만큼 rotate
    - 입력 영상 128x128
    
    d2_in: 입력 MSTAR 영상 (128x128 형태)
    theta_start: 입력 회전 각도 범위 시작 [degree]
    theta_end: 입력 회전 각도 범위 끝 [degree]
    
    Out: 회전된 SAR 영상
    """
    def __init__(self, theta_start, theta_end):        
        self.theta_start = theta_start
        self.theta_end = theta_end
        
    def __call__(self, d2_in):
        if d2_in.shape[0] != 128:
            raise NameError('Invalid Input SAR size')
        
        self.theta = random.uniform(self.theta_start, self.theta_end)
        rot_matrix = cv2.getRotationMatrix2D((64, 64), self.theta, 1)

        d2_out = cv2.warpAffine(d2_in, rot_matrix, (128,128))
        
        return d2_out

# =============================================================================
# In[2] Scaling
# =============================================================================
def aug_scaling(d2_in, R_scale=1.0, CR_scale=1.0, flag_plot=False):
    """
    - 입력 MSTAR 영상 Range or Cross range 방향으로 scaling
    
    d2_in: 입력 MSTAR 영상
    R_scale: Range 방향 scale factor
    CR_scale: Cross Range 방향 scale factor
    
    input size 홀수일 시 코두 수정 필요
    """
    
    if ((R_scale+CR_scale)/2) < 1:
        d2_scale = cv2.resize(d2_in, dsize=None, fx=R_scale, fy=CR_scale, interpolation=cv2.INTER_AREA)
    else:
        d2_scale = cv2.resize(d2_in, dsize=None, fx=R_scale, fy=CR_scale, interpolation=cv2.INTER_CUBIC)
    
    input_size = d2_in.shape   
    input_cen_y = int(input_size[0]/2)
    input_cen_x = int(input_size[1]/2)
    
    scale_size = d2_scale.shape
    dy = scale_size[0]-input_size[0]
    dx = scale_size[1]-input_size[1]
    d2_out = np.ones(shape=input_size, dtype=float)*np.min(d2_in)
    
    if dy>0:
        d2_scale = d2_scale[int(scale_size[0]/2)-input_cen_y:int(scale_size[0]/2)+input_cen_y,:]
    
    if dx>0:
        d2_scale = d2_scale[:,int(scale_size[1]/2)-input_cen_x:int(scale_size[1]/2)+input_cen_x]    
    
    if np.size(d2_scale)==input_size[0]*input_size[1]:
        d2_out = d2_scale
    else:
        scale_size = d2_scale.shape
        if (dy<0) and (dx<0):
            if ((scale_size[0]%2)==0) & ((scale_size[1]%2)==0):
                d2_out[input_cen_y-int(scale_size[0]/2):input_cen_y+int(scale_size[0]/2),
                       input_cen_x-int(scale_size[1]/2):input_cen_x+int(scale_size[1]/2)] = d2_scale
            elif ((scale_size[0]%2)==0) & ((scale_size[1]%2)==1):
                d2_out[input_cen_y-int(scale_size[0]/2):input_cen_y+int(scale_size[0]/2),
                       input_cen_x-int(scale_size[1]/2):input_cen_x+int(scale_size[1]/2)+1] = d2_scale
            elif ((scale_size[0]%2)==1) & ((scale_size[1]%2)==0):
                d2_out[input_cen_y-int(scale_size[0]/2):input_cen_y+int(scale_size[0]/2)+1,
                       input_cen_x-int(scale_size[1]/2):input_cen_x+int(scale_size[1]/2)] = d2_scale
            else:
                d2_out[input_cen_y-int(scale_size[0]/2):input_cen_y+int(scale_size[0]/2)+1,
                       input_cen_x-int(scale_size[1]/2):input_cen_x+int(scale_size[1]/2)+1] = d2_scale
        elif (dy<0) and (dx>=0):
            if (scale_size[0]%2)==0:
                d2_out[input_cen_y-int(scale_size[0]/2):input_cen_y+int(scale_size[0]/2),:] = d2_scale
            else:
                d2_out[input_cen_y-int(scale_size[0]/2):input_cen_y+int(scale_size[0]/2)+1,:] = d2_scale
        elif (dx<0) and (dy>=0):
            if (scale_size[1]%2)==0:
                d2_out[:,input_cen_x-int(scale_size[1]/2):input_cen_x+int(scale_size[1]/2)] = d2_scale
            else:
                d2_out[:,input_cen_x-int(scale_size[1]/2):input_cen_x+int(scale_size[1]/2)+1] = d2_scale
    
    if flag_plot==True:        
        fig = plt.figure()
        plt.imshow(d2_in, cmap=plt.get_cmap('gray'))

    
    return d2_out

class Aug_Scaling(object):
    """
    - 입력 MSTAR 영상 Range or Cross range 방향으로 scaling
    - 입력 영상 128x128
    - 영상 중심을 (64,64)로 가정
    
    d2_in: 입력 MSTAR 영상
    R_scale: Range 방향 scale factor
    CR_scale: Cross Range 방향 scale factor
    
    Out: scale된 SAR 영상
    """
    def __init__(self, R_scale=1.0, CR_scale=1.0):        
        self.R_scale = R_scale
        self.CR_scale = CR_scale  
        
    def __call__(self, d2_in):
        if (d2_in.shape[0]%2) != 0:
            raise NameError('Invalid Input SAR size')
            
        if ((self.R_scale+self.CR_scale)/2) < 1:
            d2_scale = cv2.resize(d2_in, dsize=None, fx=self.R_scale, fy=self.CR_scale, interpolation=cv2.INTER_AREA)
        else:
            d2_scale = cv2.resize(d2_in, dsize=None, fx=self.R_scale, fy=self.CR_scale, interpolation=cv2.INTER_CUBIC)
            
        input_size = d2_in.shape   
        input_cen_y = int(input_size[0]/2)
        input_cen_x = int(input_size[1]/2)
        
        scale_size = d2_scale.shape
        dy = scale_size[0]-input_size[0]
        dx = scale_size[1]-input_size[1]
        d2_out = np.ones(shape=input_size, dtype=float)*np.min(d2_in)
        
        if dy>0:
            d2_scale = d2_scale[int(scale_size[0]/2)-input_cen_y:int(scale_size[0]/2)+input_cen_y,:]
        
        if dx>0:
            d2_scale = d2_scale[:,int(scale_size[1]/2)-input_cen_x:int(scale_size[1]/2)+input_cen_x]    
        
        if np.size(d2_scale)==input_size[0]*input_size[1]:
            d2_out = d2_scale
        else:
            scale_size = d2_scale.shape
            if (dy<0) and (dx<0):
                if ((scale_size[0]%2)==0) & ((scale_size[1]%2)==0):
                    d2_out[input_cen_y-int(scale_size[0]/2):input_cen_y+int(scale_size[0]/2),
                           input_cen_x-int(scale_size[1]/2):input_cen_x+int(scale_size[1]/2)] = d2_scale
                elif ((scale_size[0]%2)==0) & ((scale_size[1]%2)==1):
                    d2_out[input_cen_y-int(scale_size[0]/2):input_cen_y+int(scale_size[0]/2),
                           input_cen_x-int(scale_size[1]/2):input_cen_x+int(scale_size[1]/2)+1] = d2_scale
                elif ((scale_size[0]%2)==1) & ((scale_size[1]%2)==0):
                    d2_out[input_cen_y-int(scale_size[0]/2):input_cen_y+int(scale_size[0]/2)+1,
                           input_cen_x-int(scale_size[1]/2):input_cen_x+int(scale_size[1]/2)] = d2_scale
                else:
                    d2_out[input_cen_y-int(scale_size[0]/2):input_cen_y+int(scale_size[0]/2)+1,
                           input_cen_x-int(scale_size[1]/2):input_cen_x+int(scale_size[1]/2)+1] = d2_scale
            elif (dy<0) and (dx>=0):
                if (scale_size[0]%2)==0:
                    d2_out[input_cen_y-int(scale_size[0]/2):input_cen_y+int(scale_size[0]/2),:] = d2_scale
                else:
                    d2_out[input_cen_y-int(scale_size[0]/2):input_cen_y+int(scale_size[0]/2)+1,:] = d2_scale
            elif (dx<0) and (dy>=0):
                if (scale_size[1]%2)==0:
                    d2_out[:,input_cen_x-int(scale_size[1]/2):input_cen_x+int(scale_size[1]/2)] = d2_scale
                else:
                    d2_out[:,input_cen_x-int(scale_size[1]/2):input_cen_x+int(scale_size[1]/2)+1] = d2_scale
        

        
        return d2_out

class Aug_RandomScaling(object):
    """
    - 입력 MSTAR 영상 Range or Cross range 방향으로 random scaling
    - 입력 영상 128x128
    - 영상 중심을 (64,64)로 가정
    - flag_save: 0.5의 확률로만 augment, 0.5확률로는 그대로
    
    d2_in: 입력 MSTAR 영상
    R_scale_start: Range 방향 scale factor 범위 start
    R_scale_end: Range 방향 scale factor 범위 end
    CR_scale_start: Cross Range 방향 scale factor 범위 start
    CR_scale_end: Cross Range 방향 scale factor 범위 end
    
    Out: scale된 SAR 영상
    """
    def __init__(self, R_scale_start, R_scale_end, CR_scale_start, CR_scale_end, flag_save=False):        
        self.R_scale_start = R_scale_start
        self.R_scale_end = R_scale_end
        self.CR_scale_start = CR_scale_start  
        self.CR_scale_end = CR_scale_end
        self.flag_save = flag_save
        
    def __call__(self, d2_in):
        if (d2_in.shape[0]%2) != 0:
            raise NameError('Invalid Input SAR size')
        
        if self.flag_save==True:
            flag_exe = random.randint(0,1)
        else:
            flag_exe = 1
        
        if flag_exe:
            R_scale = random.uniform(self.R_scale_start, self.R_scale_end)
            CR_scale = random.uniform(self.CR_scale_start, self.CR_scale_start)
            
            if ((R_scale+CR_scale)/2) < 1:
                d2_scale = cv2.resize(d2_in, dsize=None, fx=R_scale, fy=CR_scale, interpolation=cv2.INTER_AREA)
            else:
                d2_scale = cv2.resize(d2_in, dsize=None, fx=R_scale, fy=CR_scale, interpolation=cv2.INTER_CUBIC)
                
            input_size = d2_in.shape   
            input_cen_y = int(input_size[0]/2)
            input_cen_x = int(input_size[1]/2)
            
            scale_size = d2_scale.shape
            dy = scale_size[0]-input_size[0]
            dx = scale_size[1]-input_size[1]
            d2_out = np.ones(shape=input_size, dtype=float)*np.min(d2_in)
            
            if dy>0:
                d2_scale = d2_scale[int(scale_size[0]/2)-input_cen_y:int(scale_size[0]/2)+input_cen_y,:]
            
            if dx>0:
                d2_scale = d2_scale[:,int(scale_size[1]/2)-input_cen_x:int(scale_size[1]/2)+input_cen_x]    
            
            if np.size(d2_scale)==input_size[0]*input_size[1]:
                d2_out = d2_scale
            else:
                scale_size = d2_scale.shape
                if (dy<0) and (dx<0):
                    if ((scale_size[0]%2)==0) & ((scale_size[1]%2)==0):
                        d2_out[input_cen_y-int(scale_size[0]/2):input_cen_y+int(scale_size[0]/2),
                               input_cen_x-int(scale_size[1]/2):input_cen_x+int(scale_size[1]/2)] = d2_scale
                    elif ((scale_size[0]%2)==0) & ((scale_size[1]%2)==1):
                        d2_out[input_cen_y-int(scale_size[0]/2):input_cen_y+int(scale_size[0]/2),
                               input_cen_x-int(scale_size[1]/2):input_cen_x+int(scale_size[1]/2)+1] = d2_scale
                    elif ((scale_size[0]%2)==1) & ((scale_size[1]%2)==0):
                        d2_out[input_cen_y-int(scale_size[0]/2):input_cen_y+int(scale_size[0]/2)+1,
                               input_cen_x-int(scale_size[1]/2):input_cen_x+int(scale_size[1]/2)] = d2_scale
                    else:
                        d2_out[input_cen_y-int(scale_size[0]/2):input_cen_y+int(scale_size[0]/2)+1,
                               input_cen_x-int(scale_size[1]/2):input_cen_x+int(scale_size[1]/2)+1] = d2_scale
                elif (dy<0) and (dx>=0):
                    if (scale_size[0]%2)==0:
                        d2_out[input_cen_y-int(scale_size[0]/2):input_cen_y+int(scale_size[0]/2),:] = d2_scale
                    else:
                        d2_out[input_cen_y-int(scale_size[0]/2):input_cen_y+int(scale_size[0]/2)+1,:] = d2_scale
                elif (dx<0) and (dy>=0):
                    if (scale_size[1]%2)==0:
                        d2_out[:,input_cen_x-int(scale_size[1]/2):input_cen_x+int(scale_size[1]/2)] = d2_scale
                    else:
                        d2_out[:,input_cen_x-int(scale_size[1]/2):input_cen_x+int(scale_size[1]/2)+1] = d2_scale
                        
        else:
             d2_out = d2_in   
                        
        return d2_out

# =============================================================================
# In[2]
# =============================================================================
def aug_randomerase(d2_in, erase_patchsize_list = [3,5,7], flag_plot=False):
    """
    - 입력 MSTAR 영상 내 image를 random하게 제거
    
    d2_in: 입력 MSTAR 영상
    erase_patchsize_list: 3x3, 5x5, 7x7 등 제거하는 patch
    """
    d2_out = d2_in.copy()
    
    erase_range = np.where(d2_in>0)
    erase_x_range = erase_range[1]
    erase_y_range = erase_range[0]
    
    flag_erase = random.randint(0,1)
    if flag_erase == True:
        ind_patch = random.randint(0,2)
        erase_patchsize = erase_patchsize_list[ind_patch]
        erase_point_ind = random.randint(0, len(erase_range[0])-1)
        erase_point_x = erase_x_range[erase_point_ind]
        erase_point_y = erase_y_range[erase_point_ind]
        
        d2_out[erase_point_y-int(erase_patchsize/2):erase_point_y+int(erase_patchsize/2)+1
            ,erase_point_x-int(erase_patchsize/2):erase_point_x+int(erase_patchsize/2)+1] = float(0)
    
    if flag_plot==True:       
            
        fig = plt.figure()
        plt.imshow(d2_in, cmap=plt.get_cmap('gray'))
    
        fig = plt.figure()
        plt.imshow(d2_out, cmap=plt.get_cmap('gray'))
    
    return d2_out


class Aug_RandomErasing(object):
    """
    - 입력 MSTAR 영상 image를 random하게 제거
    - 입력 영상 128x128 아니어도 됨
    - 0.5의 확률로만 augment, 0.5확률로는 그대로
    
    d2_in: 입력 MSTAR 영상
    erase_patchsize_list: 3x3, 5x5, 7x7 등 제거하는 patch
    
    Out: 특정 부분이 제거된 영상 (0.5) or 그대로 (0.5)
    """
    def __init__(self, erase_patchsize_list = [3,5,7]):        
        self.erase_patchsize_list = erase_patchsize_list
        
    def __call__(self, d2_in):
        if len(d2_in.shape) != 2:
            raise NameError('Invalid Input SAR size')
            
        d2_out = d2_in.copy()
    
        erase_range = np.where(d2_in>0)
        erase_x_range = erase_range[1]
        erase_y_range = erase_range[0]
        
        flag_erase = random.randint(0,1)
        if flag_erase == True:
            ind_patch = random.randint(0,2)
            erase_patchsize = self.erase_patchsize_list[ind_patch]
            erase_point_ind = random.randint(0, len(erase_range[0])-1)
            erase_point_x = erase_x_range[erase_point_ind]
            erase_point_y = erase_y_range[erase_point_ind]
            
            d2_out[erase_point_y-int(erase_patchsize/2):erase_point_y+int(erase_patchsize/2)+1
                ,erase_point_x-int(erase_patchsize/2):erase_point_x+int(erase_patchsize/2)+1] = float(0)
        
        return d2_out
# =============================================================================
# In[2]
# =============================================================================

def aug_elastic_distortion(d2_in, sigma=15, kernel_size=5, alpha=1, flag_plot=False):
    """
    - 입력 MSTAR 영상 elastic distortion을 통한 data augmentation
    - S. A. Wagner, “SAR ATR by a combination of convolutional neural network and support vector machines,” 
    IEEE Trans. Aerosp. Electron. Syst., vol. 52, no. 6, pp. 2861–2872, Dec. 2016.
    - https://hj-harry.github.io/HJ-blog/2019/01/30/Elastic-distortion.html
    참조
    
    d2_in: 입력 MSTAR 영상 in dBscale
    sigma: Smoothing Kernel의 std (클수록 왜곡 적어짐)
    kernel_size: Smoothing kernel 크기 (클수록 target도 많이 smooth되는듯)    
    alpha: amplitude scaling factor (클수록 왜곡 엄청심해짐)
    """
    [sy,sx] = d2_in.shape
    dx = np.random.uniform(-1, 1, (sx,sy))
    dy = np.random.uniform(-1, 1, (sx,sy))
    
    dx_gauss = cv2.GaussianBlur(dx, (kernel_size,kernel_size), sigma)
    dy_gauss = cv2.GaussianBlur(dy, (kernel_size,kernel_size), sigma)
    
    gauss_norm = np.sqrt(dx_gauss**2 + dy_gauss**2)
    
    dx_norm = (alpha*dx_gauss)/gauss_norm
    dy_norm = (alpha*dy_gauss)/gauss_norm
    
    
    indy, indx = np.indices((sx, sy), dtype=np.float32)
    
    map_x = dx_norm + indx
    map_x = map_x.reshape(sy, sx).astype(np.float64)
    map_y = dy_norm + indy
    map_y = map_y.reshape(sy, sx).astype(np.float64)
    
    indices = (map_y), (map_x)
    
    d2_out = map_coordinates(d2_in, indices, order=1, mode='reflect')
    
    if flag_plot==True:
        fig = plt.figure()
        plt.imshow(d2_in, cmap=plt.get_cmap('gray'))
        
        fig = plt.figure()
        plt.imshow(d2_out, cmap=plt.get_cmap('gray'))
        
    return d2_out

class Aug_RandomElasticDistortion(object):
    """
    - 입력 MSTAR 영상 elastic distortion을 통한 data augmentation
    - S. A. Wagner, “SAR ATR by a combination of convolutional neural network and support vector machines,” 
    IEEE Trans. Aerosp. Electron. Syst., vol. 52, no. 6, pp. 2861–2872, Dec. 2016.
    - https://hj-harry.github.io/HJ-blog/2019/01/30/Elastic-distortion.html
    참조
    - 입력 영상 128x128 아니어도 됨
    - 0.5의 확률로만 augment, 0.5확률로는 그대로
    
    d2_in: 입력 MSTAR 영상 in dBscale
    sigma_start: Smoothing Kernel의 std 범위 시작 (클수록 왜곡 적어짐)
    sigma_end: Smoothing Kernel의 std 범위 꿑 (클수록 왜곡 적어짐)
    kernel_size_start: Smoothing kernel 크기 범위 시작 (클수록 target도 많이 smooth되는듯)    
    kernel_size_end: Smoothing kernel 크기 범위 끝 (클수록 target도 많이 smooth되는듯)    
    alpha: amplitude scaling factor (클수록 왜곡 엄청심해짐)
    """
    def __init__(self, sigma_start, sigma_end, kernel_size_start=5, kernel_size_end=5, alpha_start=1, alpha_end=1):        
        self.sigma_start = sigma_start
        self.sigma_end = sigma_end
        self.kernel_size_start = kernel_size_start
        self.kernel_size_end = kernel_size_end
        self.alpha_start = alpha_start
        self.alpha_end = alpha_end
        
    def __call__(self, d2_in):
        if len(d2_in.shape) != 2:
            raise NameError('Invalid Input SAR size')
        
        flag_exe = random.randint(0,1)
        
        if flag_exe:
            sigma = random.uniform(self.sigma_start, self.sigma_end)
            alpha = random.uniform(self.alpha_start, self.alpha_end)
            kernel_size = random.randint(int(self.kernel_size_start), int(self.kernel_size_end))
            
            [sy,sx] = d2_in.shape
            dx = np.random.uniform(-1, 1, (sx,sy))
            dy = np.random.uniform(-1, 1, (sx,sy))
            
            dx_gauss = cv2.GaussianBlur(dx, (kernel_size,kernel_size), sigma)
            dy_gauss = cv2.GaussianBlur(dy, (kernel_size,kernel_size), sigma)
            
            gauss_norm = np.sqrt(dx_gauss**2 + dy_gauss**2)
            
            dx_norm = (alpha*dx_gauss)/gauss_norm
            dy_norm = (alpha*dy_gauss)/gauss_norm            
            
            indy, indx = np.indices((sx, sy), dtype=np.float32)
            
            map_x = dx_norm + indx
            map_x = map_x.reshape(sy, sx).astype(np.float64)
            map_y = dy_norm + indy
            map_y = map_y.reshape(sy, sx).astype(np.float64)
            
            indices = (map_y), (map_x)
            
            d2_out = map_coordinates(d2_in, indices, order=1, mode='reflect')
        else:
            d2_out = d2_in
                    
        return d2_out 
    