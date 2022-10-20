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

# =============================================================================
# In[2] Rotation
# =============================================================================
def aug_rotate(d2_in, theta, flag_plot=False):
    """
    - Rotate the image in theta
    
    d2_in: Input MSTAR SAR
    theta: angle [degree]
    flag_plot: 1: plot the output
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
    - Rotate the image in theta
    - Input: 128x128 SAR image
    
    d2_in: Input MSTAR SAR 
    theta: angle [degree]
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
    - Rotate the image randomly
    
    d2_in: Input MSTAR SAR 
    theta_start: Angle_start [degree]
    theta_end: Angle_end [degree]
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
    - Scale the image in range or cross range domain
    
    d2_in: Input MSTAR SAR 
    R_scale: scale factor in range direction
    CR_scale: scale factor in cross range direction
    
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
    - Scale the image in range or cross range domain
    
    d2_in: Input MSTAR SAR 
    R_scale: scale factor in range direction
    CR_scale: scale factor in cross range direction
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
    - Scale the image in range or cross range domain with random scaling factor
    - Input image 128x128 with (64,64) centor
    - flag_save: Augment with 0.5 probability
    
    d2_in: Input SAR image
    R_scale_start: Scaling factor start in range domain
    R_scale_end: Scaling factor end in range domain
    CR_scale_start: Scaling factor start in cross range domain
    CR_scale_end: Scaling factor end in cross range domain
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
    - Randomly erase some patch in the given image
    
    d2_in: Input SAR image
    erase_patchsize_list: path size, e.g., 3x3, 5x5, 7x7 
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
    - Randomly erase some patch in the given image
    - flag_save: Augment with 0.5 probability
    
    d2_in: Input SAR image
    erase_patchsize_list: path size, e.g., 3x3, 5x5, 7x7
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
    - Elastic distortion for SAR image
    - S. A. Wagner, “SAR ATR by a combination of convolutional neural network and support vector machines,” 
    IEEE Trans. Aerosp. Electron. Syst., vol. 52, no. 6, pp. 2861–2872, Dec. 2016.
    - Refer to https://hj-harry.github.io/HJ-blog/2019/01/30/Elastic-distortion.html
    
    d2_in: Input SAR image in dBscale
    sigma: Std of Smoothing Kernel (higher sigma -> lower distortion)
    kernel_size: Size of Smoothing kernel (higher size -> higher distortion)    
    alpha: amplitude scaling factor (higher alpha -> higher distortion)
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
    - Elastic distortion for SAR image
    - S. A. Wagner, “SAR ATR by a combination of convolutional neural network and support vector machines,” 
    IEEE Trans. Aerosp. Electron. Syst., vol. 52, no. 6, pp. 2861–2872, Dec. 2016.
    - Refer to https://hj-harry.github.io/HJ-blog/2019/01/30/Elastic-distortion.html
    
    d2_in: Input SAR image in dBscale
    sigma: Std of Smoothing Kernel (higher sigma -> lower distortion)
    kernel_size: Size of Smoothing kernel (higher size -> higher distortion)    
    alpha: amplitude scaling factor (higher alpha -> higher distortion)
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
    