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

import numpy as np
import matplotlib.pyplot as plt
import random

import cv2
import scipy.ndimage as ndi

from scipy.interpolate import interp2d
# =============================================================================
# In[2]
# =============================================================================
def prepro_DB(d2_in):
    if np.min(d2_in) ==0:
        d2_temp = d2_in.copy()
        d2_temp[d2_temp == 0] = np.nan
        min_temp = np.nanmin(d2_temp)
        d2_in[d2_in == 0] = min_temp
        
        
    d2_out = 20*np.log10(d2_in) # dB scale        
    return d2_out



# =============================================================================
# In[3]
# =============================================================================
def prepro_targetseg(d2_in, th_bin_min=30, th_target_std=50, th_count=1800, tgt_range=[30,100,30,100], flag_peri=True, morp_ind = 1, flag_plot=False):
    """
    - 입력 MSTAR 영상으로부터 표적 segment
    - R. Meth, “Target/shadow segmentation and aspect estimation in synthetic aperture radar imagery,” 
    in Proc. SPIE 3370, Algorithms for Synthetic Aperture Radar Imagery V, 1998, vol. 23, no. 5, pp. 188–196.   
    - F. Zhou, L. W
    ang, X. Bai, and Y. Hui, “SAR ATR of Ground Vehicles Based on LM-BN-CNN,” 
    IEEE Trans. Geosci. Remote Sens., vol. 56, no. 12, pp. 7282–7293, Dec. 2018.
    참조
    
    th_count: 위에서부터 순서대로 몇 pixel 넘어가면 클러터라고 생각되는지
    tgt_range: 표적이 100% 있다고 생각되는 지역
    """
    # d2_in: log-scale input
    
    # 1. Normalize
    d2_norm = (d2_in-np.min(d2_in))/(np.max(d2_in)-np.min(d2_in))
    
    # 2. Denoise
    d2_denon = ssignal.wiener(d2_norm)
    
    # 3. Masking
    hist, bin_edges = np.histogram(d2_denon, bins=128)
    bin_sum = 0
    for bin_ind in range(len(hist)):
        bin_num = hist[len(hist)-1-bin_ind]
        bin_sum = bin_sum + bin_num
        if bin_num>th_bin_min:
            img_ind = (bin_edges[len(hist)-1-bin_ind]<=d2_denon) & (d2_denon<bin_edges[len(hist)-bin_ind])
            cord_y, cord_x = np.where(img_ind==True)
            cord_std = np.sqrt(np.std(cord_x)**2+np.std(cord_y)**2)
            
            if (cord_std>th_target_std) or (bin_sum>th_count):
                d2_mask = (bin_edges[len(hist)-1-bin_ind+1]<=d2_denon)
                cord_y_total, cord_x_total = np.where(d2_mask==True)
                break
    
    # 위치 정보 이용하여 mask 조정
    d2_mask[0:tgt_range[0],:] = False
    d2_mask[tgt_range[1]:128,:] = False
    d2_mask[:,0:tgt_range[2]] = False
    d2_mask[:,tgt_range[3]:128] = False
            
    # 4. Morphology (dilate and erode)
    morp_kernel1 = np.array(([0,0,0,1,0,0,0],
                       [0,0,1,1,1,0,0],
                       [0,1,1,1,1,1,0],
                       [1,1,1,1,1,1,1],
                       [0,1,1,1,1,1,0],
                       [0,0,1,1,1,0,0],
                       [0,0,0,1,0,0,0]), np.uint8)

    morp_kernel2 = np.array(([0,1,1,1,0],
                           [1,1,1,1,1],
                           [1,1,1,1,1],
                           [1,1,1,1,1],
                           [0,1,1,1,0]), np.uint8)
    
    morp_kernel3 = np.array(([1,1,1],
                             [1,1,1],
                             [1,1,1]), np.uint8)  #4x4
    
    if morp_ind == 1:
        morp_kernel = morp_kernel2
    elif morp_ind == 0:
        morp_kernel = morp_kernel1
    else:
        morp_kernel = morp_kernel3    
    
    d2_morp = cv2.dilate(np.float32(d2_mask), morp_kernel, iterations = 1)    
    d2_morp = cv2.erode(d2_morp, morp_kernel, iterations = 1)
    
    # 5. Contour and Find Maximum contour
    contours, _ = cv2.findContours(np.uint8(d2_morp), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours)>3:
        cont_size = np.zeros(len(contours))
        for i in range(len(contours)):
            cont_temp = contours[i]
            cont_size[i] = contours[i].shape[0]
        
        cont_sort_ind = np.argsort(cont_size)
        cont_ind_cand = cont_sort_ind[-3:]
            
        cont_area = np.zeros(3)
        for i in range(3):
            ind_temp = cont_ind_cand[i]
            cont_temp = contours[ind_temp]
            cont_area[i] = cv2.contourArea(cont_temp)
        
        ###########################################################################
#         v1: max area만 선택
        cont_right = np.argmax(cont_area)    
        cont_right_ind = cont_ind_cand[cont_right]
        
        cont_main = contours[cont_right_ind]
        cord_x_cont = cont_main[:,0,0]
        cord_y_cont = cont_main[:,0,1]
        
        d2_adj = np.zeros([128,128])
        for i in range(128):
            for j in range(128):
                flag_cont = cv2.pointPolygonTest(cont_main, (j,i), False)
                if flag_cont>=0:
                    d2_adj[i,j] = flag_cont
        
        ###########################################################################
        # v2: max area + (그냥 area 70넘고 중심 64로부터 +-15 이내에 있으면 다 선택)
#        cont_right = np.where(cont_area>70)
#        cont_right_ind = cont_ind_cand[cont_right]
#        cont_max = np.argmax(cont_area)
#        cont_max_ind = cont_ind_cand[cont_max]
#        d2_adj = np.zeros([128,128])
#        cord_x_cont = np.array([], dtype=int)
#        cord_y_cont = np.array([], dtype=int)
#        for k in cont_right_ind:
#            cont_main = contours[k]
#            cont_cen = np.mean(cont_main,axis=0)
#            if ((49<cont_cen[0,0]<79) and (49<cont_cen[0,1]<79)) or (k==cont_max_ind):
#                cord_x_cont = np.concatenate((cord_x_cont,cont_main[:,0,0]))
#                cord_y_cont = np.concatenate((cord_y_cont,cont_main[:,0,1]))
#                for i in range(128):
#                    for j in range(128):
#                        flag_cont = cv2.pointPolygonTest(cont_main, (j,i), False)
#                        if flag_cont>=0:
#                            d2_adj[i,j] = flag_cont
        
    
    elif len(contours)==0:
        d2_adj = np.zeros([128,128])
        cord_x_cont = 0
        cord_y_cont = 0
    else:
        cont_size = np.zeros(len(contours))
        for i in range(len(contours)):
            cont_size[i] = contours[i].shape[0]
        cont_max_ind = np.argmax(cont_size)  
        cont_main = contours[cont_max_ind]
        cord_x_cont = cont_main[:,0,0]
        cord_y_cont = cont_main[:,0,1]
        
        d2_adj = np.zeros([128,128])
        for i in range(128):
            for j in range(128):
                flag_cont = cv2.pointPolygonTest(cont_main, (j,i), False)
                if flag_cont>=0:
                    d2_adj[i,j] = flag_cont
        
    # 테두리도 포함
    if flag_peri==True:
        d2_adj[cord_y_cont, cord_x_cont] = 1.
        
            
    # 6. Mask
    d2_out = d2_norm*d2_adj
    
    if flag_plot==True:
#        fig = plt.figure()
#        plt.imshow(d2_denon, cmap=plt.get_cmap('gray'))
#               
##        fig = plt.figure()
##        plt.hist(np.resize(d2_norm,[16384]), bins=128)
##        
##        fig = plt.figure()
##        plt.hist(np.resize(d2_denon,[16384]), bins=128)
##        
#        fig = plt.figure()
#        plt.imshow(d2_norm, cmap=plt.get_cmap('gray'))
#        plt.plot(cord_x_total,cord_y_total,'r.')
#        
#        
##        fig = plt.figure()
##        plt.imshow(d2_count, cmap=plt.get_cmap('gray'))
#        
#        fig = plt.figure()
#        plt.imshow(d2_morp, cmap=plt.get_cmap('gray'))
##        
##        fig = plt.figure()
##        plt.imshow(d2_morp, cmap=plt.get_cmap('gray'))
##        plt.plot(cord_x_cont,cord_y_cont,'b.')
#        
##        fig = plt.figure()
##        plt.imshow(d2_adj, cmap=plt.get_cmap('gray'))
        
        fig = plt.figure()
        plt.imshow(d2_norm, cmap=plt.get_cmap('gray'))
        plt.plot(cord_x_cont,cord_y_cont,'b.')
        
#        fig = plt.figure()
#        plt.imshow(d2_out, cmap=plt.get_cmap('gray'))
    
    return d2_out, d2_adj, cord_x_cont, cord_y_cont

# =============================================================================
# In[4]
# =============================================================================
def prepro_shadowseg(d2_in, th_shadow_per=1/3, th_count=15, sdw_range=[5,80,30,99], flag_peri=True, flag_plot=False):
    """
    - 입력 MSTAR 영상으로부터 shadow segment
    - M. Chang and X. You, “Target recognition in SAR images based on information-decoupled representation,”
    Remote Sens., vol. 10, no. 1, 2018.
    참조
    
    d2_in: 입력 MSTAR 영상 in dBscale
    th_shadow_per: 영상 amplitude의 밑에서 얼만큼을 shadow 및 clutter로 볼 것인지
    th_count: Count filter에서 5x5 filter 내 몇개 이상의 pixel이 있을 시 shadow로 볼 것인지
    sdw_range: shadow가 있을 것이라 예상되는 범위[i: 1~2 , j:3~4]
    """
    
    #1 th를 통해 큰 값 제거
    d2_norm = (d2_in-np.min(d2_in))/(np.max(d2_in)-np.min(d2_in))
    d1_norm = np.reshape(d2_norm, [16384])
    d1_norm = np.sort(d1_norm)
    
    th_shadow = d1_norm[np.int(16384*th_shadow_per)]
    d2_afterth = np.where(d2_norm>th_shadow, 0, 1) #th를 넘는 것은 0으로 하고, 나머지는 그대로
    
    #2 Count filter
    d2_count = np.zeros((128,128), dtype=float)
    for i in np.arange(sdw_range[0],sdw_range[1]+1):
        for j in np.arange(sdw_range[2],sdw_range[3]+1):
            window_count = d2_afterth[np.max([0,i-2]):np.min([128,i+3]), np.max([0,j-2]):np.min([128,j+3])]
            if np.sum(window_count>0) > th_count:
                d2_count[i,j] = d2_afterth[i,j]
    
    
    #3 Morphology close
    morp_kernel3 = np.array(([1,1,1],
                             [1,1,1],
                             [1,1,1]), np.uint8)  #4x4
    d2_morp = cv2.morphologyEx(d2_count, cv2.MORPH_CLOSE, morp_kernel3)
    
    #4 가장 큰 Countour만 선택
    contours, _ = cv2.findContours(np.uint8(d2_morp), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cont_size = np.zeros(len(contours))
    for i in range(len(contours)):
        cont_size[i] = contours[i].shape[0]
    cont_max_ind = np.argmax(cont_size)  
    cont_main = contours[cont_max_ind]
    cord_x_cont = cont_main[:,0,0]
    cord_y_cont = cont_main[:,0,1]
    
    d2_adj = np.zeros([128,128])
    for i in range(128):
        for j in range(128):
            flag_cont = cv2.pointPolygonTest(cont_main, (j,i), False)
            if flag_cont>=0:
                d2_adj[i,j] = flag_cont

    # 테두리도 포함
    if flag_peri==True:
        d2_adj[cord_y_cont, cord_x_cont] = 1.
    
    #5 Mast
    d2_out = d2_norm*d2_adj

    
    if flag_plot==True:
#        fig = plt.figure()
#        plt.imshow(d2_norm, cmap=plt.get_cmap('gray'))
#        
#        fig = plt.figure()
#        plt.imshow(d2_afterth, cmap=plt.get_cmap('gray'))
#        
#        fig = plt.figure()
#        plt.imshow(d2_count, cmap=plt.get_cmap('gray'))
#        
#        fig = plt.figure()
#        plt.imshow(d2_morp, cmap=plt.get_cmap('gray'))
#        
#        fig = plt.figure()
#        plt.imshow(d2_morp, cmap=plt.get_cmap('gray'))
#        plt.plot(cord_x_cont,cord_y_cont,'r.')
#        
#        fig = plt.figure()
#        plt.imshow(d2_adj, cmap=plt.get_cmap('gray'))
        
        fig = plt.figure()
        plt.imshow(d2_norm, cmap=plt.get_cmap('gray'))
        plt.plot(cord_x_cont,cord_y_cont,'r.')
        
#        fig = plt.figure()
#        plt.imshow(d2_out, cmap=plt.get_cmap('gray'))
    
    
    return d2_out, d2_adj, cord_x_cont, cord_y_cont

# =============================================================================
# In[5]
# =============================================================================
def prepro_crop(d2_in, SAR_center=[64,64], crop_size=[64,64]):
    """
    - 입력 MSTAR 영상 crop
    
    d2_in: 입력 MSTAR 영상 (HxW 꼴)
    SAR_center: Crop 영상의 중심 (y,x) 좌표
    crop_size: Crop 영상의 (height,width)
    """
    if ((SAR_center[0] < int(crop_size[0]/2)) or (SAR_center[1] < int(crop_size[1]/2))
        or ((SAR_center[0]+int(crop_size[0]/2)) > d2_in.shape[0]) or ((SAR_center[1]+int(crop_size[1]/2)) > d2_in.shape[1])):
        print('crop error occur')
        raise Exception('error')
    else:
        d2_out = d2_in[SAR_center[0]-int(crop_size[0]/2):SAR_center[0]+int(crop_size[0]/2),
                        SAR_center[1]-int(crop_size[1]/2):SAR_center[1]+int(crop_size[1]/2)]
    
    return d2_out

class Prepro_Crop(object):
    """
    - 입력 MSTAR 영상 crop
    
    d2_in: 입력 MSTAR 영상
    SAR_center: Crop 영상의 중심 (y,x) 좌표
    crop_size: Crop 영상의 (height,width)
    """
    def __init__(self, SAR_center, crop_size):
        self.SAR_center = SAR_center
        self.crop_size = crop_size
        
    def __call__(self, d2_in):
        if (self.crop_size[0]%2)==0:
            d2_out = d2_in[self.SAR_center[0]-int(self.crop_size[0]/2):self.SAR_center[0]+int(self.crop_size[0]/2),
                       self.SAR_center[1]-int(self.crop_size[1]/2):self.SAR_center[1]+int(self.crop_size[1]/2)]
        else:
            d2_out = d2_in[self.SAR_center[0]-int(self.crop_size[0]/2):self.SAR_center[0]+int(self.crop_size[0]/2)+1,
                       self.SAR_center[1]-int(self.crop_size[1]/2):self.SAR_center[1]+int(self.crop_size[1]/2)+1]    
            
        return d2_out
# =============================================================================
# In[5]
# =============================================================================
def plot_segmentation(SAR_data, SAR_label, label_list):
    d1_datlist = SAR_label
    data_list = SAR_data
    list_target = label_list
    
    sample_dat_list = []
    sample_idx_list = []
    sample_tgt_xcord_list = []
    sample_tgt_ycord_list = []
    sample_sdw_xcord_list = []
    sample_sdw_ycord_list = []
    sample_tgt_list = []
    sample_sdw_list = []
    
    for i in range(len(list_target)):
        target_name = list_target[i]
        d1_idx = np.arange(d1_datlist.index(target_name), d1_datlist.index(target_name)+d1_datlist.count(target_name))
        
        sample_idx = random.choice(d1_idx)
        sample_dat = data_list[:,sample_idx].copy()
        sample_dat = np.resize(sample_dat, [128,128])
        
        sample_dat_db = prepro_DB(sample_dat)
        sample_dat_norm = (sample_dat_db-np.min(sample_dat_db))/(np.max(sample_dat_db)-np.min(sample_dat_db))
        tgt_seg, tgt_xcord, tgt_ycord = prepro_targetseg(sample_dat_db)
        sdw_seg, sdw_xcord, sdw_ycord = prepro_shadowseg(sample_dat_db)
        
        sample_dat_list.append(sample_dat_norm)
        sample_idx_list.append(sample_idx)
        sample_tgt_xcord_list.append(tgt_xcord)
        sample_tgt_ycord_list.append(tgt_ycord)
        sample_sdw_xcord_list.append(sdw_xcord)
        sample_sdw_ycord_list.append(sdw_ycord)
        sample_tgt_list.append(tgt_seg)
        sample_sdw_list.append(sdw_seg)
    

    fig, axes = plt.subplots(10,4, figsize=(14,12))   
    for i in range(len(list_target)):
        axes[i,0].imshow(sample_dat_list[i], cmap=plt.get_cmap('gray'))
        axes[i,1].imshow(sample_dat_list[i], cmap=plt.get_cmap('gray'))
        axes[i,1].plot(sample_tgt_xcord_list[i],sample_tgt_ycord_list[i],'b.')
        axes[i,1].plot(sample_sdw_xcord_list[i],sample_sdw_ycord_list[i],'r.')
        axes[i,2].imshow(sample_tgt_list[i], cmap=plt.get_cmap('gray'))
        axes[i,3].imshow(sample_sdw_list[i], cmap=plt.get_cmap('gray'))
        
    for i in range(len(list_target)):
        fig, axes = plt.subplots(1,2, figsize=(12,8))  
        axes[0].imshow(sample_dat_list[i], cmap=plt.get_cmap('gray'))
        axes[1].imshow(sample_dat_list[i], cmap=plt.get_cmap('gray'))
        axes[1].plot(sample_tgt_xcord_list[i],sample_tgt_ycord_list[i],'b.')
        axes[1].plot(sample_sdw_xcord_list[i],sample_sdw_ycord_list[i],'r.')
        
# =============================================================================
# In[5] Polar mapping
# =============================================================================
def prepro_polarmapping(d2_in, flag_shadow, Nr=60, Rmin=0, Rmax=40, 
                        Ntheta=60, thetamin=0, thetamax=2*np.pi, flag_center=False, flag_plot=False):
    """
    - 입력 MSTAR 영상 을 polar mapping 수행 
    
    d2_in: 입력 MSTAR 영상 (semented) (128x128 꼴)
    flag_shadow: True-shadow를 polar 변환 / False-target을 polar 변환
    Nr: R(반지름) 방향으로 몇개를 sampling 할지
    Rmin: 중심으로부터 Rmin부터 sampling 시작
    Rmax: Rmin부터 Rmax까지 Nr만큼 sampling Rmax-40은 대략 표적이 최대로 분포한 범위 
    60x60에서는 아래위가 중심으로부터 30인데 대각선은 42.42인 점 고려
    
    d2_out: 출력 mapping 영상 (NrxNtheta 꼴)
    """
    R_vec = np.linspace(Rmin, Rmax, Nr)
    theta_vec = np.linspace(thetamin, thetamax, Ntheta)
    
    if flag_shadow==False:
        # 중심 계산기 side 0으로 해놓고 계산하는 것 추가
        if np.sum(flag_center)==False:
            center_tgt = ndi.center_of_mass(d2_in)
        else:
            center_tgt = flag_center
        
        cord_cart_x = np.arange(0,d2_in.shape[1], dtype=float)
        cord_cart_y = np.arange(0,d2_in.shape[0], dtype=float)
        
        cord_polar_x = np.zeros([Nr,Ntheta], dtype=float)
        cord_polar_y = np.zeros([Nr,Ntheta], dtype=float)
        for i in range(len(R_vec)):
            for j in range(len(theta_vec)):
                cord_polar_x[i,j] = center_tgt[1] + R_vec[i]*np.cos(-theta_vec[j])
                cord_polar_y[i,j] = center_tgt[0] + R_vec[i]*np.sin(-theta_vec[j])
        
        f = interp2d(cord_cart_x, cord_cart_y, d2_in, kind='linear')    
        d2_polar = np.zeros([Nr,Ntheta], dtype=float)
        for i in range(len(R_vec)):
            for j in range(len(theta_vec)):
                d2_polar[i,j] = f(cord_polar_x[i,j], cord_polar_y[i,j])
        
        if flag_plot==True:
            fig = plt.figure()
            plt.imshow(d2_in, cmap=plt.get_cmap('gray'))
            plt.scatter(cord_polar_x, cord_polar_y,  color='blue', s=3,  marker='.')
            plt.scatter(center_tgt[1],center_tgt[0], color='red', s=50,  marker='*')
            
            fig = plt.figure()
            plt.imshow(d2_polar, cmap=plt.get_cmap('gray'))
    else:
        d2_sdw_adj = np.zeros(d2_in.shape, dtype=float)
        d2_sdw_adj[-96:,:] = d2_in[0:96,:] # 32만큼 밑으로 밀어냄
        d2_sdw_adj[0:32,:] = d2_in[-32:,:]
        
        if np.sum(flag_center)==False:
            center_sdw = ndi.center_of_mass(d2_sdw_adj)
        else:
            center_sdw = flag_center.copy()
            center_sdw[0] = center_sdw[0]+32 # 밀어냈으므로 center도 32만큼 같이 밀어냄
        
        cord_cart_x = np.arange(0,d2_in.shape[1], dtype=float)
        cord_cart_y = np.arange(0,d2_in.shape[0], dtype=float)
        
        cord_polar_x = np.zeros([Nr,Ntheta], dtype=float)
        cord_polar_y = np.zeros([Nr,Ntheta], dtype=float)
        for i in range(len(R_vec)):
            for j in range(len(theta_vec)):
                cord_polar_x[i,j] = center_sdw[1] + R_vec[i]*np.cos(-theta_vec[j])
                cord_polar_y[i,j] = center_sdw[0] + R_vec[i]*np.sin(-theta_vec[j])
        
        f = interp2d(cord_cart_x, cord_cart_y, d2_sdw_adj, kind='linear')
        d2_polar = np.zeros([Nr,Ntheta], dtype=float)
        for i in range(len(R_vec)):
            for j in range(len(theta_vec)):
                d2_polar[i,j] = f(cord_polar_x[i,j], cord_polar_y[i,j])
        
        if flag_plot==True:
            fig = plt.figure()
            plt.imshow(d2_in, cmap=plt.get_cmap('gray'))
            
            fig = plt.figure()
            plt.imshow(d2_sdw_adj, cmap=plt.get_cmap('gray'))
            plt.scatter(cord_polar_x, cord_polar_y,  color='blue', s=3,  marker='.')
            plt.scatter(center_sdw[1],center_sdw[0], color='red', s=50,  marker='*')
            
            fig = plt.figure()
            plt.imshow(d2_polar, cmap=plt.get_cmap('gray'))
                
    return d2_polar

