"""
Created on Fri Mar 27 21:54:13 2020

Refer to
https://github.com/hamza-latif/MSTAR_tensorflow/blob/master/readmstar.py

Load MSTAR Dataset

"""
# =============================================================================
# In[1]
# =============================================================================
import numpy as np
import _pickle as pickle
import sys
import os
from fnmatch import fnmatch

from sub_preprocessing import prepro_DB
from sub_preprocessing import prepro_targetseg
from sub_preprocessing import prepro_shadowseg

# =============================================================================
# In[2]
# =============================================================================
def readMSTARFile(filename):
# raw_input('Enter the mstar file to read: ')

	#print filename

    f = open(filename, 'rb')

    a = ''.encode()

    phoenix_header = []

    while 'PhoenixHeaderVer'.encode() not in a:
        a = f.readline()

    a = f.readline()

    while 'EndofPhoenixHeader'.encode() not in a:
        phoenix_header.append(a)
        a = f.readline()

    data = np.fromfile(f, dtype='>f4')

    #	print(data.shape)
    #
    #	magdata = data[:128*128]
    #	phasedata = data[128*128:]
    
    #	if you want to print an image
    #    imdata = magdata*255
    #
    #	imdata = imdata.astype('uint8')

    targetSerNum = '-'

    for line in phoenix_header:
        #print line
        if 'TargetType'.encode() in line:
            targetType = line.strip().split('='.encode())[1].strip()
            targetType = targetType.decode()
        elif 'TargetSerNum'.encode() in line:
            targetSerNum = line.strip().split('='.encode())[1].strip()
            targetSerNum = targetSerNum.decode()
        elif 'NumberOfColumns'.encode() in line:
            cols = int(line.strip().split('='.encode())[1].strip())
        elif 'NumberOfRows'.encode() in line:
            rows = int(line.strip().split('='.encode())[1].strip())
		
    label = targetType# + '_' + targetSerNum

    roffset = (rows-128)//2
    coffset = (cols-128)//2
    data_m = data[:rows*cols]
    data_m = data_m.reshape((rows,cols))
    data_m = data_m[roffset:(128+roffset),coffset:(128+coffset)]
    # For using phase together
#    data_p = data[rows*cols:]
#    data_p = data_p.reshape((rows,cols))
#    data_p = data_p[roffset:(128+roffset),coffset:(128+coffset)]

    return data_m.astype('float32'), label, targetSerNum

def readMSTARFile_phase(filename):
# raw_input('Enter the mstar file to read: ')

	#print filename

    f = open(filename, 'rb')

    a = ''.encode()

    phoenix_header = []

    while 'PhoenixHeaderVer'.encode() not in a:
        a = f.readline()

    a = f.readline()

    while 'EndofPhoenixHeader'.encode() not in a:
        phoenix_header.append(a)
        a = f.readline()

    data = np.fromfile(f, dtype='>f4')

    #	print(data.shape)
    #
    #	magdata = data[:128*128]
    #	phasedata = data[128*128:]
    
    #	if you want to print an image
    #    imdata = magdata*255
    #
    #	imdata = imdata.astype('uint8')

    targetSerNum = '-'

    for line in phoenix_header:
        #print line
        if 'TargetType'.encode() in line:
            targetType = line.strip().split('='.encode())[1].strip()
            targetType = targetType.decode()
        elif 'TargetSerNum'.encode() in line:
            targetSerNum = line.strip().split('='.encode())[1].strip()
            targetSerNum = targetSerNum.decode()
        elif 'NumberOfColumns'.encode() in line:
            cols = int(line.strip().split('='.encode())[1].strip())
        elif 'NumberOfRows'.encode() in line:
            rows = int(line.strip().split('='.encode())[1].strip())
		
    label = targetType# + '_' + targetSerNum

    roffset = (rows-128)//2
    coffset = (cols-128)//2
    data_m = data[:rows*cols]
    data_m = data_m.reshape((rows,cols))
    data_m = data_m[roffset:(128+roffset),coffset:(128+coffset)]
    # For using phase together
    data_p = data[rows*cols:]
    data_p = data_p.reshape((rows,cols))
    data_p = data_p[roffset:(128+roffset),coffset:(128+coffset)]

    return data_m.astype('float32'), data_p.astype('float32'), label, targetSerNum

def readMSTARDir(dirname):
	data = np.zeros([128*128,0],dtype = 'float32')
	labels = []
	serNums = []
	files = os.listdir(dirname)

	for f in files:
		fullpath = os.path.join(dirname,f)
		if os.path.isdir(fullpath):
			if 'SLICY' in f:
				continue
			d,l,sn = readMSTARDir(fullpath)
			data = np.concatenate((data,d),axis=1)
			labels = labels + l
			serNums = serNums + sn
		else:
#			print(fullpath)
			if not fnmatch(f,'*.[0-9][0-9][0-9]'):
				continue
			d,l,sn = readMSTARFile(os.path.join(dirname,f))
#			print(d.shape)
#            if d.shape != (128,128):
#                global err1
#                err1 = d
#                sys.exit()
    
			data = np.concatenate((data,d.reshape(-1,1)),axis=1)
			labels = labels + [l]
			serNums = serNums + [sn]

	return data, labels, serNums

def readMSTARDir_phase(dirname):
    data_a = np.zeros([128*128,0], dtype='float32')
    data_p = np.zeros([128*128,0], dtype='float32')
    labels = []
    serNums = []
    files = os.listdir(dirname)
    
    for f in files:
        fullpath = os.path.join(dirname,f)
        if os.path.isdir(fullpath):
            if 'SLICY' in f:
                continue
            d,p,l,sn = readMSTARDir_phase(fullpath)
            data_a = np.concatenate((data_a,d),axis=1)
            data_p = np.concatenate((data_p,p),axis=1)
            labels = labels + l
            serNums = serNums + sn
        else:
            if not fnmatch(f,'*.[0-9][0-9][0-9]'):
                continue
            d,p,l,sn = readMSTARFile_phase(os.path.join(dirname,f))
            
            data_a = np.concatenate((data_a,d.reshape(-1,1)),axis=1)
            data_p = np.concatenate((data_p,p.reshape(-1,1)),axis=1)
            labels = labels + [l]
            serNums = serNums + [sn]
            
    return data_a, data_p, labels, serNums
            



def main1():
	if len(sys.argv) < 3:
		sys.exit()

	filename = sys.argv[1]
	outputfile = sys.argv[2]

	data, labels, serNums = readMSTARDir(os.path.join(filename,'TRAIN'))

	mstar_dic_train = dict()

	mstar_dic_train['data'] = data
	mstar_dic_train['labels'] = labels
	mstar_dic_train['serial numbers'] = serNums

	data, labels, serNums = readMSTARDir(os.path.join(filename,'TEST'))

	mstar_dic_test = dict()

	mstar_dic_test['data'] = data
	mstar_dic_test['labels'] = labels
	mstar_dic_test['serial numbers'] = serNums

	labels = list(set(labels))

	label_dict = dict()

	for i in range(len(labels)):
		label_dict[labels[i]] = i

	for i in range(len(mstar_dic_train['labels'])):
		mstar_dic_train['labels'][i] = label_dict[mstar_dic_train['labels'][i]]

	for i in range(len(mstar_dic_test['labels'])):
		mstar_dic_test['labels'][i] = label_dict[mstar_dic_test['labels'][i]]


	f = open(os.path.join(outputfile,'data_batch_1'),'wb')
	pickle.dump(mstar_dic_train,f)

	f.close()

	f = open(os.path.join(outputfile,'test_batch'),'wb')
	pickle.dump(mstar_dic_test,f)

	f.close()

	meta_dic = dict()

	meta_dic['num_cases_per_batch'] = len(mstar_dic_train['labels'])
	meta_dic['label_names'] = labels

	f = open(os.path.join(outputfile,'batches.meta'),'wb')
	pickle.dump(meta_dic,f)

	f.close()

def main2():
	if len(sys.argv) < 3:
		sys.exit()

	filename = sys.argv[1]
	outputfile = sys.argv[2]

	data, labels, serNums = readMSTARDir(filename)

	mstar_dic_train = dict()

	mstar_dic_train['data'] = data
	mstar_dic_train['labels'] = labels
	mstar_dic_train['serial numbers'] = serNums

	labels = list(set(labels))

	label_dict = dict()

	for i in range(len(labels)):
		label_dict[labels[i]] = i

	for i in range(len(mstar_dic_train['labels'])):
		mstar_dic_train['labels'][i] = label_dict[mstar_dic_train['labels'][i]]

	f = open(os.path.join(outputfile,'data_batch_1'),'wb')
	pickle.dump(mstar_dic_train,f)

	f.close()

	meta_dic = dict()

	meta_dic['num_cases_per_batch'] = len(mstar_dic_train['labels'])
	meta_dic['label_names'] = labels

	f = open(os.path.join(outputfile,'batches.meta'),'wb')
	pickle.dump(meta_dic,f)

	f.close()
def main3():
	if len(sys.argv) < 3:
		sys.exit()

	filename = sys.argv[1]
	outputfile = sys.argv[2]

	data, labels, serNums = readMSTARDir(os.path.join(filename,'17_DEG'))

	mstar_dic_train = dict()

	mstar_dic_train['data'] = data
	mstar_dic_train['labels'] = labels
	mstar_dic_train['serial numbers'] = serNums

	data, labels, serNums = readMSTARDir(os.path.join(filename,'15_DEG'))

	mstar_dic_test = dict()

	mstar_dic_test['data'] = data
	mstar_dic_test['labels'] = labels
	mstar_dic_test['serial numbers'] = serNums

	labels = list(set(labels))

	label_dict = dict()

	for i in range(len(labels)):
		label_dict[labels[i]] = i

	for i in range(len(mstar_dic_train['labels'])):
		mstar_dic_train['labels'][i] = label_dict[mstar_dic_train['labels'][i]]

	for i in range(len(mstar_dic_test['labels'])):
		mstar_dic_test['labels'][i] = label_dict[mstar_dic_test['labels'][i]]


	f = open(os.path.join(outputfile,'data_batch_1'),'wb')
	pickle.dump(mstar_dic_train,f)

	f.close()

	f = open(os.path.join(outputfile,'test_batch'),'wb')
	pickle.dump(mstar_dic_test,f)

	f.close()

	meta_dic = dict()

	meta_dic['num_cases_per_batch'] = len(mstar_dic_train['labels'])
	meta_dic['label_names'] = labels

	f = open(os.path.join(outputfile,'batches.meta'),'wb')
	pickle.dump(meta_dic,f)

	f.close()

def load_MSTAR_SOC(dat_path, dBscale=True, Target_segment=True, Shadow_segment=True):
    #
    #
    # Load MSTAR_SOC data
    #
    #
    print('Loading MSTAR_SOC_DB...')
    
    dataset_type_path = '10-class (17-15)'
    path_train = '17_DEG'
    path_test = '15_DEG'
    
    list_class = ['2s1_gun','bmp2_tank','brdm2_truck','btr70_transport','btr60_transport',
               'd7_bulldozer','t62_tank','t72_tank','zil131_truck','zsu23-4_gun']
    
    filepath_train = dat_path + '/' + dataset_type_path + '/' + path_train
    filepath_test = dat_path + '/' + dataset_type_path + '/' + path_test
    
    X_train, Y_train, ver_train = readMSTARDir(filepath_train)
    X_test, Y_test, ver_test = readMSTARDir(filepath_test)
    
    # Resize
    X_train_resize = np.zeros([128,128,X_train.shape[1]])
    X_test_resize = np.zeros([128,128,X_test.shape[1]])
    for i in range(X_train.shape[1]):
        X_train_resize[:,:,i] = np.resize(X_train[:,i], (128,128))
    for i in range(X_test.shape[1]):
        X_test_resize[:,:,i] = np.resize(X_test[:,i], (128,128))
    
    if dBscale==True:
        for i in range(X_train.shape[1]):
            X_train_resize[:,:,i] = prepro_DB(X_train_resize[:,:,i])
        for i in range(X_test.shape[1]):
            X_test_resize[:,:,i] = prepro_DB(X_test_resize[:,:,i])
    
    if Target_segment==True:
        X_train_target = np.zeros([128,128,X_train.shape[1]])
        X_test_target = np.zeros([128,128,X_test.shape[1]])
        for i in range(X_train.shape[1]):
            X_train_target[:,:,i], _, _ = prepro_targetseg(X_train_resize[:,:,i])
        for i in range(X_test.shape[1]):
            X_test_target[:,:,i], _, _ = prepro_targetseg(X_test_resize[:,:,i])
            
    if Shadow_segment==True:
        X_train_shadow = np.zeros([128,128,X_train.shape[1]])
        X_test_shadow = np.zeros([128,128,X_test.shape[1]])
        for i in range(X_train.shape[1]):
            X_train_shadow[:,:,i], _, _ = prepro_shadowseg(X_train_resize[:,:,i])
        for i in range(X_test.shape[1]):
            X_test_shadow[:,:,i], _, _ = prepro_shadowseg(X_test_resize[:,:,i])
    
    print('MSTAR_SOC_DB loaded')
        
    if (Target_segment==True) and (Shadow_segment==True):
        return [X_train_target, X_train_shadow], [X_test_target, X_test_shadow], Y_train, Y_test, list_class
    elif (Target_segment==True) and (Shadow_segment==False):
        return X_train_target, X_test_target, Y_train, Y_test, list_class
    elif (Target_segment==False) and (Shadow_segment==True):
        return X_train_shadow, X_test_shadow, Y_train, Y_test, list_class
    else:
        return X_train_resize, X_test_resize, Y_train, Y_test, list_class
    

# clutter load
#file_path = 'D://4. Data/MSTAR/PublicClutter/MSTAR_PUBLIC_CLUTTER_CD2/CLUTTER/15_DEG'
#files = os.listdir(file_path)
#clut_ind = 8
#filename = os.path.join(file_path,files[clut_ind])
##
#f = open(filename, 'rb')
#
#a = ''.encode()
#
#phoenix_header = []
#
#while 'PhoenixHeaderVer'.encode() not in a:
#    a = f.readline()
#
#a = f.readline()
#
#while 'EndofPhoenixHeader'.encode() not in a:
#    phoenix_header.append(a)
#    a = f.readline()
#
#data = np.fromfile(f, dtype='>f4')
#
#data_m = data[-2629616:]
#data_m = data_m.reshape((1784,1474))
#fig = plt.figure()
#plt.imshow(np.log(data_m), cmap=plt.get_cmap('gray'))
        
    

# target load_phase
file_path = 'D://7. Research/4. SAR classification/Data/MSTAR/'
dataset_type_path = '10-class (17-15)'
path_train = '17_DEG'
path_test = '15_DEG'
list_class = ['2s1_gun','bmp2_tank','brdm2_truck','btr70_transport','btr60_transport',
               'd7_bulldozer','t62_tank','t72_tank','zil131_truck','zsu23-4_gun']
filepath_test = file_path + '/' + dataset_type_path + '/' + path_test
X_A, X_P, Y_test, ver_test = readMSTARDir_phase(filepath_test)
aa = np.array(Y_test)
for i in range(len(list_class)):
    temp = list_class[i]
    aa = np.where(aa==temp, i ,aa)
aa = np.array(aa, dtype='int')

#np.savez_compressed('D://7. Research/4. SAR classification/Data/MSTAR/X_15_A'
#                    , X_A)
#np.savez_compressed('D://7. Research/4. SAR classification/Data/MSTAR/X_15_P'
#                    , X_P)
#np.savez_compressed('D://7. Research/4. SAR classification/Data/MSTAR/Y'
#                    , aa)
    
#if __name__ == '__main__':
#	main3()
#if __name__ == '__main__':
#    file_path = 'D://7. Research/4. SAR classification/Data/MSTAR/10-class (17-15)'
#    folder = '17_DEG'
#    
#    a,c,d = readMSTARFile("HB03788.000")
#    a1,c1,d1 = readMSTARDir(file_path+'/'+folder)
# print phoenix_header

