#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author: Kang Pan, Zhuokun Yang

""" Motor imagery"""

import yaml
import mne
import pywt
import scipy.io
import numpy as np
from   scipy.fftpack import fft

f    = open("./test_info.yaml",encoding = 'utf-8')
conf = yaml.load(f,Loader = yaml.SafeLoader)


Subject    = conf["Subject"   ]                            # subject id
TestDate   = conf["TestDate"  ]                            # the date when train data was recorded
TestIDlist = conf["TestIDlist"]                            # train data id
TestMode   = conf["TestMode"  ]                            # 1D task or 2D task

picks      = conf["ChannelPicks"]                          # the channel need to process
high_pass  = conf["high_pass"   ]                          # high frequency in high pass filter
low_pass   = conf["low_pass"    ]                          # low frequency in low pass filter 
WinSize    = conf["WinSize"     ]                          # the window size of a sample

root_path = "./EEGdata/{}/{}/".format(Subject,TestDate)

data  = []
label = []
for TestID in TestIDlist:
    edf_name = "{}/{}_{}_{}.edf".format(TestID,Subject,TestDate,TestID)
    evt_name  = "{}/{}_task_{}.mat".format(TestID,Subject,TestMode)

    raw = mne.io.read_raw_edf(root_path + edf_name)

    raw.load_data()

    # Filter the raw signal with a band pass filter 

    raw.filter(high_pass, low_pass, fir_design='firwin')

    evt_mat  = scipy.io.loadmat(root_path+evt_name) 
    readylen = evt_mat["readylen"][0]
    startlen = evt_mat["startlen"][0]
    endlen   = evt_mat["endlen"  ][0]
    labels   = evt_mat["labels"  ][0]

    raw_data = raw._data

    for trial in range(len(readylen)):
        for step in range(startlen[trial],endlen[trial]-WinSize+1,1000):
            print(step)
            data.append(raw_data[picks,step:step+WinSize:4])
            dirction = 1 if labels[trial]>0 else 0 #0:left 1:right
            label.append(dirction) 
data = np.array(data)
label = np.array(label)



# signal is decomposed to level 5 with 'db4' wavelet

def wpd(X): 
    coeffs = pywt.WaveletPacket(X,'db4',mode='symmetric',maxlevel=5)
    return coeffs
             
def feature_bands(x):
    Bands = np.empty((8,x.shape[0],x.shape[1],30)) # 8 freq band coefficients are chosen from the range 4-32Hz
    for i in range(x.shape[0]):
        for ii in range(x.shape[1]):
             pos = []
             C   = wpd(x[i,ii,:]) 
             pos = np.append(pos,[node.path for node in C.get_level(5, 'natural')])
             for b in range(1,9):
                Bands[b-1,i,ii,:] = C[pos[b]].data
    return Bands

def feature_extract(x):
        feature = np.empty((1,x.shape[0],x.shape[1],30)) # 8 freq band coefficients are chosen from the range 4-32Hz
        for i in range(x.shape[0]):
            print(i)
            for ii in range(x.shape[1]):
                for time in range(30):
                    b_data = x[i,ii,time*24:(time+2)*24] 
                    Y  = fft(b_data, 512)
                    Y  = np.abs(Y)
                    ps = Y**2 / 512
                    feature[0,i,ii,time] = np.sum(ps[44:68]) # 8-12Hz
        return feature

wpd_data  = feature_bands(data)
burd_data = feature_extract(data)

feature_data = np.concatenate((wpd_data, burd_data), axis = 0)

np.save(root_path + "dataset.npy", feature_data)
np.save(root_path + "labels.npy", label)

