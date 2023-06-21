#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author: Kang Pan, Zhuokun Yang

""" Motor imagery"""

import yaml
import numpy as np
import pandas as pd
import pickle
import scipy.io
from mne.decoding import CSP 
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras import regularizers
from sklearn.model_selection import ShuffleSplit, KFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

f    = open("./test_info.yaml",encoding = 'utf-8')
conf = yaml.load(f,Loader = yaml.SafeLoader)


Subject   = conf["Subject" ]                           # subject id
TestDate  = conf["TestDate"]                           # the date when train data was recorded
TestMode  = conf["TestMode"]                           # 1D task or 2D task

if TestMode == '1D':
    classnum = 2
else:
    classnum = 4

def feature_scalling(X):
    mmin = X.min()
    mmax = X.max()
    return (X - mmin) / (mmax - mmin), mmin, mmax

def feature_norm(X):
    mmean = X.mean()
    sstd  = X.std()
    return (X - mmean) / sstd, mmean, sstd

def build_classifier(num_layers = 1):
    classifier = Sequential()
    # First Layer
    classifier.add(Dense(units = 128, input_dim = classnum*fea_N, kernel_initializer = 'uniform', activation = 'relu', 
                         kernel_regularizer=regularizers.l2(0.01))) # L2 regularization
    classifier.add(Dropout(0.5))
    # Intermediate Layers
    for itr in range(num_layers):
        classifier.add(Dense(units = 128, kernel_initializer = 'uniform', activation = 'relu', 
                             kernel_regularizer=regularizers.l2(0.01))) # L2 regularization
        classifier.add(Dropout(0.5))   

    # Last Layer
    classifier.add(Dense(units = classnum, kernel_initializer = 'uniform', activation = 'softmax'))
    classifier.compile(optimizer = 'rmsprop' , loss = 'categorical_crossentropy', metrics = ['accuracy'])
    return classifier

root_path = "./EEGdata/{}/{}/".format(Subject,TestDate)

labels       = np.load(root_path+"labels.npy")
load_data    = np.load(root_path+"dataset.npy")

feature_data = load_data
feature_data, mmean, sstd = feature_norm(feature_data)
# feature_data[:8,:,:,:], mmin, mmax = feature_scalling(feature_data[:8,:,:,:])

scipy.io.savemat(root_path+"parameter.mat", {'mmean': mmean, 'sstd': sstd})


fea_N  = np.shape(feature_data)[0]


# OneHotEncoding Labels
enc   = OneHotEncoder()
X_out = enc.fit_transform(labels.reshape(-1,1)).toarray()

# Cross Validation Split
# cv = ShuffleSplit(n_splits = 5, test_size = 0.2, random_state = 0)
cv = KFold(n_splits = 5)

acc    = []
ka     = []
prec   = []
recall = []

for train_idx, test_idx in cv.split(labels):
    np.random.shuffle(train_idx)
    Csp = []; ss = []; nn = [] # empty lists
    label_train, label_test = labels[train_idx],  labels[test_idx]
    y_train    , y_test     = X_out [train_idx],  X_out [test_idx]
    
    # CSP filter applied separately for all Frequency band coefficients
    
    Csp     = [CSP(n_components = classnum, reg=None, log=True, norm_trace=False) for _ in range(fea_N)]
    ss      = preprocessing.StandardScaler()
    X_train = ss.fit_transform(np.concatenate(tuple(Csp[x].fit_transform(feature_data[x,train_idx,:,:],label_train) for x  in range(fea_N)),axis=-1))
    X_test  = ss.transform(np.concatenate(tuple(Csp[x].transform(feature_data[x,test_idx,:,:]) for x  in range(fea_N)),axis=-1))
    
    X_train = X_train.reshape((np.shape(X_train)[0],np.shape(X_train)[1],-1))
    X_test  = X_test.reshape((np.shape(X_test)[0],np.shape(X_test)[1],-1))


    nn = build_classifier()  
    
    nn.fit(X_train, y_train, batch_size = 32, epochs = 500)
    
    y_pred = nn.predict(X_test)
    pred   = (y_pred == y_pred.max(axis=1)[:,None]).astype(int)

    acc   .append(accuracy_score   (y_test.argmax(axis=1), pred.argmax(axis=1)))
    ka    .append(cohen_kappa_score(y_test.argmax(axis=1), pred.argmax(axis=1)))
    prec  .append(precision_score  (y_test.argmax(axis=1), pred.argmax(axis=1), average='weighted'))
    recall.append(recall_score     (y_test.argmax(axis=1), pred.argmax(axis=1), average='weighted'))


scores = {'Accuracy':acc,'Kappa':ka,'Precision':prec,'Recall':recall}

Es  = pd.DataFrame(scores)

avg = {'Accuracy':[np.mean(acc)],'Kappa':[np.mean(ka)],'Precision':[np.mean(prec)],'Recall':[np.mean(recall)]}

Avg = pd.DataFrame(avg)


T = pd.concat([Es,Avg])

T.index = ['F1','F2','F3','F4','F5','Avg']#,'F6','F7','F8','F9','F10','Avg']
T.index.rename('Fold',inplace=True)

print(T)

# train the model with all the data
train_idx = np.arange(len(labels))
np.random.shuffle(train_idx)
Csp = []; ss = []; nn = [] # empty lists
label_train = labels[train_idx]
y_train     = X_out [train_idx]

# CSP filter applied separately for all Frequency band coefficients

Csp     = [CSP(n_components=classnum, reg=None, log=True, norm_trace=False) for _ in range(fea_N)]
ss      = preprocessing.StandardScaler()
X_train = ss.fit_transform(np.concatenate(tuple(Csp[x].fit_transform(feature_data[x,train_idx,:,:],label_train) for x  in range(fea_N)),axis=-1))

pickle.dump(Csp,open(root_path+'csp.txt', 'wb'))
pickle.dump(ss,open(root_path+'ss.txt', 'wb'))

X_train = X_train.reshape((np.shape(X_train)[0],np.shape(X_train)[1],-1))

nn = build_classifier()  

nn.fit(X_train, y_train, batch_size = 32, epochs = 500)

nn.save(root_path+'model.h5')