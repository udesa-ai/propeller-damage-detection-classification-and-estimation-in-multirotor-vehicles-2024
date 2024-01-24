#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 27 09:58:11 2022

@author: guillermomarzik
"""

import time

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.signal as sig
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm as tqdm
from scipy import stats
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from torchinfo import summary
from scipy.signal import butter,filtfilt
import sklearn.pipeline
import sklearn.preprocessing
import sklearn.svm
from sklearn.inspection import permutation_importance
import sklearn.cluster

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

print("Using", device)


df = pd.read_pickle("df_int.pickle")

#==============================================================================
#SELECT DATASET
#df = df.query('tipo == "Recorte asimétrico" or tipo == "Ninguna"')
#df = df.query('tipo == "Recorte simétrico" or tipo == "Ninguna"')
#df = df.query('tipo == "Recorte transversal simétrico"')
#df = df.query('tipo == "Recorte asimétrico" or tipo == "Recorte simétrico" or tipo == "Ninguna"')
df = df.query('tipo == "Recorte asimétrico" or tipo == "Recorte simétrico"')

#==============================================================================
#FILTERS
window = 256
overlap = 32
filt_width = 5
filter_order = 10
fs = 222
n_filt=round(fs/2/filt_width)
l_f = np.arange(1,(fs//2)-filt_width,filt_width) 
h_f = np.arange(filt_width+1,(fs//2),filt_width)

sos = []
for i in range(len(l_f)):
    sos.append(sig.iirfilter(filter_order,[l_f[i], h_f[i]], rs=60, 
                                  btype='band',analog=False, ftype='cheby2', fs=fs, output='sos', rp=None))
    

#FEATURES
class1=np.zeros((232,0))
class2=np.zeros((232,0))
class3=np.zeros((232,0))
class4=np.zeros((232,0))
# features=np.zeros((12,0))   #only moments

for idx in tqdm.tqdm(range(0,df.shape[0],overlap), desc="Building dataset"):
    motors = df[["motor"]].iloc[idx : idx + window].to_numpy()
    if np.all(motors == motors[0]):
        data =df[
            [
                "acc_x","acc_y","acc_z",
                "gyro_x","gyro_y","gyro_z",
                "My","Mx","Mz","T",
                "tipo","motor","prof1","prof2",
            ]
        ].iloc[idx : idx + window]
        
        data_np = data[
                    [
                        "acc_x",
                        "acc_y",
                        "acc_z",
                        "gyro_x",
                        "gyro_y",
                        "gyro_z",
                        "My",
                        "Mx",
                        "Mz",
                        "T",
                    ]
                ].to_numpy()
            
        out = np.zeros((n_filt,10))
        for i,j in enumerate(sos):
            out[i,:] = np.sum((sig.sosfilt(j,data_np,axis=0))**2,axis=0)
        max_out = np.max(out,axis=0)
        filters = out/(max_out + 0.001)
        
        # momentos orden 
        mom_pitch = np.concatenate((np.expand_dims(data["My"].mean(), axis=0), stats.moment(data["My"], moment = [2,3,4])))
        mom_roll = np.concatenate((np.expand_dims(data["Mx"].mean(), axis=0), stats.moment(data["Mx"], moment = [2,3,4])))
        mom_yaw = np.concatenate((np.expand_dims(data["Mz"].mean(), axis=0), stats.moment(data["Mz"], moment = [2,3,4])))
        
        new_feat = np.concatenate((filters.flatten(), mom_pitch, mom_roll, mom_yaw))
        # new_feat = np.concatenate((mom_pitch, mom_roll, mom_yaw))   #only moments
        
        if motors[0]==1:
            class1=np.concatenate((class1,np.expand_dims(new_feat,axis=0).T),axis=1)
        elif motors[0]==2:
            class2=np.concatenate((class2,np.expand_dims(new_feat,axis=0).T),axis=1)
        elif motors[0]==3:
            class3=np.concatenate((class3,np.expand_dims(new_feat,axis=0).T),axis=1)
        elif motors[0]==4:
            class4=np.concatenate((class4,np.expand_dims(new_feat,axis=0).T),axis=1)

#%% SAVE
df = pd.DataFrame(class1)
df.to_csv('C:/Users/cpose/Desktop/main/results/symmasymm/class1.csv', index=False)
df = pd.DataFrame(class2)
df.to_csv('C:/Users/cpose/Desktop/main/results/symmasymm/class2.csv', index=False)
df = pd.DataFrame(class3)
df.to_csv('C:/Users/cpose/Desktop/main/results/symmasymm/class3.csv', index=False)
df = pd.DataFrame(class4)
df.to_csv('C:/Users/cpose/Desktop/main/results/symmasymm/class4.csv', index=False)

#%% LOAD
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.signal as sig
import torch
import tqdm as tqdm
from scipy import stats
from scipy.signal import butter,filtfilt
import sklearn.pipeline
import sklearn.preprocessing
import sklearn.svm
from sklearn.inspection import permutation_importance
import sklearn.cluster
df = pd.read_csv('C:/Users/cpose/Desktop/main/results/symmasymm/class1.csv', header=None, skiprows=1)
class1 = df.to_numpy()
df = pd.read_csv('C:/Users/cpose/Desktop/main/results/symmasymm/class2.csv', header=None, skiprows=1)
class2 = df.to_numpy()
df = pd.read_csv('C:/Users/cpose/Desktop/main/results/symmasymm/class3.csv', header=None, skiprows=1)
class3 = df.to_numpy()
df = pd.read_csv('C:/Users/cpose/Desktop/main/results/symmasymm/class4.csv', header=None, skiprows=1)
class4 = df.to_numpy()


#%% CLUSTER
n_centr=4000

cent1 = sklearn.cluster.KMeans(n_centr)
cent2 = sklearn.cluster.KMeans(n_centr)
cent3 = sklearn.cluster.KMeans(n_centr)
cent4 = sklearn.cluster.KMeans(n_centr)

print("Calculating M1 "+str(n_centr)+"-point cluster")
cent1.fit(class1.T)
print("Calculating M2 "+str(n_centr)+"-point cluster")
cent2.fit(class2.T)
print("Calculating M3 "+str(n_centr)+"-point cluster")
cent3.fit(class3.T)
print("Calculating M4 "+str(n_centr)+"-point cluster")
cent4.fit(class4.T)

cent1=cent1.cluster_centers_
cent2=cent2.cluster_centers_
cent3=cent3.cluster_centers_
cent4=cent4.cluster_centers_

features = np.concatenate((cent1,cent2,cent3,cent4))
labels = np.concatenate((1*np.ones(n_centr), 2*np.ones(n_centr), 3*np.ones(n_centr), 4*np.ones(n_centr)))

df = pd.DataFrame(cent1)
df.to_csv('C:/Users/cpose/Desktop/main/results/symmasymm/cent1.csv', index=False)
df = pd.DataFrame(cent2)
df.to_csv('C:/Users/cpose/Desktop/main/results/symmasymm/cent2.csv', index=False)
df = pd.DataFrame(cent3)
df.to_csv('C:/Users/cpose/Desktop/main/results/symmasymm/cent3.csv', index=False)
df = pd.DataFrame(cent4)
df.to_csv('C:/Users/cpose/Desktop/main/results/symmasymm/cent4.csv', index=False)

df = pd.DataFrame(features)
df.to_csv('C:/Users/cpose/Desktop/main/results/symmasymm/features.csv', index=False)
df = pd.DataFrame(labels)
df.to_csv('C:/Users/cpose/Desktop/main/results/symmasymm/labels.csv', index=False)

#%%
ncentr = 4000

df = pd.read_csv('C:/Users/cpose/Desktop/main/results/symmasymm/cent1.csv', header=None, skiprows=1)
cent1 = df.to_numpy()
df = pd.read_csv('C:/Users/cpose/Desktop/main/results/symmasymm/cent2.csv', header=None, skiprows=1)
cent2 = df.to_numpy()
df = pd.read_csv('C:/Users/cpose/Desktop/main/results/symmasymm/cent3.csv', header=None, skiprows=1)
cent3 = df.to_numpy()
df = pd.read_csv('C:/Users/cpose/Desktop/main/results/symmasymm/cent4.csv', header=None, skiprows=1)
cent4 = df.to_numpy()

df = pd.read_csv('C:/Users/cpose/Desktop/main/results/symmasymm/features.csv', header=None, skiprows=1)
features = df.to_numpy()
df = pd.read_csv('C:/Users/cpose/Desktop/main/results/symmasymm/labels.csv', header=None, skiprows=1)
labels = df.to_numpy()

clf = joblib.load('C:/Users/cpose/Desktop/main/results/symmasymm/Classifier')

#%% TRAIN
print("\nStart training...")


#Hyperparameters
c =  1  #  1e-3
error_tol = 1e-5 #default


# Building classifier:
clf = sklearn.svm.SVC(C = c, probability = True, kernel = 'linear', tol = error_tol)

# Normalization of features: 
clf = sklearn.pipeline.make_pipeline(sklearn.preprocessing.StandardScaler(),clf)

# Transforming numbers in labels:
# lab_enc = sklearn.preprocessing.LabelEncoder()
# training_outputs = lab_enc.fit_transform(training_data.values[:,-1])

# Training:
clf.fit(features,labels.flatten())

# Saving:
joblib.dump(clf, 'C:/Users/cpose/Desktop/main/results/symmasymm/Classifier')


print("\nDone.")


#%% TESTING

def butter_lowpass_filter(data, cutoff, fs, order):
    normal_cutoff = cutoff / (0.5*fs)
    # Get the filter coefficients 
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

fullfeatures = np.concatenate((class1.T,class2.T,class3.T,class4.T))
fulllabels = np.concatenate((1*np.ones(class1.shape[1]), 2*np.ones(class2.shape[1]), 3*np.ones(class3.shape[1]), 4*np.ones(class4.shape[1])))
predict = clf.predict_proba(fullfeatures)

m1 = predict[:,0]
m2 = predict[:,1]
m3 = predict[:,2]
m4 = predict[:,3]
#%%

ffilt=0.01
plt.figure()
plt.subplot(2,2,1)
plt.plot(butter_lowpass_filter(m1,ffilt,1,5)),plt.plot(fulllabels.flatten())
plt.subplot(2,2,2)
plt.plot(butter_lowpass_filter(m2,ffilt,1,5)),plt.plot(fulllabels.flatten())
plt.subplot(2,2,3)
plt.plot(butter_lowpass_filter(m3,ffilt,1,5)),plt.plot(fulllabels.flatten())
plt.subplot(2,2,4)
plt.plot(butter_lowpass_filter(m4,ffilt,1,5)),plt.plot(fulllabels.flatten())

thresh = 0.5
plt.figure()
plt.subplot(2,2,1)
plt.plot(butter_lowpass_filter(m1,ffilt,1,5)>thresh),plt.plot(fulllabels.flatten())
plt.subplot(2,2,2)
plt.plot(butter_lowpass_filter(m2,ffilt,1,5)>thresh),plt.plot(fulllabels.flatten())
plt.subplot(2,2,3)
plt.plot(butter_lowpass_filter(m3,ffilt,1,5)>thresh),plt.plot(fulllabels.flatten())
plt.subplot(2,2,4)
plt.plot(butter_lowpass_filter(m4,ffilt,1,5)>thresh),plt.plot(fulllabels.flatten())

df = pd.DataFrame(predict)
df.to_csv('C:/Users/cpose/Desktop/main/results/symmasymm/predict.csv', index=False)
df = pd.DataFrame(fulllabels)
df.to_csv('C:/Users/cpose/Desktop/main/results/symmasymm/fulllabels.csv', index=False)

#%% FEATURE IMPORTANCE
# Features
features_names=[]
for num_filt in range(0,22):
    for sensor in ["ax","ay","az","gx","gy","gz","My","Mx","Mz","T"]:
        features_names.append((sensor+" power in "+str(num_filt*5)+"-"+str(num_filt*5+5)+"Hz"))
    
features_names.extend(['Pitch mean', 'Pitch Variance', 'Pitch Skewness', 'Roll Kurtosis', 'Roll mean', 'Roll Variance', 'Roll Skewness', 'Roll Kurtosis', 'Yaw mean', 'Yaw Variance', 'Yaw Skewness', 'Yaw Kurtosis',])
perm_importance = permutation_importance(clf, fullfeatures,fulllabels.flatten())

fname = np.array(features_names)

sorted_idx = perm_importance.importances_mean.argsort()
plt.figure()
plt.barh(fname[sorted_idx[-10:]], perm_importance.importances_mean[sorted_idx[-10:]])
plt.xlabel("Permutation Importance")

perm_imp_values = perm_importance.importances_mean[sorted_idx]
perm_imp_names = fname[sorted_idx]
df = pd.DataFrame(perm_imp_values)
df.to_csv('C:/Users/cpose/Desktop/main/results/symmasymm/perm_imp_values.csv', index=False)
df = pd.DataFrame(perm_imp_names)
df.to_csv('C:/Users/cpose/Desktop/main/results/symmasymm/perm_imp_names.csv', index=False)