# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 11:13:07 2023

@author: cpose
"""

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

window = 256
overlap = 32
filt_width = 1
filter_order = 10
fs = 222
n_filt=round(fs/2/filt_width)
l_f = np.arange(1,(fs//2)-filt_width,filt_width) 
h_f = np.arange(filt_width+1,(fs//2),filt_width)
sos = []
for i in range(len(l_f)):
    sos.append(sig.iirfilter(filter_order,[l_f[i], h_f[i]], rs=60, 
                                  btype='band',analog=False, ftype='cheby2', fs=fs, output='sos', rp=None))

def dataset_to_features(df):
    
    nfeat=int((fs/2)//filt_width*4 + 2*4 + 1)
    features=np.zeros((nfeat,0))
    
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
                            # "acc_z",
                            # "gyro_x",
                            # "gyro_y",
                            # "gyro_z",
                            "My",
                            "Mx",
                            # "Mz",
                            # "T",
                        ]
                    ].to_numpy()
                
            out = np.zeros((n_filt,np.shape(data_np)[1]))
            for i,j in enumerate(sos):
                out[i,:] = np.sum((sig.sosfilt(j,data_np,axis=0))**2,axis=0)
            max_out = np.max(out,axis=0)
            filters = out/(max_out + 0.001)
            peak = np.argmax(filters.max(axis=1))
            
            # momentos orden 
            mom_pitch = np.concatenate((np.expand_dims(data["My"].mean(), axis=0), stats.moment(data["My"], moment = [2,3,4])))
            mom_roll = np.concatenate((np.expand_dims(data["Mx"].mean(), axis=0), stats.moment(data["Mx"], moment = [2,3,4])))
            # mom_yaw = np.concatenate((np.expand_dims(data["Mz"].mean(), axis=0), stats.moment(data["Mz"], moment = [2,3,4])))
            
            new_feat = np.concatenate((filters.flatten(), mom_pitch, mom_roll, np.expand_dims(peak, axis=0)))#, mom_yaw))
            features = np.concatenate((features,np.expand_dims(new_feat,axis=0).T),axis=1)
            
    return features
       
def butter_lowpass_filter(data, cutoff, fs, order):
    normal_cutoff = cutoff / (0.5*fs)
    # Get the filter coefficients 
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

scale = 2 #prev 100
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.flatten = nn.Flatten()
        self.layers = nn.Sequential(
            # nn.Linear(1122, 16*scale), #prev
            nn.Linear(232, 16*scale),
            nn.ReLU(),
            nn.Linear(16*scale, 4*scale),
            nn.ReLU(),
            nn.Linear(4*scale, 2*scale),
	    nn.ReLU(),
	    nn.Linear(2*scale, 2),
            # nn.Dropout(p=0.1),
        )

    def forward(self, x):
        #x = self.flatten(x)
        x = self.layers(x)
        return x


net = Net().to("cpu")

fail_list = pd.read_excel("Lista roturas.xlsx")
                                                                                                 
#%% Symm / Asymm inference
net = torch.load("C:/Users/cpose/Desktop/main/results/symmasymm/nn_mod.pickle")

data= pd.read_pickle("df_int.pickle")

plt.figure(dpi=1200, figsize=(10,6))

pred_full=np.empty([0,2])

for index, row in fail_list.iterrows():
    
    p1 = row['Profundidad 1 [mm]']
    p2 = row['Profundidad 2 [mm]']
    tiporot = str(row['Tipo de rotura'])
    
    if tiporot == "Recorte simétrico" or tiporot == "Recorte asimétrico":

        data_flight = data.query(f'prof1 == {p1} and prof2 == {p2} and tipo == "{tiporot}" and motor == 1')
        
        features = dataset_to_features(data_flight)
        with torch.no_grad():
            pred = net(torch.from_numpy(features.T).float())
            pred = pred.numpy()
        
        pred_full = np.concatenate((pred_full,pred))
        
        ffilt=0.01
        # plt.figure()
        # plt.suptitle(f"{tiporot} {p1}-{p2}")
        # plt.subplot(2,1,1)
        # plt.plot(butter_lowpass_filter(pred[:,0],ffilt,1,5))
        # plt.title("Asymm")
        # plt.ylim([-5, max(abs(p1-p2)+5,max(pred[:,0]))])
        # plt.subplot(2,1,2)
        # plt.plot(butter_lowpass_filter(pred[:,1],ffilt,1,5))
        # plt.title("Symm")
        # plt.ylim([-5, max(p1+p2+5,max(pred[:,1]))])
        
        plt.plot(butter_lowpass_filter(pred[:,0],ffilt,1,5),butter_lowpass_filter(pred[:,1],ffilt,1,5), alpha=0.5, linewidth=0.5, marker='.', linestyle="None")
        plt.plot(np.mean(pred[:,0]),np.mean(pred[:,1]), color=plt.gca().lines[-1].get_color(), marker='^', markersize=15, linestyle="None")
        plt.plot(abs(p1-p2),p1+p2, marker='x', color=plt.gca().lines[-1].get_color(), markersize=15, linestyle="None")

plt.xlabel("Difference")
plt.ylabel("Sum")
plt.title("Magnitude of failure")
plt.legend(["Prediction","Prediction mean","Real value"])
plt.show()

df = pd.DataFrame(pred_full)
df.to_csv('C:/Users/cpose/Desktop/main/results/symmasymm/pred_full_nn.csv', index=False)
        
#%% Transv inference
net = torch.load("C:/Users/cpose/Desktop/main/results/transv/nn.pickle")

data= pd.read_pickle("df_int.pickle")

plt.figure()
pred_full=np.empty([0,1])

for index, row in fail_list.iterrows():
    
    p1 = row['Profundidad 1 [mm]']
    p2 = row['Profundidad 2 [mm]']
    tiporot = str(row['Tipo de rotura'])
    
    if tiporot == "Recorte transversal simétrico":

        data_flight = data.query(f'prof1 == {p1} and prof2 == {p2} and tipo == "{tiporot}" and motor == 1')
        
        features = dataset_to_features(data_flight)
        with torch.no_grad():
            pred = net(torch.from_numpy(features.T).float())
            pred = pred.numpy()
        
        pred_full = np.concatenate((pred_full,pred))
        
        ffilt=0.01
        plt.figure()
        plt.suptitle(f"{tiporot} {p1}-{p2}")
        plt.plot(butter_lowpass_filter(pred.T,ffilt,1,5).T)

        # pred=pred.T
        # plt.plot(butter_lowpass_filter(pred,ffilt,1,5),butter_lowpass_filter(pred,ffilt,1,5), alpha=0.5, linewidth=0.5, marker='.', linestyle="None")
        # plt.plot(np.mean(pred),np.mean(pred), color=plt.gca().lines[-1].get_color(), marker='^', markersize=10, linestyle="None")
        # plt.plot(p1+p2,p1+p2, marker='x', color=plt.gca().lines[-1].get_color(), markersize=10, linestyle="None")
        # plt.xlabel("Difference")
        # plt.ylabel("Sum")
        
df = pd.DataFrame(pred_full)
df.to_csv('C:/Users/cpose/Desktop/main/results/transv/pred_full.csv', index=False)