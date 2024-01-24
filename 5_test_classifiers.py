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

def dataset_to_features(df):
    
    features=np.zeros((232,0))
    
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
            features = np.concatenate((features,np.expand_dims(new_feat,axis=0).T),axis=1)
            
    return features
       
def butter_lowpass_filter(data, cutoff, fs, order):
    normal_cutoff = cutoff / (0.5*fs)
    # Get the filter coefficients 
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

fail_list = pd.read_excel("Lista roturas.xlsx")
                                                                                                 
#%% Failure type
clf = joblib.load('C:/Users/cpose/Desktop/main/results/Classifier')

data= pd.read_pickle("df_int.pickle")
t0_full=np.array([])
t1_full=np.array([])
t2_full=np.array([])
plt.figure()
for index, row in fail_list.iterrows():
    
    p1 = row['Profundidad 1 [mm]']
    p2 = row['Profundidad 2 [mm]']
    tiporot = str(row['Tipo de rotura'])

    data_flight = data.query(f'prof1 == {p1} and prof2 == {p2} and tipo == "{tiporot}" and motor == 1')
    tiporot = str(row['Tipo de rotura'])
    
    features = dataset_to_features(data_flight)
    
    predict = clf.predict_proba(features.T)
    t0 = predict[:,0]
    t1 = predict[:,1]
    t2 = predict[:,2]
    
    t0_full = np.concatenate((t0_full,t0))
    t1_full = np.concatenate((t1_full,t1))
    t2_full = np.concatenate((t2_full,t2))
    
    ffilt=0.01
    plt.figure()
    plt.subplot(3,1,1)
    plt.title(f"{tiporot} {p1}-{p2}")
    plt.plot(butter_lowpass_filter(t0,ffilt,1,5))
    plt.ylim([0,1.2])
    plt.subplot(3,1,2)
    plt.plot(butter_lowpass_filter(t1,ffilt,1,5))
    plt.ylim([0,1.2])
    plt.subplot(3,1,3)
    plt.plot(butter_lowpass_filter(t2,ffilt,1,5))
    plt.ylim([0,1.2])
    
    # t0=butter_lowpass_filter(t0,ffilt,1,5)
    # t1=butter_lowpass_filter(t1,ffilt,1,5)
    # t2=butter_lowpass_filter(t2,ffilt,1,5)
    # if tiporot == "Ninguna":
    #     plt.plot(t0-(t1+t2)*0.5,(t1-t2)*0.866,'c.')
    # elif tiporot == "Recorte simétrico" or tiporot == "Recorte asimétrico":
    #     plt.plot(t0-(t1+t2)*0.5,(t1-t2)*0.866,'r.')
    # else:
    #     plt.plot(t0-(t1+t2)*0.5,(t1-t2)*0.866,'g.')
        
    plt.xlim([-0.75, 1.25])
    plt.ylim([-1, 1])
    
df = pd.DataFrame(t0_full)
df.to_csv('C:/Users/cpose/Desktop/main/results/t0_full.csv', index=False)
df = pd.DataFrame(t1_full)
df.to_csv('C:/Users/cpose/Desktop/main/results/t1_full.csv', index=False)
df = pd.DataFrame(t2_full)
df.to_csv('C:/Users/cpose/Desktop/main/results/t2_full.csv', index=False)
    
#%% Symmasymm motor
clf = joblib.load('C:/Users/cpose/Desktop/main/results/symmasymm/Classifier')

data= pd.read_pickle("df_int.pickle")

t1_full=np.array([])
t2_full=np.array([])
t3_full=np.array([])
t4_full=np.array([])

for index, row in fail_list.iterrows():
    
    p1 = row['Profundidad 1 [mm]']
    p2 = row['Profundidad 2 [mm]']
    tiporot = str(row['Tipo de rotura'])

    if tiporot == "Recorte simétrico" or tiporot == "Recorte asimétrico":
        data_flight_prop = data.query(f'prof1 == {p1} and prof2 == {p2} and tipo == "{tiporot}"')
        
        plt.figure()
        ffilt=0.01
        
        for mot in [1,2,3,4]:
            data_flight = data_flight_prop.query(f'motor == {mot}')
        
            features = dataset_to_features(data_flight)
            
            predict = clf.predict_proba(features.T)
            t1 = predict[:,0]
            t2 = predict[:,1]
            t3 = predict[:,2]
            t4 = predict[:,3]
            
            t1_full = np.concatenate((t1_full,t1))
            t2_full = np.concatenate((t2_full,t2))
            t3_full = np.concatenate((t3_full,t3))
            t4_full = np.concatenate((t4_full,t4))
            
            # plt.subplot(2,2,mot)
            # plt.plot(butter_lowpass_filter(predict.T,ffilt,1,5).T),plt.legend(["1","2","3","4"],loc=1)
            # plt.ylim([0,1.2])
        # plt.subplot(2,2,1)
            
            t1=butter_lowpass_filter(t1,ffilt,1,5)
            t2=butter_lowpass_filter(t2,ffilt,1,5)
            t3=butter_lowpass_filter(t3,ffilt,1,5)
            t4=butter_lowpass_filter(t4,ffilt,1,5)
            if mot == 1:
                plt.plot((t1+t4-t2-t3)*0.707,(t1+t2-t3-t4)*0.707,'c.')
            elif mot == 2:
                plt.plot((t1+t4-t2-t3)*0.707,(t1+t2-t3-t4)*0.707,'r.')
            elif mot == 3:
                plt.plot((t1+t4-t2-t3)*0.707,(t1+t2-t3-t4)*0.707,'g.')
            else:
                plt.plot((t1+t4-t2-t3)*0.707,(t1+t2-t3-t4)*0.707,'m.')
                
        plt.title(f"{tiporot} {p1}-{p2}")
        plt.xlim([-0.75, 0.75])
        plt.ylim([-0.75, 0.75])

df = pd.DataFrame(t1_full)
df.to_csv('C:/Users/cpose/Desktop/main/results/symmasymm/t1_full.csv', index=False)
df = pd.DataFrame(t2_full)
df.to_csv('C:/Users/cpose/Desktop/main/results/symmasymm/t2_full.csv', index=False)
df = pd.DataFrame(t3_full)
df.to_csv('C:/Users/cpose/Desktop/main/results/symmasymm/t3_full.csv', index=False)
df = pd.DataFrame(t4_full)
df.to_csv('C:/Users/cpose/Desktop/main/results/symmasymm/t4_full.csv', index=False)
        
#%% Symmasymm motor
clf = joblib.load('C:/Users/cpose/Desktop/main/results/symmasymm/Classifier')

data= pd.read_pickle("df_int.pickle")

t1_full=np.array([])
t2_full=np.array([])
t3_full=np.array([])
t4_full=np.array([])

for index, row in fail_list.iterrows():
    
    p1 = row['Profundidad 1 [mm]']
    p2 = row['Profundidad 2 [mm]']
    tiporot = str(row['Tipo de rotura'])

    if tiporot == "Recorte transversal simétrico":
        data_flight_prop = data.query(f'prof1 == {p1} and prof2 == {p2} and tipo == "{tiporot}"')
        
        plt.figure()
        ffilt=0.01
        
        for mot in [1,2,3,4]:
            data_flight = data_flight_prop.query(f'motor == {mot}')
        
            features = dataset_to_features(data_flight)
            
            predict = clf.predict_proba(features.T)
            t1 = predict[:,0]
            t2 = predict[:,1]
            t3 = predict[:,2]
            t4 = predict[:,3]
            
            t1_full = np.concatenate((t1_full,t1))
            t2_full = np.concatenate((t2_full,t2))
            t3_full = np.concatenate((t3_full,t3))
            t4_full = np.concatenate((t4_full,t4))
            
            # plt.subplot(2,2,mot)
            # plt.plot(butter_lowpass_filter(predict.T,ffilt,1,5).T),plt.legend(["1","2","3","4"],loc=1)
            # plt.ylim([0,1.2])
        # plt.subplot(2,2,1)
            
            t1=butter_lowpass_filter(t1,ffilt,1,5)
            t2=butter_lowpass_filter(t2,ffilt,1,5)
            t3=butter_lowpass_filter(t3,ffilt,1,5)
            t4=butter_lowpass_filter(t4,ffilt,1,5)
            if mot == 1:
                plt.plot((t1+t4-t2-t3)*0.707,(t1+t2-t3-t4)*0.707,'c.')
            elif mot == 2:
                plt.plot((t1+t4-t2-t3)*0.707,(t1+t2-t3-t4)*0.707,'r.')
            elif mot == 3:
                plt.plot((t1+t4-t2-t3)*0.707,(t1+t2-t3-t4)*0.707,'g.')
            else:
                plt.plot((t1+t4-t2-t3)*0.707,(t1+t2-t3-t4)*0.707,'m.')
                
        plt.title(f"{tiporot} {p1}-{p2}")   
        plt.xlim([-0.75, 0.75])
        plt.ylim([-0.75, 0.75])
        
df = pd.DataFrame(t1_full)
df.to_csv('C:/Users/cpose/Desktop/main/results/transv/t1_full.csv', index=False)
df = pd.DataFrame(t2_full)
df.to_csv('C:/Users/cpose/Desktop/main/results/transv/t2_full.csv', index=False)
df = pd.DataFrame(t3_full)
df.to_csv('C:/Users/cpose/Desktop/main/results/transv/t3_full.csv', index=False)
df = pd.DataFrame(t4_full)
df.to_csv('C:/Users/cpose/Desktop/main/results/transv/t4_full.csv', index=False)