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

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

#device = torch.device("cpu")

print("Using", device)


class propellerDataset(Dataset):
    def __init__(self, file_path, transform=None, target_transform=None):
        df = pd.read_pickle(file_path)
        
        #self.spect_df = df.query('tipo == "Recorte asimétrico" or tipo == "Ninguna"')
        #self.spect_df = df.query('tipo == "Recorte simétrico" or tipo == "Ninguna"')
        #self.spect_df = df.query('tipo == "Recorte transversal simétrico" or tipo == "Ninguna"')
        #self.spect_df = df.query('tipo == "Recorte asimétrico" or tipo == "Recorte simétrico" or tipo == "Ninguna"')
        self.spect_df = df.query('tipo == "Recorte asimétrico" or tipo == "Recorte simétrico"')
        
        #self.spect_df = self.spect_df.query('motor == 1')
        
        # Order spect_df by dataset
        self.spect_df = self.spect_df.sort_values(by=["dataset"])
        self.spect_df = self.spect_df.reset_index(drop=True)
        # get length of each dataset
        self.len_df = self.spect_df.groupby("dataset").size().tolist()
        # get index of each dataset adding the window size divided by the overlap
        self.idx_df = np.cumsum(self.len_df).tolist()
    

        self.transform = transform
        self.target_transform = target_transform
        self.window = 256
        self.overlap = 32
        filt_width = 1 #prev 1
        filter_order = 10
        fs = 222
        self.n_filt=round(fs/2/filt_width)
        l_f = np.arange(1,(fs//2)-filt_width,filt_width) 
        h_f = np.arange(filt_width+1,(fs//2),filt_width)

        self.sos = []
        for i in range(len(l_f)):
            self.sos.append(sig.iirfilter(filter_order,[l_f[i], h_f[i]], rs=60, 
                                          btype='band',analog=False, ftype='cheby2', fs=fs, output='sos', rp=None))
            
        #FEATURES
        nfeat=int((fs/2)//filt_width*10 + 3*4)
        features=np.zeros((nfeat,0))
        labels=np.zeros((2,0))

        #self.window-self.overlap
        for idx in tqdm.tqdm(range(0,self.spect_df.shape[0],self.overlap), desc=f"Buiding dataset for {file_path}"):
            data =self.spect_df[
                [
                    "acc_x","acc_y","acc_z",
                    "gyro_x","gyro_y","gyro_z",
                    "My","Mx","Mz","T",
                    "tipo","motor","prof1","prof2",
                ]
            ].iloc[idx : idx + self.window]
            
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
                
            out = np.zeros((self.n_filt,10))
            for i,j in enumerate(self.sos):
                out[i,:] = np.sum((sig.sosfilt(j,data_np,axis=0))**2,axis=0)
            max_out = np.max(out,axis=0)
            filters = out/(max_out + 0.001)
            print(filters.max(axis=0))
            
            # momentos orden 
            mom_pitch = np.concatenate((np.expand_dims(data["My"].mean(), axis=0), stats.moment(data["My"], moment = [2,3,4])))
            mom_roll = np.concatenate((np.expand_dims(data["Mx"].mean(), axis=0), stats.moment(data["Mx"], moment = [2,3,4])))
            mom_yaw = np.concatenate((np.expand_dims(data["Mz"].mean(), axis=0), stats.moment(data["Mz"], moment = [2,3,4])))
            
            # peak = 
            new_feat = np.concatenate((filters.flatten(), mom_pitch, mom_roll, mom_yaw))
            print("fil ",filters.shape)
            print("flat ",filters.flatten())
            
            symm = data["prof1"].mean() + data["prof2"].mean()
            asymm = abs(data["prof1"].mean() - data["prof2"].mean())
            new_label = np.array([asymm,symm])
            
            if symm%5==0 and asymm%5==0:
                features=np.concatenate((features,np.expand_dims(new_feat,axis=0).T),axis=1)
                labels=np.concatenate((labels,np.expand_dims(new_label,axis=0).T),axis=1)
                
        self.features = features.T
        self.labels = labels.T


    def __len__(self):
        # Add all the lengths of the datasets minus the window size divided by the overlap
        # length = 0
        # for i in self.len_df:
        #     length += (i - self.window)         
        # return length // self.overlap
        return self.features.shape[0]



    def __getitem__(self, idx):
        # # take into account the overlap
        # idx = idx * self.overlap
        # # get data using a circular buffer like approach
        # for dataset_endpont in self.idx_df:
        #     if idx < (dataset_endpont - self.window) :
        #         break
        #     idx = idx + self.window
        
        # idx = idx // (self.window-self.overlap)

        return (
            self.features[idx],
            self.labels[idx],
        )


#%% DATASETS
train_dataloader=propellerDataset("df_int.pickle")
train_length=int(0.7* len(train_dataloader))
test_length=int(0.15* len(train_dataloader))
valid_length=int(len(train_dataloader)-train_length-test_length)
train_dataset,test_dataset,valid_dataset=torch.utils.data.random_split(train_dataloader,(train_length,test_length,valid_length))

nw=0
pm=False

train_dataloader=DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=nw, pin_memory=pm)
test_dataloader=DataLoader(test_dataset, batch_size=8, shuffle=True, num_workers=nw, pin_memory=pm)
valid_dataloader=DataLoader(valid_dataset, batch_size=8, shuffle=False, num_workers=nw, pin_memory=pm)
#train_dataloader = DataLoader(propellerDataset("df_int.pickle"), batch_size=8, shuffle=True, num_workers=0, pin_memory=False)
eval_train_dataloader = DataLoader(propellerDataset("df_int.pickle"), batch_size=8, shuffle=False, num_workers=nw, pin_memory=pm)
eval_outdoor_dataloader = DataLoader(propellerDataset("df_ext.pickle"), batch_size=8, shuffle=False, num_workers=nw, pin_memory=pm)

#%%SAVE
torch.save(train_dataloader, 'C:/Users/cpose/Desktop/main/results/symmasymm/nn_symasymm_train_dataloader.pth')
torch.save(test_dataloader, 'C:/Users/cpose/Desktop/main/results/symmasymm/nn_symasymm_test_dataloader.pth')
torch.save(valid_dataloader, 'C:/Users/cpose/Desktop/main/results/symmasymm/nn_symasymm_valid_dataloader.pth')
torch.save(eval_train_dataloader, 'C:/Users/cpose/Desktop/main/results/symmasymm/nn_symasymm_eval_train_dataloader.pth')
torch.save(eval_outdoor_dataloader, 'C:/Users/cpose/Desktop/main/results/symmasymm/nn_symasymm_eval_outdoor_dataloader.pth')

#%%LOAD
train_dataloader=torch.load('C:/Users/cpose/Desktop/main/results/symmasymm/nn_symasymm_train_dataloader.pth')
test_dataloader=torch.load('C:/Users/cpose/Desktop/main/results/symmasymm/nn_symasymm_test_dataloader.pth')
valid_dataloader=torch.load('C:/Users/cpose/Desktop/main/results/symmasymm/nn_symasymm_valid_dataloader.pth')
eval_train_dataloader=torch.load('C:/Users/cpose/Desktop/main/results/symmasymm/nn_symasymm_eval_train_dataloader.pth')
eval_outdoor_dataloader=torch.load('C:/Users/cpose/Desktop/main/results/symmasymm/nn_symasymm_eval_outdoor_dataloader.pth')

#%% MODEL

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
        x = self.flatten(x)
        x = self.layers(x)
        return x


net = Net().to(device)
# summary(net)

# Define Optimizer and Loss Function
optimizer = torch.optim.Adadelta(net.parameters(), lr= 0.1)
loss_fn = torch.nn.MSELoss()


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()

    for batch, (X, y) in enumerate(tqdm.tqdm(dataloader, desc="Training")):
        X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)

        # Compute prediction error
        pred = model(X.float())
        loss = loss_fn(pred, y.float())

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # if batch % 20 == 0:
        #     loss, current = loss.item(), batch * len(X)
        #     print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn,pltopt):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    model = model.to(device)
    test_loss, correct = 0, 0
    ynp, prednp = [], []
    with torch.no_grad():
        for X, y in tqdm.tqdm(dataloader, desc="Testing"):
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
            #t1 = time.time()
            print(X.shape)
            print(X)
            pred = model(X.float())
            #print(time.time()-t1)
            test_loss += loss_fn(pred, y.float()).item()
            correct += abs(pred - y).sum()
            if pltopt=='plot':
                plt.scatter(pred.cpu(), y.cpu())
                plt.axis("scaled")
                plt.show()
                print("y:", y.cpu().numpy(), "pred:", pred.cpu().numpy())
            ynp.append(y.cpu().numpy())
            prednp.append(pred.cpu().numpy())
    test_loss /= num_batches
    correct /= size
    print(
        f"Average error: {correct:>0.1f}%, Avg loss: {test_loss:>8f}"
    )
    return ynp, prednp, test_loss, correct

#%% TRAIN

print("Training...:")
epochs = 100
global_loss = 100
loss_epoch_train = []
loss_epoch_valid = []
for t in range(epochs):
    print(f"\nEpoch {t+1} of {epochs}\n-------------------------------")
    train(train_dataloader, net, loss_fn, optimizer)
    print("Test dataset: ")
    y,pred,loss,c = test(test_dataloader, net, loss_fn,"noplot")
    loss_epoch_train.append(c)
    print("Validation dataset: ")
    yval,predval,lval,c = test(valid_dataloader, net, loss_fn,"noplot")
    loss_epoch_valid.append(c)
    if loss < global_loss:
        global_loss = loss
        #folder = "nets/"
        #filename = "NN_cheby10_IIR_0_1_2_3_4.joblib" #filtro eliptico
        #joblib.dump(net, folder+filename)
    y=np.concatenate(y)
    pred=np.concatenate(pred)    


print("\nPrediction over validation set: \n-------------------------------")
yval,predval,lval,c = test(valid_dataloader, net, loss_fn,"noplot")

print("\nPrediction over full train (ordered): \n-------------------------------")
ytrain,predtrain,lt,c = test(eval_train_dataloader, net, loss_fn,"noplot")
ytrain=np.concatenate(ytrain)
predtrain=np.concatenate(predtrain)

print("\nPrediction over outdoor (ordered): \n-------------------------------")
yout,predout,lt,c = test(eval_outdoor_dataloader, net, loss_fn,"noplot")
yout=np.concatenate(yout)
predout=np.concatenate(predout)

print("\nDone.")

# plot results
folder="img/"

def butter_lowpass_filter(data, cutoff, fs, order):
    normal_cutoff = cutoff / (0.5*fs)
    # Get the filter coefficients 
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

try:
    # train pred
    for i in range(pred.shape[1]):
        plt.figure()
        plt.plot(predtrain[:,i], linewidth=0.5)
        plt.plot(ytrain[:,i], alpha=0.5, linewidth=0.5, marker='.')
        pred_filt1=butter_lowpass_filter(predtrain[:,i].T,2,222,5)
        plt.plot(pred_filt1, alpha=0.8, linewidth=0.5, marker='.')
        plt.legend(["pred","y"])
        plt.savefig(folder+"1_train_"+str(i))


    # outdoor pred
    for i in range(pred.shape[1]):
        plt.figure()
        plt.plot(predout[:,i], linewidth=0.5)
        plt.plot(yout[:,i], alpha=0.5, linewidth=0.5, marker='.')
        pred_filt1=butter_lowpass_filter(predout[:,i].T,2,222,5)
        plt.plot(pred_filt1, alpha=0.8, linewidth=0.5, marker='.')
        plt.legend(["pred","y"])
        plt.savefig(folder+"2_out_"+str(i))


except:
    plt.figure()
    plt.plot(predtrain, linewidth=0.5)
    plt.plot(ytrain, alpha=0.5, linewidth=0.5, marker='.')
    pred_filt1=butter_lowpass_filter(predtrain.T,2,222,5)
    plt.plot(pred_filt1, alpha=0.8, linewidth=0.5, marker='.')
    plt.legend(["pred","y"])
    plt.savefig(folder+"1_train")
    
    plt.figure()
    plt.plot(predout, linewidth=0.5)
    plt.plot(yout, alpha=0.5, linewidth=0.5, marker='.')
    pred_filt1=butter_lowpass_filter(predout.T,2,222,5)
    plt.plot(pred_filt1, alpha=0.8, linewidth=0.5, marker='.')
    plt.legend(["pred","y"])
    plt.savefig(folder+"2_out")


try:
    plt.figure()
    plt.plot(ytrain[:,0],ytrain[:,1], marker='.', linestyle="None")
    plt.plot(predtrain[:,0],predtrain[:,1], alpha=0.5, linewidth=0.5, marker='.', linestyle="None")
    plt.legend(["asymm","symm"])
    plt.savefig(folder+"3_features_fulltrain")

    plt.figure()
    plt.plot(yval[:,0],yval[:,1], marker='.', linestyle="None")
    plt.plot(predval[:,0],predval[:,1], alpha=0.5, linewidth=0.5, marker='.', linestyle="None")
    plt.legend(["motor","symm"])
    plt.savefig(folder+"3_features_valid")
except:
    None

if device=="cuda":
    loss_epoch_train_2 = [t.cpu().numpy() for t in loss_epoch_train]
    loss_epoch_valid_2 = [t.cpu().numpy() for t in loss_epoch_valid]

plt.figure()
plt.plot(loss_epoch_train, linewidth=0.5)
plt.plot(loss_epoch_valid, linewidth=0.5)
plt.legend(["train","validation"])
plt.savefig(folder+"0_epoch_error")


print("\nSaved figures.")

torch.save(net, "C:/Users/cpose/Desktop/main/results/symmasymm/nn.pickle")
print("\nSaved network and data")

#%% SHAP
import shap
net=torch.load("C:/Users/cpose/Desktop/main/results/symmasymm/nn.pickle")

# Extract a batch from your DataLoader for analysis.
# This is just a sample batch and you can choose any batch for analysis.
data_for_analysis, _ = next(iter(train_dataloader))
for i in range(50):#range(train_dataloader.__len__()//8-1):
    data_iter, _ = next(iter(train_dataloader))
    data_for_analysis = torch.cat((data_for_analysis,data_iter))


data_for_analysis = data_for_analysis.to(device)

# Use DeepExplainer to explain the model
explainer = shap.DeepExplainer(net, data_for_analysis.float())

# Choose a specific instance for which you want to understand the output
test_instance, _ = next(iter(test_dataloader))
test_instance = test_instance.to(device)

# Get SHAP values for the test instance
# shap_values = explainer.shap_values(test_instance.float())
shap_values = explainer.shap_values(data_for_analysis.float())

# Sum the SHAP values across all output classes for visualization
shap_values_sum = np.sum(np.array(shap_values), axis=0)

# Features
features_names=[]
for num_filt in range(0,22):
    for sensor in ["ax","ay","az","gx","gy","gz","My","Mx","Mz","T"]:
        features_names.append((sensor+" power in "+str(num_filt*5)+"-"+str(num_filt*5+5)+"Hz"))
    
features_names.extend(['Pitch mean', 'Pitch Variance', 'Pitch Skewness', 'Pitch Kurtosis', 'Roll mean', 'Roll Variance', 'Roll Skewness', 'Roll Kurtosis', 'Yaw mean', 'Yaw Variance', 'Yaw Skewness', 'Yaw Kurtosis',])
fname = np.array(features_names)

# Plot the SHAP values
# shap.summary_plot(shap_values, plot_type="bar", feature_names=fname)
shap.summary_plot(shap_values, plot_type="bar", feature_names=fname)

#%%
shap_mean = np.abs(shap_values).mean(1)
shap_mean = np.sum(shap_mean, axis=0)
sorted_idx = shap_mean.argsort()

perm_imp_values = shap_mean[sorted_idx]
perm_imp_names = fname[sorted_idx]

plt.barh(fname[sorted_idx[-10:]], shap_mean[sorted_idx[-10:]])

df = pd.DataFrame(perm_imp_values)
df.to_csv('C:/Users/cpose/Desktop/main/results/symmasymm/perm_imp_values_nn.csv', index=False)
df = pd.DataFrame(perm_imp_names)
df.to_csv('C:/Users/cpose/Desktop/main/results/symmasymm/perm_imp_names_nn.csv', index=False)