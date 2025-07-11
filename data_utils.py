import numpy as np
from torch import Tensor
import librosa
from torch.utils.data import Dataset
from RawBoost import  process_Rawboost_feature	
from utils import pad
			

class Dataset_train(Dataset):
    def __init__(self, args, list_IDs, labels, base_dir, algo):
        self.list_IDs = list_IDs
        self.labels = labels
        self.base_dir = base_dir
        self.algo=algo
        self.args=args
        self.cut=66800 #16000 #66800      
    def __len__(self):
        return len(self.list_IDs)
    def __getitem__(self, index):
        utt_id = self.list_IDs[index]
        X, fs = librosa.load(self.base_dir+utt_id+'.wav', sr=16000) 
        # print("X",X.shape)
        #--------------------ADD_Rawboost_feature--------------------------
        Y=process_Rawboost_feature(X, fs, self.args, self.algo)
        # print("Y",Y.shape)
        
        #--------------------NO_Rawboost_feature---------------------------
        # X_pad= pad(X, self.cut)
        #------------------------------------------------------------------
        # print("X_pad",X_pad.shape)

        #######fixed train##############
        X_pad= pad(Y, self.cut)
        x_inp= Tensor(X_pad)
        
        # print("x_inp",x_inp.shape)
        target = self.labels[utt_id]
        return x_inp, target

class Dataset_fixed_eval(Dataset):
    def __init__(self, list_IDs, base_dir, track):
        '''self.list_IDs	: list of strings (each string: utt key),'''
        self.list_IDs = list_IDs
        self.base_dir = base_dir
        self.cut = 66800#(asvspoof5) #64000 #16000 #66800 # take ~4 sec audio 
        self.track = track
    def __len__(self):
        return len(self.list_IDs)
    def __getitem__(self, index):  
        utt_id = self.list_IDs[index]
        X, fs = librosa.load(self.base_dir+utt_id+'.wav', sr=16000)
        # print("X",X.shape)
        ######fixed test##############
        X_pad = pad(X,self.cut)
        # print("X_pad",X_pad.shape)
        x_inp = Tensor(X_pad)
        # print("x_inp",x_inp.shape)

        #######var test############
        # x_inp = Tensor(X)
        # print("x_inp",x_inp.shape)
        return x_inp, utt_id  


class Dataset_var_eval(Dataset):
    def __init__(self, list_IDs, base_dir, track):
        '''self.list_IDs	: list of strings (each string: utt key),'''
        self.list_IDs = list_IDs
        self.base_dir = base_dir
        self.cut = 66800 # take ~4 sec audio 
        self.track = track
    def __len__(self):
        return len(self.list_IDs)
    def __getitem__(self, index):  
        utt_id = self.list_IDs[index]
        X, fs = librosa.load(self.base_dir+utt_id+'.wav', sr=16000)
        # print("X",X.shape)
        ######fixed test##############
        # X_pad = pad(X,self.cut)
        # # print("X_pad",X_pad.shape)
        # x_inp = Tensor(X_pad)
        # # print("x_inp",x_inp.shape)

        #######var test############
        x_inp = Tensor(X)
        # print("x_inp",x_inp.shape)
        return x_inp, utt_id  
        # return x_inp, None, utt_id  


class Dataset_ASVSpoof19_eval(Dataset):
    def __init__(self, list_IDs, base_dir, track):
        '''self.list_IDs	: list of strings (each string: utt key),'''
        self.list_IDs = list_IDs
        self.base_dir = base_dir
        self.cut = 66800 # take ~4 sec audio 
        self.track = track
    def __len__(self):
        return len(self.list_IDs)
    def __getitem__(self, index):  
        utt_id = self.list_IDs[index]
        X, fs = librosa.load(self.base_dir+utt_id+'.wav', sr=16000)
        X_pad = pad(X,self.cut)
        x_inp = Tensor(X_pad)
        return x_inp, utt_id 
