import numpy as np
import torch
from torch.utils.data import Dataset
import pickle
import os
from torch.utils.data.dataloader import default_collate
import librosa
from utils import pad
from torch import Tensor

torch.set_default_tensor_type(torch.FloatTensor)

class SITW_all(Dataset):
    def __init__(self, path_to_protocol, part='train', padding='repeat'):
        self.base_dir='/nlp/nlp-xxuan/dataset/release_in_the_wild/'
        # self.ptd = path_to_database
        self.part = part
        protocol = '/home/xxuan/speech-deepfake/DF-SITW/SITW-labels.txt'
        self.path_to_protocol = protocol
        self.padding = padding
        self.cut=66800 #64000      

        with open(protocol, 'r') as f:
            audio_info = [info.strip().split(',') for info in f.readlines()]
            # print("audio_info",audio_info)
            self.all_info = audio_info

    def __len__(self):
        return len(self.all_info)

    def __getitem__(self, idx):
        filename, label = self.all_info[idx]
        # print("type",label.type)  
        # print("filename",filename)
        X, fs = librosa.load(self.base_dir+filename, sr=16000)
        X_pad = pad(X,self.cut)
        x_inp = Tensor(X_pad)
        label = int(label)  # 转换 label 为整数类型
        # print("type2",label.type) 
        return x_inp, filename, label

    def collate_fn(self, samples):
        return default_collate(samples)

class SITW_Var(Dataset):
    def __init__(self, path_to_protocol, part='train', padding='repeat'):
        self.base_dir='/nlp/nlp-xxuan/dataset/release_in_the_wild/'
        # self.ptd = path_to_database
        self.part = part
        protocol = '/home/xxuan/speech-deepfake/DF-SITW/SITW-labels.txt'
        self.path_to_protocol = protocol
        self.padding = padding
        self.cut=66800      

        with open(protocol, 'r') as f:
            audio_info = [info.strip().split(',') for info in f.readlines()]
            # print("audio_info",audio_info)
            self.all_info = audio_info

    def __len__(self):
        return len(self.all_info)

    def __getitem__(self, idx):
        filename, label = self.all_info[idx]
        # print("type",label.type)  
        # print("filename",filename)
        X, fs = librosa.load(self.base_dir+filename, sr=16000)
        # X_pad = pad(X,self.cut)
        x_inp = np.reshape(X,(1,-1))
        label = int(label)  # 转换 label 为整数类型
        # print("type2",label.type) 
        return x_inp, filename, label

    def collate_fn(self, samples):
        return default_collate(samples)

# class SITW(Dataset):
#     def __init__(self, path_to_protocol, part='train', padding='repeat'):
#         """
#         path_to_protocol: 子集的标签文件路径
#         part: 数据集部分，默认为 'train'
#         padding: 数据填充方式，默认为 'repeat'
#         """
#         self.base_dir = '/nlp/nlp-xxuan/dataset/release_in_the_wild/'
#         self.part = part
#         self.path_to_protocol = path_to_protocol  # 动态传入协议文件路径
#         self.padding = padding
#         self.cut = 66800  # 固定长度的音频片段

#         # 加载子集的音频信息
#         with open(self.path_to_protocol, 'r') as f:
#             audio_info = [info.strip().split(',') for info in f.readlines()]
#             self.all_info = audio_info  # 子集的所有样本信息

#     def __len__(self):
#         return len(self.all_info)

#     def __getitem__(self, idx):
#         """
#         根据索引返回样本，包括音频数据、文件名和标签
#         """
#         filename, label = self.all_info[idx]
#         X, fs = librosa.load(self.base_dir + filename, sr=16000)  # 加载音频
#         # X_pad = pad(X, self.cut)  # 填充音频到固定长度
#         # x_inp = Tensor(X_pad)

#         x_inp = np.reshape(X,(1,-1))
        
#         label = int(label)  # 将标签转换为整数
#         return x_inp, filename, label




class SITW(Dataset):
    def __init__(self, path_to_protocol, part='train', padding='repeat', min_length=10):
        """
        path_to_protocol: 子集的标签文件路径
        part: 数据集部分，默认为 'train'
        padding: 数据填充方式，可选 'repeat' 或 'zero'，默认为 'repeat'
        min_length: 设定最小长度，避免 `conv1d` 计算错误（默认 10）
        """
        self.base_dir = '/nlp/nlp-xxuan/dataset/release_in_the_wild/'
        self.part = part
        self.path_to_protocol = path_to_protocol  # 动态传入协议文件路径
        self.padding = padding
        self.min_length = min_length  # 设定最小长度，确保 `conv1d` 不会失败

        # 读取协议文件，加载子集音频信息
        with open(self.path_to_protocol, 'r') as f:
            audio_info = [info.strip().split(',') for info in f.readlines()]
            self.all_info = audio_info  # 子集的所有样本信息

    def __len__(self):
        return len(self.all_info)

    def pad_audio(self, X, target_length):
        """
        根据填充方式填充音频：
        - 'repeat'：重复填充
        - 'zero'：零填充
        """
        length = len(X)
        if length >= target_length:
            return X[:target_length]
        
        if self.padding == 'repeat':
            pad = np.tile(X, target_length // length + 1)[:target_length]
        else:  # 'zero' padding
            pad = np.zeros(target_length)
            pad[:length] = X
        
        return pad

    def __getitem__(self, idx):
        filename, label = self.all_info[idx]

        X, fs = librosa.load(self.base_dir+filename, sr=16000)
        x_inp = Tensor(X)
        label = int(label)  # 转换 label 为整数类型
        # print("type2",label.type) 
        return x_inp, filename, label



    def collate_fn(self, samples):
        """
        自定义批处理函数
        """
        return default_collate(samples)


def padding(spec, ref_len):
    width, cur_len = spec.shape
    assert ref_len > cur_len
    padd_len = ref_len - cur_len
    return torch.cat((spec, torch.zeros(width, padd_len, dtype=spec.dtype)), 1)

def repeat_padding(spec, ref_len):
    mul = int(np.ceil(ref_len / spec.shape[1]))
    spec = spec.repeat(1, mul)[:, :ref_len]
    return spec

class ASVspoof2019(Dataset):
    def __init__(self, access_type, path_to_protocol, part='train', genuine_only=False, padding='repeat'):
        self.access_type = access_type
        self.base_dir='/home/xxuan/speech-deepfake/conformer-based-classifier-for-anti-spoofing-master/datasets/LA/ASVspoof2019_LA_eval/'
        # self.ptd = path_to_database
        self.part = part
        # self.path_to_audio = os.path.join(self.ptd, access_type, 'ASVspoof2019_'+access_type+'_'+ self.part +'/flac/')
        self.genuine_only = genuine_only
        self.path_to_protocol = path_to_protocol
        self.padding = padding
        self.cut=66800      
        protocol = os.path.join(self.path_to_protocol, 'ASVspoof2019.'+access_type+'.cm.'+ self.part + '.trl.txt')
        if self.access_type == 'LA':
            self.tag = {"-": 0, "A01": 1, "A02": 2, "A03": 3, "A04": 4, "A05": 5, "A06": 6, "A07": 7, "A08": 8, "A09": 9,
                      "A10": 10, "A11": 11, "A12": 12, "A13": 13, "A14": 14, "A15": 15, "A16": 16, "A17": 17, "A18": 18,
                      "A19": 19}
        else:
            self.tag = {"-": 0, "AA": 1, "AB": 2, "AC": 3, "BA": 4, "BB": 5, "BC": 6, "CA": 7, "CB": 8, "CC": 9}
        self.label = {"spoof": 1, "bonafide": 0}

        with open(protocol, 'r') as f:
            audio_info = [info.strip().split() for info in f.readlines()]
            # print("audio_info",audio_info)
            if genuine_only:
                assert self.part in ["train", "dev"]
                if self.access_type == "LA":
                    num_bonafide = {"train": 2580, "dev": 2548}
                    self.all_info = audio_info[:num_bonafide[self.part]]
                else:
                    self.all_info = audio_info[:5400]
            else:
                self.all_info = audio_info

    def __len__(self):
        return len(self.all_info)

    def __getitem__(self, idx):
        speaker, filename, _, tag, label = self.all_info[idx]
        # print("filename",filename)
        X, fs = librosa.load(self.base_dir+filename+'.wav', sr=16000)
        X_pad = pad(X,self.cut)
        x_inp = Tensor(X_pad)
        
        # try:
        #     with open(self.ptf + '/'+ filename + self.feature + '.pkl', 'rb') as feature_handle:
        #         feat_mat = pickle.load(feature_handle)
        # except:
        #     # add this exception statement since we may change the data split
        #     def the_other(train_or_dev):
        #         assert train_or_dev in ["train", "dev"]
        #         res = "dev" if train_or_dev == "train" else "train"
        #         return res
        #     with open(os.path.join(self.path_to_features, the_other(self.part)) + '/'+ filename + self.feature + '.pkl', 'rb') as feature_handle:
        #         feat_mat = pickle.load(feature_handle)

        # feat_mat = torch.from_numpy(feat_mat)
        # this_feat_len = feat_mat.shape[1]
        # if this_feat_len > self.feat_len:
        #     startp = np.random.randint(this_feat_len-self.feat_len)
        #     feat_mat = feat_mat[:, startp:startp+self.feat_len]
        # if this_feat_len < self.feat_len:
        #     if self.padding == 'zero':
        #         feat_mat = padding(feat_mat, self.feat_len)
        #     elif self.padding == 'repeat':
        #         feat_mat = repeat_padding(feat_mat, self.feat_len)
        #     else:
        #         raise ValueError('Padding should be zero or repeat!')

        # return feat_mat, filename, self.tag[tag], self.label[label]
        return x_inp, filename, self.tag[tag], self.label[label]

    def collate_fn(self, samples):
        return default_collate(samples)

def padding(spec, ref_len):
    width, cur_len = spec.shape
    assert ref_len > cur_len
    padd_len = ref_len - cur_len
    return torch.cat((spec, torch.zeros(width, padd_len, dtype=spec.dtype)), 1)

def repeat_padding(spec, ref_len):
    mul = int(np.ceil(ref_len / spec.shape[1]))
    spec = spec.repeat(1, mul)[:, :ref_len]
    return spec

class ASVspoof2019_DEV(Dataset):
    def __init__(self, access_type, path_to_protocol, part='train', genuine_only=False, padding='repeat'):
        self.access_type = access_type
        self.base_dir='/home/xxuan/speech-deepfake/conformer-based-classifier-for-anti-spoofing-master/datasets/cleaned/ASVspoof2019_LA_dev/'
        # self.ptd = path_to_database
        self.part = part
        # self.path_to_audio = os.path.join(self.ptd, access_type, 'ASVspoof2019_'+access_type+'_'+ self.part +'/flac/')
        self.genuine_only = genuine_only
        self.path_to_protocol = path_to_protocol
        self.padding = padding
        self.cut=66800      
        protocol = os.path.join(self.path_to_protocol, 'ASVspoof2019.'+access_type+'.cm.'+ self.part + '.trl.txt')
        if self.access_type == 'LA':
            self.tag = {"-": 0, "A01": 1, "A02": 2, "A03": 3, "A04": 4, "A05": 5, "A06": 6, "A07": 7, "A08": 8, "A09": 9,
                      "A10": 10, "A11": 11, "A12": 12, "A13": 13, "A14": 14, "A15": 15, "A16": 16, "A17": 17, "A18": 18,
                      "A19": 19}
        else:
            self.tag = {"-": 0, "AA": 1, "AB": 2, "AC": 3, "BA": 4, "BB": 5, "BC": 6, "CA": 7, "CB": 8, "CC": 9}
        self.label = {"spoof": 1, "bonafide": 0}

        with open(protocol, 'r') as f:
            audio_info = [info.strip().split() for info in f.readlines()]
            # print("audio_info",audio_info)
            if genuine_only:
                assert self.part in ["train", "dev"]
                if self.access_type == "LA":
                    num_bonafide = {"train": 2580, "dev": 2548}
                    self.all_info = audio_info[:num_bonafide[self.part]]
                else:
                    self.all_info = audio_info[:5400]
            else:
                self.all_info = audio_info

    def __len__(self):
        return len(self.all_info)

    def __getitem__(self, idx):
        speaker, filename, _, tag, label = self.all_info[idx]
        # print("filename",filename)
        X, fs = librosa.load(self.base_dir+filename+'.wav', sr=16000)
        X_pad = pad(X,self.cut)
        x_inp = Tensor(X_pad)
        
        # try:
        #     with open(self.ptf + '/'+ filename + self.feature + '.pkl', 'rb') as feature_handle:
        #         feat_mat = pickle.load(feature_handle)
        # except:
        #     # add this exception statement since we may change the data split
        #     def the_other(train_or_dev):
        #         assert train_or_dev in ["train", "dev"]
        #         res = "dev" if train_or_dev == "train" else "train"
        #         return res
        #     with open(os.path.join(self.path_to_features, the_other(self.part)) + '/'+ filename + self.feature + '.pkl', 'rb') as feature_handle:
        #         feat_mat = pickle.load(feature_handle)

        # feat_mat = torch.from_numpy(feat_mat)
        # this_feat_len = feat_mat.shape[1]
        # if this_feat_len > self.feat_len:
        #     startp = np.random.randint(this_feat_len-self.feat_len)
        #     feat_mat = feat_mat[:, startp:startp+self.feat_len]
        # if this_feat_len < self.feat_len:
        #     if self.padding == 'zero':
        #         feat_mat = padding(feat_mat, self.feat_len)
        #     elif self.padding == 'repeat':
        #         feat_mat = repeat_padding(feat_mat, self.feat_len)
        #     else:
        #         raise ValueError('Padding should be zero or repeat!')

        # return feat_mat, filename, self.tag[tag], self.label[label]
        return x_inp, filename, self.tag[tag], self.label[label]

    def collate_fn(self, samples):
        return default_collate(samples)

def padding(spec, ref_len):
    width, cur_len = spec.shape
    assert ref_len > cur_len
    padd_len = ref_len - cur_len
    return torch.cat((spec, torch.zeros(width, padd_len, dtype=spec.dtype)), 1)

def repeat_padding(spec, ref_len):
    mul = int(np.ceil(ref_len / spec.shape[1]))
    spec = spec.repeat(1, mul)[:, :ref_len]
    return spec


if __name__ == "__main__":
    # path_to_database = '/data/neil/DS_10283_3336/'  # if run on GPU
    path_to_features = '/dataNVME/neil/ASVspoof2019Features/'  # if run on GPU
    path_to_protocol = '/data/neil/DS_10283_3336/LA/ASVspoof2019_LA_cm_protocols/'

