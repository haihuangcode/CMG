import os
import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms, models
# from PIL import Image
# from tqdm import tqdm
import pickle
import zipfile
from io import BytesIO
import pdb
import random
import json


# VT(pretrain)
# class Valor32k_VT(Dataset):
#     def __init__(self, video_fea_base_path):
#         super(Valor32k_VT, self).__init__()
#         with open('../VALOR32K/VALOR-32K-annotations/valor32k.json', 'r') as f:
#             self.id2cap_list = json.load(f)
#         # 创建一个字典，将 video_id 映射到 desc
#         # self.id2cap = {item['video_id']: item['desc'] for item in id2cap_list}
#         self.video_fea_base_path = video_fea_base_path

#     def __getitem__(self, index):
#         video_id = self.id2cap_list[index]['video_id']
#         cap = self.id2cap_list[index]['desc']
#         video_fea = self._load_fea(self.video_fea_base_path, video_id) # [10, 128]
       
#         if video_fea.shape[0] < 10:
#             cur_t = video_fea.shape[0]
#             add_arr = np.tile(video_fea[-1, :], (10-cur_t, 1))
#             video_fea = np.concatenate([video_fea, add_arr], axis=0)
#         elif video_fea.shape[0] > 10:
#             video_fea = video_fea[:10, :]

#         sample = {'feature': video_fea, 'cap': cap}
#         return sample

#     def _load_fea(self, fea_base_path, video_id):
#         fea_path = os.path.join(fea_base_path, "%s.zip"%video_id)
#         with zipfile.ZipFile(fea_path, mode='r') as zfile:
#             for name in zfile.namelist():
#                 if '.pkl' not in name:
#                     continue
#                 with zfile.open(name, mode='r') as fea_file:
#                     content = BytesIO(fea_file.read())
#                     fea = pickle.load(content)
#         return fea

#     def __len__(self):
#         return len(self.id2cap_list)
    
class Valor32k_VT(Dataset):
    def __init__(self, mode):
        super(Valor32k_VT, self).__init__()
        if mode == 'train':
            with open('../VALOR32K/VALOR-32K-annotations/desc_train.json', 'r') as f:
                self.id2cap_list = json.load(f)
        elif mode == 'val':
            with open('../VALOR32K/VALOR-32K-annotations/desc_val.json', 'r') as f:
                self.id2cap_list = json.load(f)
        elif mode == 'test':
            with open('../VALOR32K/VALOR-32K-annotations/desc_test.json', 'r') as f:
                self.id2cap_list = json.load(f)
        elif mode == '500':
            with open('../VALOR32K/VALOR-32K-annotations/random_500.json', 'r') as f:
                self.id2cap_list = json.load(f)
        elif mode == '1000':
            with open('../VALOR32K/VALOR-32K-annotations/random_1000.json', 'r') as f:
                self.id2cap_list = json.load(f)
        # 创建一个字典，将 video_id 映射到 desc
        # self.id2cap = {item['video_id']: item['desc'] for item in id2cap_list}
        self.video_fea_base_path = '../VALOR32K/feature/video/zip'

    def __getitem__(self, index):
        video_id = self.id2cap_list[index]['video_id']
        cap = self.id2cap_list[index]['desc']
        video_fea = self._load_fea(self.video_fea_base_path, video_id) # [10, 128]
        length = video_fea.shape[0]

        return video_fea, cap, index, length, index

    def _load_fea(self, fea_base_path, video_id):
        fea_path = os.path.join(fea_base_path, "%s.zip"%video_id)
        with zipfile.ZipFile(fea_path, mode='r') as zfile:
            for name in zfile.namelist():
                if '.pkl' not in name:
                    continue
                with zfile.open(name, mode='r') as fea_file:
                    content = BytesIO(fea_file.read())
                    fea = pickle.load(content)
        return fea

    def __len__(self):
        return len(self.id2cap_list)
    
# 内容和VT一样
class Valor32k_AT(Dataset):
    def __init__(self, mode):
        super(Valor32k_AT, self).__init__()
        if mode == 'train':
            with open('../VALOR32K/VALOR-32K-annotations/desc_train.json', 'r') as f:
                self.id2cap_list = json.load(f)
        elif mode == 'val':
            with open('../VALOR32K/VALOR-32K-annotations/desc_val.json', 'r') as f:
                self.id2cap_list = json.load(f)
        elif mode == 'test':
            with open('../VALOR32K/VALOR-32K-annotations/desc_test.json', 'r') as f:
                self.id2cap_list = json.load(f)
        elif mode == '500':
            with open('../VALOR32K/VALOR-32K-annotations/random_500.json', 'r') as f:
                self.id2cap_list = json.load(f)
        elif mode == '1000':
            with open('../VALOR32K/VALOR-32K-annotations/random_1000.json', 'r') as f:
                self.id2cap_list = json.load(f)
        # 创建一个字典，将 video_id 映射到 desc
        # self.id2cap = {item['video_id']: item['desc'] for item in id2cap_list}
        self.video_fea_base_path = '../VALOR32K/feature/audio/zip'

    def __getitem__(self, index):
        video_id = self.id2cap_list[index]['video_id']
        cap = self.id2cap_list[index]['desc']
        video_fea = self._load_fea(self.video_fea_base_path, video_id) # [10, 128]
        length = video_fea.shape[0]

        return video_fea, cap, index, length, index

    def _load_fea(self, fea_base_path, video_id):
        fea_path = os.path.join(fea_base_path, "%s.zip"%video_id)
        with zipfile.ZipFile(fea_path, mode='r') as zfile:
            for name in zfile.namelist():
                if '.pkl' not in name:
                    continue
                with zfile.open(name, mode='r') as fea_file:
                    content = BytesIO(fea_file.read())
                    fea = pickle.load(content)
        return fea

    def __len__(self):
        return len(self.id2cap_list)
    
class Valor32k_VA(Dataset):
    def __init__(self, mode):
        super(Valor32k_VA, self).__init__()
        if mode == 'train':
            with open('../VALOR32K/VALOR-32K-annotations/desc_train.json', 'r') as f:
                self.id2cap_list = json.load(f)
        elif mode == 'val':
            with open('../VALOR32K/VALOR-32K-annotations/desc_val.json', 'r') as f:
                self.id2cap_list = json.load(f)
        elif mode == 'test':
            with open('../VALOR32K/VALOR-32K-annotations/desc_test.json', 'r') as f:
                self.id2cap_list = json.load(f)
        elif mode == '500':
            with open('../VALOR32K/VALOR-32K-annotations/random_500.json', 'r') as f:
                self.id2cap_list = json.load(f)
        elif mode == '1000':
            with open('../VALOR32K/VALOR-32K-annotations/random_1000.json', 'r') as f:
                self.id2cap_list = json.load(f)
        # 创建一个字典，将 video_id 映射到 desc
        # self.id2cap = {item['video_id']: item['desc'] for item in id2cap_list}
        self.video_fea_base_path = '../VALOR32K/feature/video/zip'
        self.audio_fea_base_path = '../VALOR32K/feature/audio/zip'

    def __getitem__(self, index):
        video_id = self.id2cap_list[index]['video_id']
        video_fea = self._load_fea(self.video_fea_base_path, video_id) # [10, 128]
        audio_fea = self._load_fea(self.audio_fea_base_path, video_id) # [10, 128]
        if audio_fea.shape[0] < 10:
            cur_t = audio_fea.shape[0]
            add_arr = np.tile(audio_fea[-1, :], (10-cur_t, 1))
            audio_fea = np.concatenate([audio_fea, add_arr], axis=0)
        elif audio_fea.shape[0] > 10:
            audio_fea = audio_fea[:10, :]

        length = video_fea.shape[0]

        return video_fea, audio_fea, index, length, index

    def _load_fea(self, fea_base_path, video_id):
        fea_path = os.path.join(fea_base_path, "%s.zip"%video_id)
        with zipfile.ZipFile(fea_path, mode='r') as zfile:
            for name in zfile.namelist():
                if '.pkl' not in name:
                    continue
                with zfile.open(name, mode='r') as fea_file:
                    content = BytesIO(fea_file.read())
                    fea = pickle.load(content)
        return fea

    def __len__(self):
        return len(self.id2cap_list)
    
    
class Valor32k_VAT(Dataset):
    def __init__(self, mode):
        super(Valor32k_VAT, self).__init__()
        if mode == 'train':
            with open('../VALOR32K/VALOR-32K-annotations/desc_train.json', 'r') as f:
                self.id2cap_list = json.load(f)
        elif mode == 'val':
            with open('../VALOR32K/VALOR-32K-annotations/desc_val.json', 'r') as f:
                self.id2cap_list = json.load(f)
        elif mode == 'test':
            with open('../VALOR32K/VALOR-32K-annotations/desc_test.json', 'r') as f:
                self.id2cap_list = json.load(f)
        elif mode == '500':
            with open('../VALOR32K/VALOR-32K-annotations/random_500.json', 'r') as f:
                self.id2cap_list = json.load(f)
        elif mode == '1000':
            with open('../VALOR32K/VALOR-32K-annotations/random_1000.json', 'r') as f:
                self.id2cap_list = json.load(f)
        # 创建一个字典，将 video_id 映射到 desc
        # self.id2cap = {item['video_id']: item['desc'] for item in id2cap_list}
        self.video_fea_base_path = '../VALOR32K/feature/video/zip'
        self.audio_fea_base_path = '../VALOR32K/feature/audio/zip'

    def __getitem__(self, index):
        text_fea = self.id2cap_list[index]['desc']
        video_id = self.id2cap_list[index]['video_id']
        video_fea = self._load_fea(self.video_fea_base_path, video_id) # [10, 128]
        audio_fea = self._load_fea(self.audio_fea_base_path, video_id) # [10, 128]
        if audio_fea.shape[0] < 10:
            cur_t = audio_fea.shape[0]
            add_arr = np.tile(audio_fea[-1, :], (10-cur_t, 1))
            audio_fea = np.concatenate([audio_fea, add_arr], axis=0)
        elif audio_fea.shape[0] > 10:
            audio_fea = audio_fea[:10, :]

        length = video_fea.shape[0]

        return text_fea, video_fea, audio_fea, index, length, index

    def _load_fea(self, fea_base_path, video_id):
        fea_path = os.path.join(fea_base_path, "%s.zip"%video_id)
        with zipfile.ZipFile(fea_path, mode='r') as zfile:
            for name in zfile.namelist():
                if '.pkl' not in name:
                    continue
                with zfile.open(name, mode='r') as fea_file:
                    content = BytesIO(fea_file.read())
                    fea = pickle.load(content)
        return fea

    def __len__(self):
        return len(self.id2cap_list)