import os
import h5py
import torch
import pandas as pd
import pickle
import zipfile
from io import BytesIO
import numpy as np
from torch.utils.data import Dataset, DataLoader


def generate_category_list_vgg2ucf():
    file_path = 'VGGSoundsameUCF101.txt'
    category_list = []
    with open(file_path, 'r') as fr:
        for line in fr.readlines():
            category_list.append(line.strip())
    return category_list

def generate_category_list_ucf2vgg():
    file_path = 'UCF101sameVGGSound.txt'
    category_list = []
    with open(file_path, 'r') as fr:
        for line in fr.readlines():
            category_list.append(line.strip())
    return category_list

class VGGSoundDataset(Dataset):
    def __init__(self, meta_csv_path, fea_base_path, split=None, modality=None):
        super(VGGSoundDataset, self).__init__()
        self.modality = modality
        self.fea_base_path = fea_base_path
        self.split_df = pd.read_csv(meta_csv_path,sep=',')
        self.all_categories = generate_category_list_vgg2ucf()

    def __getitem__(self, index):
        one_video_df = self.split_df.iloc[index]
        video_id = one_video_df['video_id']
        category = one_video_df['category']

        audio_fea = self._load_fea(self.fea_base_path, video_id) # [10, 128]
        avc_label = np.ones(10)
        avel_label = self._obtain_avel_label(avc_label, category) 

        if self.modality=='audio':
            if audio_fea.shape[0] < 10:
                cur_t = audio_fea.shape[0]
                add_arr = np.tile(audio_fea[-1, :], (10-cur_t, 1))
                audio_fea = np.concatenate([audio_fea, add_arr], axis=0)
            elif audio_fea.shape[0] > 10:
                audio_fea = audio_fea[:10, :]
            audio_fea = audio_fea.astype(np.float64)
        
        return torch.from_numpy(audio_fea), torch.from_numpy(avel_label)

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
    
    def _obtain_avel_label(self, avc_label, category):
        # avc_label: [1, 10]
        class_id = self.all_categories.index(category)
        T, category_num = 10, len(self.all_categories)
       
        label = np.zeros((T, category_num + 1)) # add 'background' category [10, 141+1]
        bg_flag = 1 - avc_label
        label[:, class_id] = avc_label
        label[:, -1] = bg_flag
        return label 

    def __len__(self,):
        return len(self.split_df)
    
class UCFDataset(Dataset):
    def __init__(self, meta_csv_path, fea_base_path, split=None, modality=None):
        super(UCFDataset, self).__init__()
        self.modality = modality
        self.fea_base_path = fea_base_path
        self.split_df = pd.read_csv(meta_csv_path,sep=',')
        self.all_categories = generate_category_list_ucf2vgg()

    def __getitem__(self, index):
        one_video_df = self.split_df.iloc[index]
        video_id = one_video_df['video_id']
        category = one_video_df['category']

        video_fea = self._load_fea(self.fea_base_path, video_id) # [10, 7, 7, 512]
        avc_label = np.ones(10)
        avel_label = self._obtain_avel_label(avc_label, category) # [10，17]

        if video_fea.shape[0] < 10:
            cur_t = video_fea.shape[0]
            add_arr = np.tile(video_fea[-1, :], (10-cur_t,1,1,1))
            video_fea = np.concatenate([video_fea, add_arr], axis=0)
        elif video_fea.shape[0] > 10:
            video_fea = video_fea[:10, :, :, :]
            
        video_fea = video_fea.astype(np.float64)
        
        return torch.from_numpy(video_fea), torch.from_numpy(avel_label)

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
    
    def _obtain_avel_label(self, avc_label, category):
        # avc_label: [1, 10]
        class_id = self.all_categories.index(category)
        T, category_num = 10, len(self.all_categories)
        label = np.zeros((T, category_num + 1)) # add 'background' category [10, 141+1]
        bg_flag = 1 - avc_label
        label[:, class_id] = avc_label
        label[:, -1] = bg_flag
        return label 

    def __len__(self,):
        return len(self.split_df)