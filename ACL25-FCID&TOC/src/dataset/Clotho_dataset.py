import os
import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import pickle
import zipfile
from io import BytesIO
    
# AT
class ClothoDataset(Dataset):

    def __init__(self, csv_path):
        super(ClothoDataset, self).__init__()
        self.id2cap = pd.read_csv(csv_path)
        self.audio_fea_base_path = '../clotho2.1/feature/zip'
        self.num_captions_per_audio = 5

    def __getitem__(self, index):
        audio_idx = index // self.num_captions_per_audio
        row = self.id2cap.iloc[audio_idx]  # 获取整行数据
        audio_name = row['file_name'][:-4]

        cap_idx = index % self.num_captions_per_audio + 1 #index从0开始计数
        cap = row['caption_'+str(cap_idx)]  # 获取caption

        audio_fea = self._load_fea(self.audio_fea_base_path, audio_name) 
        length = audio_fea.shape[0]

        # audio_fea = torch.mean(audio_fea,dim=0,keepdim=True)

        return audio_fea, cap, audio_idx, length, index

    def _load_fea(self, fea_base_path, audio_name):
        fea_path = os.path.join(fea_base_path, "%s.zip"%audio_name)
        with zipfile.ZipFile(fea_path, mode='r') as zfile:
            for name in zfile.namelist():
                if '.pkl' not in name:
                    continue
                with zfile.open(name, mode='r') as fea_file:
                    content = BytesIO(fea_file.read())
                    fea = pickle.load(content)
        return fea

    def __len__(self):
        return len(self.id2cap*self.num_captions_per_audio)
    