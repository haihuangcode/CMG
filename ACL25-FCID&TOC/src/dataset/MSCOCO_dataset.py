import os
import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import pickle
import zipfile
from io import BytesIO
import json
    
# AT
class MSCOCODataset(Dataset):
    def __init__(self):
        super(MSCOCODataset, self).__init__()
        with open('../MSCOCO/val_part_1.json', 'r') as f:
            self.id2cap = json.load(f)
            self.id2cap = self.id2cap['annotations']
        self.video_fea_base_path = '../MSCOCO/feature/zip'
        # import pdb
        # pdb.set_trace()
        # print('22')


    def __getitem__(self, index):
        video_name, cap = self.id2cap[index]['image_id'], self.id2cap[index]['caption']
        video_name = f"{video_name:012}"
        video_fea = self._load_fea(self.video_fea_base_path, video_name) 
        length = video_fea.shape[0]

        # video_fea = torch.mean(video_fea,dim=0,keepdim=True)

        return video_fea, cap, index, length, index#第3个传index是为了和clotho那边一样的返回值，实际没用上，因为这里vt是一对一的

    def _load_fea(self, fea_base_path, video_name):
        fea_path = os.path.join(fea_base_path, "%s.zip"%video_name)
        with zipfile.ZipFile(fea_path, mode='r') as zfile:
            for name in zfile.namelist():
                if '.pkl' not in name:
                    continue
                with zfile.open(name, mode='r') as fea_file:
                    content = BytesIO(fea_file.read())
                    fea = pickle.load(content)
        return fea

    def __len__(self):
        return len(self.id2cap)
    