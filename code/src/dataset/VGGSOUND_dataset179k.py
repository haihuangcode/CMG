"""used for train 81k"""
import os
import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm
import pickle
import zipfile
from io import BytesIO
import pdb
import csv

# The number of categories is the same for 81k and 40k.
def generate_category_list():
    file_path = 'VggsoundAVEL40kCategories.txt'
    category_list = []
    with open(file_path, 'r') as fr:
        for line in fr.readlines():
            category_list.append(line.strip())
    return category_list


class VGGSoundDataset(Dataset):
    def __init__(self, meta_csv_path, audio_fea_base_path, video_fea_base_path, avc_label_base_path, split='train'):
        super(VGGSoundDataset, self).__init__()
        self.audio_fea_base_path = audio_fea_base_path
        self.video_fea_base_path = video_fea_base_path
        self.avc_label_base_path = avc_label_base_path
        all_df = pd.read_csv(meta_csv_path)
        self.split_df = all_df
        # Output the proportion of train, test, and valid.
        print(f'{len(self.split_df)}/{len(all_df)} videos are used for {split}')
        self.all_categories = generate_category_list()
        print(f'total {len(self.all_categories)} classes in VggsoundAVEL81k')


    def __getitem__(self, index):
        one_video_df = self.split_df.iloc[index]
        video_id = one_video_df['id'][:-4]# drop '.mp4'

        audio_fea = self._load_fea(self.audio_fea_base_path, video_id) # [10, 128]
        video_fea = self._load_fea(self.video_fea_base_path, video_id) # [10, 7, 7, 512]

        if audio_fea.shape[0] < 10:
            cur_t = audio_fea.shape[0]
            add_arr = np.tile(audio_fea[-1, :], (10-cur_t, 1))
            audio_fea = np.concatenate([audio_fea, add_arr], axis=0)
        elif audio_fea.shape[0] > 10:
            audio_fea = audio_fea[:10, :]
        
        return torch.from_numpy(video_fea), \
               torch.from_numpy(audio_fea)

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

    def __len__(self,):
        return len(self.split_df)