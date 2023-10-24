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
import random
SEED = 57
random.seed(SEED)

def generate_category_list():
    file_path = 'VggsoundAVEL40kCategories.txt'
    category_list = []
    with open(file_path, 'r') as fr:
        for line in fr.readlines():
            category_list.append(line.strip())
    return category_list

# AV
class VGGSoundDataset(Dataset):
    def __init__(self, meta_csv_path, audio_fea_base_path, video_fea_base_path, avc_label_base_path, split='train'):
        super(VGGSoundDataset, self).__init__()
        self.audio_fea_base_path = audio_fea_base_path
        self.video_fea_base_path = video_fea_base_path
        self.avc_label_base_path = avc_label_base_path
        all_df = pd.read_csv(meta_csv_path)
        # train = 24k
        # train + test + val = 40k
        df_train = all_df[all_df['split'] == 'train']
        df_test = all_df[all_df['split'] == 'test']
        df_val = all_df[all_df['split'] == 'val']
        self.split_df = pd.concat([df_train,df_test,df_val])
        # Output the proportion of train, test, and valid.
        print(f'{len(self.split_df)}/{len(all_df)} videos are used for {split}')
        self.all_categories = generate_category_list()
        print(f'total {len(self.all_categories)} classes in VggsoundAVEL40k')

    def __getitem__(self, index):
        one_video_df = self.split_df.iloc[index]
        category, video_id = one_video_df['category'], one_video_df['video_id']

        audio_fea = self._load_fea(self.audio_fea_base_path, video_id) # [10, 128]
        video_fea = self._load_fea(self.video_fea_base_path, video_id) # [10, 7, 7, 512]
        avc_label = self._load_fea(self.avc_label_base_path, video_id) # [10，1]
        avel_label = self._obtain_avel_label(avc_label, category) # [10，142]

        if audio_fea.shape[0] < 10:
            cur_t = audio_fea.shape[0]
            add_arr = np.tile(audio_fea[-1, :], (10-cur_t, 1))
            audio_fea = np.concatenate([audio_fea, add_arr], axis=0)
        elif audio_fea.shape[0] > 10:
            audio_fea = audio_fea[:10, :]

        return torch.from_numpy(video_fea), \
               torch.from_numpy(audio_fea), \
               torch.from_numpy(avel_label)

        
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
    
# AT
class VGGSoundDataset_AT(Dataset):

    def __init__(self, meta_csv_path, audio_fea_base_path, split='train'):
        super(VGGSoundDataset_AT, self).__init__()
        self.label2prompt = pd.read_csv('vggsoundCategories2Prompts.csv')
        self.audio_fea_base_path = audio_fea_base_path
        all_df = pd.read_csv(meta_csv_path)

        df_train = all_df[all_df['split'] == 'train']
        df_test = all_df[all_df['split'] == 'test']
        df_val = all_df[all_df['split'] == 'val']
        self.split_df = pd.concat([df_train,df_test,df_val])

        print(f'{len(self.split_df)}/{len(all_df)} audios are used for {split}')
        self.all_categories = generate_category_list()
        print(f'total {len(self.all_categories)} classes in Vggsound40K_AT')

    def __getitem__(self, index):
        one_video_df = self.split_df.iloc[index]
        category, audio_id = one_video_df['category'], one_video_df['video_id']

        audio_fea = self._load_fea(self.audio_fea_base_path, audio_id) # [10, 128]
        text_fea = self.label2prompt.loc[self.label2prompt['label'] == category].values[0][1]
       
        if audio_fea.shape[0] < 10:
            cur_t = audio_fea.shape[0]
            add_arr = np.tile(audio_fea[-1, :], (10-cur_t, 1))
            audio_fea = np.concatenate([audio_fea, add_arr], axis=0)
        elif audio_fea.shape[0] > 10:
            audio_fea = audio_fea[:10, :]

        sample = {'audio_fea': audio_fea, 'text_fea': text_fea}
        return sample

    def _load_fea(self, fea_base_path, audio_id):
        fea_path = os.path.join(fea_base_path, "%s.zip"%audio_id)
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
    

#AVT
class VGGSoundDataset_AVT(Dataset):
    def __init__(self, meta_csv_path, audio_fea_base_path, video_fea_base_path, split='train'):
        super(VGGSoundDataset_AVT, self).__init__()
        self.label2prompt = pd.read_csv('vggsoundCategories2Prompts.csv')
        self.audio_fea_base_path = audio_fea_base_path
        self.video_fea_base_path = video_fea_base_path
        all_df = pd.read_csv(meta_csv_path)

        df_train = all_df[all_df['split'] == 'train']
        df_test = all_df[all_df['split'] == 'test']
        df_val = all_df[all_df['split'] == 'val']
        self.split_df = pd.concat([df_train,df_test,df_val])

        print(f'{len(self.split_df)}/{len(all_df)} samples are used for {split}')
        self.all_categories = generate_category_list()
        print(f'total {len(self.all_categories)} classes in Vggsound40K_AVT')

    def __getitem__(self, index):
        one_video_df = self.split_df.iloc[index]
        category, audio_id = one_video_df['category'], one_video_df['video_id']

        audio_fea = self._load_fea(self.audio_fea_base_path, audio_id) # [10, 128]
        video_fea = self._load_fea(self.video_fea_base_path, audio_id) # [10, 7, 7, 512]
        text_fea = self.label2prompt.loc[self.label2prompt['label'] == category].values[0][1]
        
        if audio_fea.shape[0] < 10:
            cur_t = audio_fea.shape[0]
            add_arr = np.tile(audio_fea[-1, :], (10-cur_t, 1))
            audio_fea = np.concatenate([audio_fea, add_arr], axis=0)
        elif audio_fea.shape[0] > 10:
            audio_fea = audio_fea[:10, :]

        sample = {'video_fea': video_fea, 'audio_fea': audio_fea, 'text_fea': text_fea}
        return sample

    def _load_fea(self, fea_base_path, audio_id):
        fea_path = os.path.join(fea_base_path, "%s.zip"%audio_id)
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