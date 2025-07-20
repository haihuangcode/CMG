import os
import numpy as np
import pandas as pd
import h5py
import torch
from torch.utils.data import Dataset, DataLoader
import pickle
import zipfile
from io import BytesIO

class AVEDataset(Dataset):
    def __init__(self, data_root, split='train'):
        super(AVEDataset, self).__init__()
        self.split = split
        feature_root = '../AVE-ECCV18-master/data'
        self.visual_feature_path = os.path.join(feature_root, 'visual_feature.h5')
        self.audio_feature_path = os.path.join(feature_root, 'audio_feature.h5')
        # Now for the supervised task
        self.labels_path = os.path.join(data_root, f'{split}_labels.h5')
        self.sample_order_path = os.path.join(data_root, f'{split}_order.h5')
        self.h5_isOpen = False

    def __getitem__(self, index):
        if not self.h5_isOpen:
            self.visual_feature = h5py.File(self.visual_feature_path, 'r')['avadataset']
            self.audio_feature = h5py.File(self.audio_feature_path, 'r')['avadataset']
            self.labels = h5py.File(self.labels_path, 'r')['avadataset']
            self.sample_order = h5py.File(self.sample_order_path, 'r')['order']
            self.h5_isOpen = True
        sample_index = self.sample_order[index]
        visual_feat = self.visual_feature[sample_index]
        audio_feat = self.audio_feature[sample_index]
        label = self.labels[sample_index]
        return visual_feat, audio_feat, label

    def __len__(self):
        f = h5py.File(self.sample_order_path, 'r')
        sample_num = len(f['order'])
        f.close()
        return sample_num

def generate_category_list():
    file_path = '../AVE_AVVP/AVE_AVVP_Categories.txt'
    category_list = []
    with open(file_path, 'r') as fr:
        for line in fr.readlines():
            category_list.append(line.strip())
    return category_list

class AVVPDataset(Dataset):
    # for AVEL task
    def __init__(self, meta_csv_path, fea_base_path, split='train', modality='video'):
        super(AVVPDataset, self).__init__()
        self.modality = modality
        self.fea_base_path = fea_base_path
        self.split_df = pd.read_csv(meta_csv_path,sep='\t')
        self.all_categories = generate_category_list()
        print(f'total {len(self.all_categories)} positive classes in AVVP, 1 negative classes in AVVP')
        print(f'{len(self.split_df)} samples are used for {split}')

    def __getitem__(self, index):
        one_video_df = self.split_df.iloc[index]
        categorys, video_id = one_video_df['event_labels'].split(','), one_video_df['filename']
        onsets, offsets = one_video_df['onset'].split(','), one_video_df['offset'].split(',')
        onsets = list(map(int, onsets))
        offsets = list(map(int, offsets))

        fea = self._load_fea(self.fea_base_path, video_id[:11])
        
        if(self.modality=='audio'):
            if fea.shape[0] < 10:
                cur_t = fea.shape[0]
                add_arr = np.tile(fea[-1, :], (10-cur_t, 1))
                fea = np.concatenate([fea, add_arr], axis=0)
            elif fea.shape[0] > 10:
                fea = fea[:10, :]
        
        avel_label = self._obtain_avel_label(onsets, offsets, categorys) # [10，26]
        
        return torch.from_numpy(fea), \
               torch.from_numpy(avel_label), \
               video_id
        
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
    
    def _obtain_avel_label(self, onsets, offsets, categorys):
        T, category_num = 10, len(self.all_categories)
        label = np.zeros((T, category_num + 1)) # add 'background' category [10, 25+1]
        label[:, -1] = np.ones(T) 
        iter_num = len(categorys)
        for i in range(iter_num):
            avc_label = np.zeros(T)
            avc_label[onsets[i]:offsets[i]] = 1
            class_id = self.all_categories.index(categorys[i])
            bg_flag = 1 - avc_label


            """
            The "&" operation on lists is used to find the common elements between two lists, 
            rather than performing a bitwise "and" operation on each element, 
            so it needs to be implemented using a loop.
            The reason for using "|" here is that if it is not "|", 
            but a simple assignment, it will cause the previous part of the same label to be overwritten.
            
            IgN7v8nWmx8_30_40	0,5,0,6,9	1,9,5,8,10	Speech,Speech,Violin_fiddle,Violin_fiddle,Violin_fiddle
            For example, in the example given above, 
            the second "Speech" will overwrite the first "Speech", so "|" operation needs to be used.
            """
            for j in range(10):
                label[j, class_id] = int(label[j, class_id]) | int(avc_label[j])
            
            """
            The "&" operation on lists is used to find the common elements between two lists, 
            rather than performing a bitwise "and" operation on each element, 
            so it needs to be implemented using a loop.
            """
            for j in range(10):
                label[j, -1] = int(label[j, -1]) & int(bg_flag[j])
        return label 

    def __len__(self,):
        return len(self.split_df)