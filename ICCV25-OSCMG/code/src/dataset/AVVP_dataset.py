import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import pickle
import zipfile
from io import BytesIO

def generate_category_list():
    file_path = 'data/AVVP/data/AVVP_Categories.txt'
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
        # print("onsets:",onsets)
        # print("offsets:",offsets)
        
        # 数据是用的前11个char做的名字
        fea = self._load_fea(self.fea_base_path, video_id[:11])
        
        if(self.modality=='audio'):
            if fea.shape[0] < 10:
                cur_t = fea.shape[0]
                add_arr = np.tile(fea[-1, :], (10-cur_t, 1))
                fea = np.concatenate([fea, add_arr], axis=0)
            elif fea.shape[0] > 10:
                fea = fea[:10, :]
        
        avel_label = self._obtain_avel_label(onsets, offsets, categorys) # [10，26]
        
        # print("_____________________________________________________________________")
        # print(categorys)
        # print(video_id)
        # print(avel_label)
        
        return torch.from_numpy(fea), \
               torch.from_numpy(avel_label), \
               video_id
        
    def _load_fea(self, fea_base_path, video_id):
        fea_path = os.path.join(fea_base_path, "%s.zip"%video_id)
        with zipfile.ZipFile(fea_path, mode='r') as zfile:
            for name in zfile.namelist():
                if '.pkl' not in name:
                    continue
                with zfile.open(name, mode='r') as fea_file:#fea_file是.pkl文件
                    content = BytesIO(fea_file.read())
                    fea = pickle.load(content)
        return fea
    
    def _obtain_avel_label(self, onsets, offsets, categorys):# avc_label: [1, 10]
        T, category_num = 10, len(self.all_categories)
        # 25正标签和1个负标签
        label = np.zeros((T, category_num + 1)) # add 'background' category [10, 25+1]
        label[:, -1] = np.ones(T) # 负标签初始置为1
        iter_num = len(categorys)
        for i in range(iter_num):
            avc_label = np.zeros(T)
            avc_label[onsets[i]:offsets[i]] = 1
            class_id = self.all_categories.index(categorys[i])
            bg_flag = 1 - avc_label
            #list的&运算是求两个list里相同的元素，而不是这里想要实现的每一位相与，所以用循环实现
            # 这里之所以要做|是因为，如果不是|，而是单纯赋值，会导致同类的标签前面部分被覆盖
            # IgN7v8nWmx8_30_40	0,5,0,6,9	1,9,5,8,10	Speech,Speech,Violin_fiddle,Violin_fiddle,Violin_fiddle
            # 比如上面举得这个例子，第二个Speech会把第一个Speech覆盖，所以得做|运算
            for j in range(10):
                label[j, class_id] = int(label[j, class_id]) | int(avc_label[j])
            # label[:, class_id] = avc_label

            #list的&运算是求两个list里相同的元素，而不是这里想要实现的每一位相与，所以用循环实现
            for j in range(10):
                label[j, -1] = int(label[j, -1]) & int(bg_flag[j])
        return label 

    def __len__(self,):
        return len(self.split_df)


class AVVPDatasetTrain(Dataset):
    # for AVEL task
    def __init__(self, meta_csv_path, fea_base_path, split='train', modality='video'):
        super(AVVPDatasetTrain, self).__init__()
        self.modality = modality
        self.fea_base_path = fea_base_path
        self.split_df = pd.read_csv(meta_csv_path, sep='\t')
        self.all_categories = generate_category_list()
        print(f'total {len(self.all_categories)} classes in AVVPTrain')
        print(f'{len(self.split_df)} samples are used for Train')

    def __getitem__(self, index):
        one_video_df = self.split_df.iloc[index]
        categorys, video_id = one_video_df['event_labels'].split(','), one_video_df['filename']
        # 数据是用的前11个char做的名字
        fea = self._load_fea(self.fea_base_path, video_id[:11])
        if(self.modality=='audio'):
            if fea.shape[0] < 10:
                # if(fea.shape[0]==0):
                #     print(video_id)
                #     with open("/root/autodl-tmp/AVVP/data/data/audio_error_name.csv", "w") as csvfile: 
                #         writer = csv.writer(csvfile)
                #         writer.writerow([video_id])
                #     csvfile.close()
                # else:
                cur_t = fea.shape[0]
                add_arr = np.tile(fea[-1, :], (10-cur_t, 1))
                fea = np.concatenate([fea, add_arr], axis=0)
            elif fea.shape[0] > 10:
                fea = fea[:10, :]

        avc_label = np.ones(10) # [10，1]
        avel_label = self._obtain_avel_label(avc_label, categorys) # [10，26]

        # print("_____________________________________________________________________")
        # print(categorys)
        # print(video_id)
        # print(avel_label)

        return torch.from_numpy(fea), \
               torch.from_numpy(avel_label)
        
    def _load_fea(self, fea_base_path, video_id):
        fea_path = os.path.join(fea_base_path, "%s.zip"%video_id)
        with zipfile.ZipFile(fea_path, mode='r') as zfile:
            for name in zfile.namelist():
                if '.pkl' not in name:
                    continue
                with zfile.open(name, mode='r') as fea_file:#fea_file是.pkl文件
                    content = BytesIO(fea_file.read())
                    fea = pickle.load(content)
        return fea


    def _obtain_avel_label(self, avc_label, categorys):# avc_label: [1, 10]
        T, category_num = 10, len(self.all_categories)
        # 25正标签和1个负标签
        label = np.zeros((T, category_num + 1)) # add 'background' category [10, 25+1]
        for category in categorys:
            class_id = self.all_categories.index(category)
            bg_flag = 1 - avc_label
            label[:, class_id] = avc_label
            label[:, -1] = bg_flag

        return label 

    def __len__(self,):
        return len(self.split_df)
    
class AVVPDatasetEval(Dataset):
    # for AVEL task
    def __init__(self, meta_csv_path, fea_base_path, split='train', modality='video'):
        super(AVVPDatasetEval, self).__init__()
        self.modality = modality
        self.fea_base_path = fea_base_path
        self.split_df = pd.read_csv(meta_csv_path)
        self.all_categories = generate_category_list()
        print(f'total {len(self.all_categories)} classes in AVVPEval')
        print(f'{len(self.split_df)} samples are used for Eval')

    def __getitem__(self, index):
        one_video_df = self.split_df.iloc[index]
        category, video_id = one_video_df['event_labels'], one_video_df['filename']
        onset, offset = one_video_df['onset'].astype(int), one_video_df['offset'].astype(int)
        # 数据是用的前11个char做的名字
        fea = self._load_fea(self.fea_base_path, video_id[:11])
        
        if(self.modality=='audio'):
            if fea.shape[0] < 10:
                cur_t = fea.shape[0]
                add_arr = np.tile(fea[-1, :], (10-cur_t, 1))
                fea = np.concatenate([fea, add_arr], axis=0)
            elif fea.shape[0] > 10:
                fea = fea[:10, :]
        
        fea = fea[onset:offset, :]
        
        avc_label = np.ones(offset-onset) # [offset-onset，1]
        avel_label = self._obtain_avel_label(onset, offset, avc_label, category) # [offset-onset，26]
        sample = {'feature': torch.from_numpy(fea), 'label': torch.from_numpy(avel_label), 'length':offset-onset}
        # 面对变长序列需要特殊处理
        return sample
        # return torch.from_numpy(fea), \
        #        torch.from_numpy(avel_label)
        """
        1.部分视频不存在，如：'/root/autodl-tmp/AVVP/feature/video/zip/P7x4FV4lg_5.zip'
        2.提取后的视频有13867个,但dataset_full里面并没有这么多
        3.pad后的数据处理
        都解决了
        """
        
    def _load_fea(self, fea_base_path, video_id):
        fea_path = os.path.join(fea_base_path, "%s.zip"%video_id)
        with zipfile.ZipFile(fea_path, mode='r') as zfile:
            for name in zfile.namelist():
                if '.pkl' not in name:
                    continue
                with zfile.open(name, mode='r') as fea_file:#fea_file是.pkl文件
                    content = BytesIO(fea_file.read())
                    fea = pickle.load(content)
        return fea


    def _obtain_avel_label(self, onset, offset, avc_label, category):
        # avc_label: [1, 10]
        class_id = self.all_categories.index(category)
        T, category_num = offset-onset, len(self.all_categories)
        # 25正标签和1个负标签
        label = np.zeros((T, category_num + 1)) # add 'background' category [10, 25+1]
        bg_flag = 1 - avc_label
        # 每个样本都有两个标签，一个是事件能检测到（即audio+video融合能检测到event），另一个是background，除此之外全是空。
        label[:, class_id] = avc_label
        label[:, -1] = bg_flag

        return label 
    
    # def _obtain_avel_label(self, onset, offset, avc_label, categorys):# avc_label: [1, 10]
    #     T, category_num = 10, len(self.all_categories)
    #     # 25正标签和1个负标签
    #     label = np.zeros((T, category_num + 1)) # add 'background' category [10, 25+1]
    #     for category in categorys:
    #         class_id = self.all_categories.index(category)
    #         bg_flag = 1 - avc_label
    #         label[:, class_id] = avc_label
    #         label[:, -1] = bg_flag

    #     return label 

    def __len__(self,):
        return len(self.split_df)