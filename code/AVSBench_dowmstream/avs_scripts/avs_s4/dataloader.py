import os
from wave import _wave_params
import torch
import torch.nn as nn
from torch.utils.data import Dataset

import numpy as np
import pandas as pd
import pickle

import cv2
from PIL import Image
from torchvision import transforms

from config import cfg
import pdb
from bert_embedding import BertEmbedding
import pickle
import zipfile
from io import BytesIO

def load_image_in_PIL_to_Tensor(path, mode='RGB', transform=None):
    img_PIL = Image.open(path).convert(mode)
    if transform:
        img_tensor = transform(img_PIL)
        return img_tensor
    return img_PIL


def load_audio_lm(audio_lm_path):
    with open(audio_lm_path, 'rb') as fr:
        audio_log_mel = pickle.load(fr)
    audio_log_mel = audio_log_mel.detach() # [5, 1, 96, 64]
    return audio_log_mel

bert_embedding = BertEmbedding()
with open('cnt.pkl', 'rb') as fp:
    id2idx = pickle.load(fp)

class S4Dataset(Dataset):
    """Dataset for single sound source segmentation"""
    def __init__(self, split='train'):
        super(S4Dataset, self).__init__()
        self.split = split
        self.mask_num = 1 if self.split == 'train' else 5
        self.label2prompt = pd.read_csv('AVSBenchCategories2Prompts.csv')
        df_all = pd.read_csv(cfg.DATA.ANNO_CSV, sep=',')
        self.df_split = df_all[df_all['split'] == split]
        print("{}/{} videos are used for {}".format(len(self.df_split), len(df_all), self.split))
        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        self.mask_transform = transforms.Compose([
            transforms.ToTensor(),
        ])


    def __getitem__(self, index):
        df_one_video = self.df_split.iloc[index]
        video_name, category = df_one_video[0], df_one_video[2]
        img_base_path =  os.path.join(cfg.DATA.DIR_IMG, self.split, category, video_name)
        # audio_lm_path = os.path.join(cfg.DATA.DIR_AUDIO_LOG_MEL, self.split, category, video_name + '.pkl')
        audio_feature_path = os.path.join(cfg.DATA.DIR_AUDIO_FEATURE, self.split, "zip",category)
        mask_base_path = os.path.join(cfg.DATA.DIR_MASK, self.split, category, video_name)
        # audio_log_mel = load_audio_lm(audio_lm_path)
        audio_feature = self._load_fea(audio_feature_path, video_name)
        
        if audio_feature.shape[0] < 5:
            cur_t = audio_feature.shape[0]
            add_arr = np.tile(audio_feature[-1, :], (5-cur_t, 1))
            audio_feature = np.concatenate([audio_feature, add_arr], axis=0)
        elif audio_feature.shape[0] > 5:
            audio_feature = audio_feature[:5, :]
        
        
        text_fea = self.label2prompt.loc[self.label2prompt['label'] == category].values[0][1]
        # audio_lm_tensor = torch.from_numpy(audio_log_mel)
        imgs, masks = [], []
        for img_id in range(1, 6):
            img = load_image_in_PIL_to_Tensor(os.path.join(img_base_path, "%s_%d.png"%(video_name, img_id)), transform=self.img_transform)
            imgs.append(img)
        for mask_id in range(1, self.mask_num + 1):
            mask = load_image_in_PIL_to_Tensor(os.path.join(mask_base_path, "%s_%d.png"%(video_name, mask_id)), transform=self.mask_transform, mode='1')
            masks.append(mask)
        imgs_tensor = torch.stack(imgs, dim=0)
        masks_tensor = torch.stack(masks, dim=0)
        
        sample = {'imgs_tensor': imgs_tensor,
                      'audio_fea': audio_feature,
                      'masks_tensor': masks_tensor,
                      'category': category,
                      'video_name': video_name,
                      'text_fea': text_fea}
        
        return sample
        
           
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
        return len(self.df_split)
    

    def collate_func(self, samples):
        
        bsz = len(samples)
        result = bert_embedding([sample['text_fea'] for sample in samples])
        query = []
        query_words = []
        for a, b in result:
            words = []
            words_emb = []
            for word, emb in zip(a, b):
                idx = bert_embedding.vocab.token_to_idx[word]
                if idx in id2idx and idx != 0:
                    words_emb.append(emb)
                    words.append(id2idx[idx])
            query.append(np.asarray(words_emb))
            query_words.append(words)

        query_len = []
        for i, sample in enumerate(query):
            # query_len.append(min(len(sample), 10))#max_num_words:10
            query_len.append(10)#max_num_words:10
        query1 = np.zeros([bsz, max(query_len), 768]).astype(np.float32)
        query_idx = np.zeros([bsz, max(query_len)]).astype(np.float32)
        for i, sample in enumerate(query):
            keep = min(sample.shape[0], query1.shape[1])
            query1[i, :keep] = sample[:keep]
            query_idx[i, :keep] = query_words[i][:keep]
        query_len = np.asarray(query_len)
        query, query_len = torch.from_numpy(query1).float(), torch.from_numpy(query_len).long()
        query_idx = torch.from_numpy(query_idx).long()
        
        image_tensors = [sample['imgs_tensor'] for sample in samples]
        stacked_images = np.stack(image_tensors)
        imgs_tensor = torch.from_numpy(stacked_images).float()
        
        maskeds_tensors = [sample['masks_tensor'] for sample in samples]
        stacked_masks = np.stack(maskeds_tensors)
        masks_tensor = torch.from_numpy(stacked_masks).float()
        
        categorys = [sample['category'] for sample in samples]
        video_names = [sample['video_name'] for sample in samples]
    
        return {
            'query': query,
            'imgs_tensor':imgs_tensor,
            'audio_fea': torch.from_numpy(np.asarray([sample['audio_fea'] for sample in samples])).float(),
            'masks_tensor':masks_tensor,
            'category':categorys,
            'video_name': video_names,
        }


if __name__ == "__main__":
    train_dataset = S4Dataset('train')
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                     batch_size=2,
                                                     shuffle=False,
                                                     num_workers=8,
                                                     pin_memory=True)

    for n_iter, batch_data in enumerate(train_dataloader):
        imgs, audio, mask = batch_data # [bs, 5, 3, 224, 224], [bs, 5, 1, 96, 64], [bs, 1, 1, 224, 224]
        pdb.set_trace()
    print('n_iter', n_iter)
    pdb.set_trace()
