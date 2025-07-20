#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author  : Xinhao Mei @CVSSP, University of Surrey
# @E-mail  : x.mei@surrey.ac.uk

"""
Evaluation tools adapted from https://github.com/fartashf/vsepp/blob/master/evaluation.py
"""

import numpy as np
import torch
import random
from sentence_transformers import util
# from loguru import logger

def l2norm(X):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=1, keepdim=True).sqrt()
    X = torch.div(X, norm)
    return X


# # evaluation tools
# n = 5
# def a2t(audio_embs, cap_embs, num_audios, audio_cap_num, return_ranks=False):
#     # audio to caption retrieval
#     index_list = []

#     ranks = np.zeros(num_audios)
#     top1 = np.zeros(num_audios)
#     mAP10 = np.zeros(num_audios)
#     for index in range(num_audios):
#         # get query audio
#         audio = audio_embs[n * index].reshape(1, audio_embs.shape[1])

#         # compute scores
#         d = util.cos_sim(torch.Tensor(audio), torch.Tensor(cap_embs)).squeeze(0).numpy()
#         inds = np.argsort(d)[::-1]
#         index_list.append(inds[0])

#         inds_map = []

#         rank = 1e20
#         for i in range(n * index, n * index + n, 1):
#             tmp = np.where(inds == i)[0][0]
#             if tmp < rank:
#                 rank = tmp
#             if tmp < 10:
#                 inds_map.append(tmp + 1)
#         inds_map = np.sort(np.array(inds_map))
#         if len(inds_map) != 0:
#             mAP10[index] = np.sum((np.arange(1, len(inds_map) + 1) / inds_map)) / n
#         else:
#             mAP10[index] = 0.
#         ranks[index] = rank
#         top1[index] = inds[0]
#     # compute metrics
#     r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
#     r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
#     r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
#     r50 = 100.0 * len(np.where(ranks < 50)[0]) / len(ranks)
#     mAP10 = 100.0 * np.sum(mAP10) / len(ranks)
#     medr = np.floor(np.median(ranks)) + 1
#     meanr = ranks.mean() + 1
#     if return_ranks:
#         return r1, r5, r10, r50, medr, meanr, ranks, top1
#     else:
#         return r1, r5, r10, r50, medr, meanr

# def t2a(audio_embs, cap_embs, audio_cap_num, return_ranks=False):
#     # caption to audio retrieval
#     num_audios = int(audio_embs.shape[0] / n)

#     audios = np.array([audio_embs[i]for i in range(0, audio_embs.shape[0], n)])

#     ranks = np.zeros(n * num_audios)
#     top1 = np.zeros(n * num_audios)

#     for index in range(num_audios):

#         # get query captions
#         queries = cap_embs[n * index: n * index + n]

#         # compute scores
#         d = util.cos_sim(torch.Tensor(queries), torch.Tensor(audios)).numpy()

#         inds = np.zeros(d.shape)
#         for i in range(len(inds)):
#             inds[i] = np.argsort(d[i])[::-1]
#             ranks[n * index + i] = np.where(inds[i] == index)[0][0]
#             top1[n * index + i] = inds[i][0]

#     # compute metrics
#     r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
#     r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
#     r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
#     r50 = 100.0 * len(np.where(ranks < 50)[0]) / len(ranks)
#     mAP10 = 100.0 * np.sum(1 / (ranks[np.where(ranks < 10)[0]] + 1)) / len(ranks)
#     medr = np.floor(np.median(ranks)) + 1
#     meanr = ranks.mean() + 1
#     if return_ranks:
#         return r1, r5, r10, r50, medr, meanr, ranks, top1
#     else:
#         return r1, r5, r10, r50, medr, meanr
    

# def a2t(audio_embs, cap_embs, audio_cap_num, return_ranks=False):
#     num_audios = len(audio_cap_num)
#     ranks = np.zeros(num_audios)
#     top1 = np.zeros(num_audios)
#     mAP10 = np.zeros(num_audios)
#     start_idx = 0

#     for index in range(num_audios):
#         # get query audio
#         # audio = audio_embs[index].reshape(1, -1)
#         audio = audio_embs[start_idx].reshape(1, -1)

#         # determine the number of captions for the current audio
#         num_caps = audio_cap_num[index]

#         # compute scores
#         cap_range = cap_embs[start_idx:start_idx+num_caps]
#         scores = util.cos_sim(torch.Tensor(audio), torch.Tensor(cap_range)).squeeze(0).numpy()

#         inds = np.argsort(scores)[::-1]
#         top1[index] = inds[0]
#         rank = np.where(inds == 0)[0][0]
#         ranks[index] = rank

#         # compute mAP for top 10 ranks
#         map_inds = inds[:10]  # Taking top 10 indices
#         map_ranks = 1.0 / (map_inds + 1)
#         mAP10[index] = np.mean(map_ranks) if len(map_ranks) > 0 else 0.0

#         # Update start index for the next audio
#         start_idx += num_caps

#     # compute metrics
#     r1 = 100.0 * np.mean(ranks < 1)
#     r5 = 100.0 * np.mean(ranks < 5)
#     r10 = 100.0 * np.mean(ranks < 10)
#     r50 = 100.0 * np.mean(ranks < 50)
#     mean_map10 = 100.0 * np.mean(mAP10)
#     medr = np.median(ranks) + 1
#     meanr = np.mean(ranks) + 1

#     if return_ranks:
#         return r1, r5, r10, r50, medr, meanr, ranks, top1
#     else:
#         return r1, r5, r10, r50, medr, meanr
    
def a2t(audio_embs, cap_embs, audio_cap_num, return_ranks=False):
    num_audios = len(audio_cap_num)
    ranks = np.zeros(num_audios)
    top1 = np.zeros(num_audios)
    mAP10 = np.zeros(num_audios)
    start_idx = 0

    for index in range(num_audios):
        # Get the current audio embedding
        audio = audio_embs[start_idx].reshape(1, -1)

        # Compute cosine similarity scores with all captions
        scores = util.cos_sim(torch.Tensor(audio), torch.Tensor(cap_embs)).squeeze(0).numpy()

        # Sorting the indices based on similarity scores
        inds = np.argsort(scores)[::-1]
        top1[index] = inds[0]

        # Find the best rank of correct caption for this audio
        best_rank = float('inf')
        for i in range(audio_cap_num[index]):
            rank = np.where(inds == start_idx + i)[0][0]
            if rank < best_rank:
                best_rank = rank
        ranks[index] = best_rank

        # Computing mAP for the top 10 ranks
        relevant_inds = inds[:10]
        relevant_ranks = relevant_inds < start_idx + audio_cap_num[index]
        relevant_ranks = relevant_ranks.nonzero()[0] + 1  # 1-based indexing
        mAP10[index] = np.mean(1 / relevant_ranks) if len(relevant_ranks) > 0 else 0

        # Update start index for the captions
        start_idx += audio_cap_num[index]

    # Compute metrics
    r1 = 100.0 * np.mean(ranks < 1)
    r5 = 100.0 * np.mean(ranks < 5)
    r10 = 100.0 * np.mean(ranks < 10)
    r50 = 100.0 * np.mean(ranks < 50)
    mean_map10 = 100.0 * np.mean(mAP10)
    medr = np.median(ranks) + 1
    meanr = np.mean(ranks) + 1

    if return_ranks:
        return r1, r5, r10, r50, medr, meanr, ranks, top1
    else:
        return r1, r5, r10, r50, medr, meanr

    
def t2a(audio_embs, cap_embs, audio_cap_num, return_ranks=False):
    num_audios = len(audio_cap_num)
    # audios = audio_embs.reshape(num_audios, -1)
    audios = []
    index = 0
    for i in range(num_audios):
        audios.append(audio_embs[index])
        index += audio_cap_num[i] 
    ranks = []
    top1 = []
    start_idx = 0

    for index in range(num_audios):
        # get query captions
        num_caps = audio_cap_num[index]
        queries = cap_embs[start_idx:start_idx+num_caps]

        # compute scores
        scores = util.cos_sim(torch.Tensor(queries), torch.Tensor(audios)).numpy()

        for i in range(num_caps):
            inds = np.argsort(scores[i])[::-1]
            rank = np.where(inds == index)[0][0]
            ranks.append(rank)
            top1.append(inds[0])

        # Update start index for the next audio
        start_idx += num_caps

    # compute metrics
    ranks = np.array(ranks)
    r1 = 100.0 * np.mean(ranks < 1)
    r5 = 100.0 * np.mean(ranks < 5)
    r10 = 100.0 * np.mean(ranks < 10)
    r50 = 100.0 * np.mean(ranks < 50)
    mean_map10 = 100.0 * np.mean(1 / (ranks + 1))
    medr = np.median(ranks) + 1
    meanr = np.mean(ranks) + 1

    if return_ranks:
        return r1, r5, r10, r50, medr, meanr, ranks, top1
    else:
        return r1, r5, r10, r50, medr, meanr
