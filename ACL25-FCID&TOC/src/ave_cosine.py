import logging
import os
import time
import random
import json
from tqdm import tqdm
import sys
import torch
from itertools import chain
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
# from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import StepLR, MultiStepLR
import numpy as np
from configs.opts import parser
from model.main_model_2 import Semantic_Decoder,AVT_VQVAE_Encoder
from utils import AverageMeter, Prepare_logger, get_and_save_args
from utils.container import metricsContainer
from utils.Recorder import Recorder
import torch.nn.functional as F
# from sklearn.cluster import KMeans

# =================================  seed config ============================
SEED = 43
random.seed(SEED)
np.random.seed(seed=SEED)
torch.manual_seed(seed=SEED)
torch.cuda.manual_seed(seed=SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# =============================================================================
def main():
    # utils variable
    global args, logger, writer, dataset_configs
    # statistics variable
    global best_accuracy, best_accuracy_epoch
    best_accuracy, best_accuracy_epoch = 0, 0
    # configs
    dataset_configs = get_and_save_args(parser)
    parser.set_defaults(**dataset_configs)
    args = parser.parse_args()
    # select GPUs
    # os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    '''Create snapshot_pred dir for copying code and saving models '''
    if not os.path.exists(args.snapshot_pref):
        os.makedirs(args.snapshot_pref)

    if os.path.isfile(args.resume):
        args.snapshot_pref = os.path.dirname(args.resume)

    logger = Prepare_logger(args, eval=args.evaluate)

    if not args.evaluate:
        logger.info(f'\nCreating folder: {args.snapshot_pref}')
        logger.info('\nRuntime args\n\n{}\n'.format(json.dumps(vars(args), indent=4)))
    else:
        logger.info(f'\nLog file will be save in a {args.snapshot_pref}/Eval.log.')

    '''dataset selection'''
    if args.dataset_name == 'ave_va' or args.dataset_name == 'ave_av':
        from dataset.AVE_dataset import AVEDataset as AVEDataset
    else: 
        raise NotImplementedError
    
  
    '''Dataloader selection'''
    data_root = '/../../data/AVE-ECCV18-master/data'
    train_dataloader = DataLoader(
        AVEDataset(data_root, split='train'),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=False
    )

    '''model setting'''
    video_dim = 512
    audio_dim = 128
    video_output_dim = 2048
    audio_output_dim = 256
    text_lstm_dim = 128
    text_output_dim = 256
    n_embeddings = 400
    embedding_dim = 256
    start_epoch = -1
    model_resume = True
    total_step = 0



    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # AV
    # Encoder = AV_VQVAE_Encoder(video_dim, audio_dim, video_output_dim, audio_output_dim, n_embeddings, embedding_dim)
    
    # AVT
    Encoder = AVT_VQVAE_Encoder(audio_dim, video_dim, text_lstm_dim*2, audio_output_dim, video_output_dim, text_output_dim, n_embeddings, embedding_dim)

    choose_channel = args.choose_channel
    print('--------------choose_channel:',choose_channel)
    Decoder = Semantic_Decoder(input_dim=choose_channel, class_num=28) 

    
    Encoder.double()
    Decoder.double()
    Encoder.to(device)
    Decoder.to(device)
    optimizer = torch.optim.Adam(chain(Encoder.parameters(), Decoder.parameters()), lr=args.lr)

    scheduler = MultiStepLR(optimizer, milestones=[10, 20, 30], gamma=0.5)
    
    '''loss'''
    criterion = nn.BCEWithLogitsLoss().cuda()
    criterion_event = nn.CrossEntropyLoss().cuda()

    if model_resume is True:
        # path_checkpoints = "/../../data/checkpoints/unicode2/cpc-pro31-AVT2-400-model-3.pt"
        path_checkpoints = "/../../data/checkpoints/nips2023_AVT_vgg40k_size400.pt"
        # path_checkpoints = "/../../data/checkpoints/rqvae/VQ-AVT-400-model-3.pt"
        print(path_checkpoints)
        checkpoints = torch.load(path_checkpoints)
        # for key,value in checkpoints.items():
        #     print(key)
        Encoder.load_state_dict(checkpoints['Encoder_parameters'])
        
        # print(Encoder)
        
        start_epoch = checkpoints['epoch']
        logger.info("Resume from number {}-th model.".format(start_epoch))

    # '''Tensorboard and Code backup'''
    # writer = SummaryWriter(args.snapshot_pref)
    # recorder = Recorder(args.snapshot_pref, ignore_folder="Exps/")
    # recorder.writeopt(args)

    '''Training and Evaluation'''

    """选择部分channel"""
    indices = cal_criterion(Encoder.Cross_quantizer.embedding.cuda(), choose_channel, args.toc_max_num, args.toc_min_num)

    """cosine"""
    vectors_before = Encoder.Cross_quantizer.embedding.cpu()
    vectors_after = Encoder.Cross_quantizer.embedding[:,indices].cpu()

    import seaborn as sns
    import matplotlib.pyplot as plt
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mtick
    from matplotlib.backends.backend_pdf import PdfPages

    # Compute the cosine similarity matrix
    cosine_sim_matrix_before = torch.nn.functional.cosine_similarity(vectors_before[:, None, :], vectors_before[None, :, :], dim=-1)
    cosine_sim_matrix_after = torch.nn.functional.cosine_similarity(vectors_after[:, None, :], vectors_after[None, :, :], dim=-1)

    # # Plot the cosine similarity matrix using a heatmap
    # plt.figure(figsize=(10, 8))
    # sns.heatmap(cosine_sim_matrix_before, cmap='viridis')
    # plt.title('Before Code Select')
    # plt.savefig('Cosine_before.png')

    

    # # Plot the cosine similarity matrix using a heatmap
    # plt.figure(figsize=(10, 8))
    # sns.heatmap(cosine_sim_matrix_after, cmap='viridis')
    # plt.title('After Code Select')
    # plt.savefig('Cosine_after.png')

    """1"""
    # Plot the cosine similarity matrices using heatmaps in a single figure with reduced margins
    # fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    # sns.heatmap(cosine_sim_matrix_before, cmap='viridis', ax=axes[0], cbar_kws={'shrink': 0.8})
    # axes[0].set_title('Before Code Select')
    # sns.heatmap(cosine_sim_matrix_after, cmap='viridis', ax=axes[1], cbar_kws={'shrink': 0.8})
    # axes[1].set_title('After Code Select')

    # # Adjust layout and save the figure
    # plt.tight_layout()
    # plt.savefig('Cosine_before_after.png')
    # plt.show()

    """2"""
    with PdfPages('TOC_before_after_new.png') as pp:
        # Plot the cosine similarity matrices using heatmaps in a single figure with reduced margins and custom tick labels
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        plt.rcParams.update({'font.size': 20})
        # plt.xticks(fontsize=16)
        # plt.yticks(fontsize=16)
        sns.heatmap(cosine_sim_matrix_before, cmap='viridis', ax=axes[0], cbar_kws={'shrink': 0.8})
        

        axes[0].set_title('Before TOC Select')
        axes[0].set_xticks([0, 100, 200, 300, 400])
        axes[0].set_yticks([0, 100, 200, 300, 400])
        axes[0].set_xticklabels(['0', '100', '200', '300', '400'],fontsize=16)
        axes[0].set_yticklabels(['0', '100', '200', '300', '400'],fontsize=16)

        sns.heatmap(cosine_sim_matrix_after, cmap='viridis', ax=axes[1], cbar_kws={'shrink': 0.8})
        # plt.xticks(fontsize=16)
        # plt.yticks(fontsize=16)
        axes[1].set_title('After TOC Select')
        axes[1].set_xticks([0, 100, 200, 300, 400])
        axes[1].set_yticks([0, 100, 200, 300, 400])
        axes[1].set_xticklabels(['0', '100', '200', '300', '400'],fontsize=16)
        axes[1].set_yticklabels(['0', '100', '200', '300', '400'],fontsize=16)

        # Adjust layout and save the figure
        plt.tight_layout()
        # plt.savefig('TOC_before_after.png')
        plt.savefig(pp, format='png', bbox_inches='tight')

    print('Donw!')

def cal_criterion(feats, choose_channel, max_num, min_num):
    code_num, code_dim = feats.shape
    
    sim_sum = torch.zeros((code_dim)).cuda()
    count = 0
    for i in range(code_num):
        for j in range(code_num):
            if i != j:
                sim_sum += feats[i, :] * feats[j, :]
                count += 1
    sim = sim_sum / count
    
    criterion = (-0.7) * sim + 0.3 * torch.var(feats, dim=0)

    _, max_indices = torch.topk(criterion, k=choose_channel//int(max_num+min_num)*int(max_num))
    print(max_indices)
    _, min_indices = torch.topk(criterion, k=choose_channel//int(max_num+min_num)*int(min_num), largest=False)
    print(min_indices)
    indices = torch.cat((max_indices, min_indices),dim=0)
    # print(indices)
    return indices


if __name__ == '__main__':
    main()