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
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import StepLR, MultiStepLR
import numpy as np
from configs.opts import parser
from model.main_model_2 import AV_VQVAE_Encoder, AT_VQVAE_Encoder, AV_VQVAE_Decoder, AT_VQVAE_Decoder, AVT_VQVAE_Encoder, AVT_VQVAE_Decoder
from model.CLUB import CLUBSample_group
from model.CPC import Cross_CPC, Cross_CPC_AVT
from utils import AverageMeter, Prepare_logger, get_and_save_args
from utils.container import metricsContainer
from utils.Recorder import Recorder
import torch.nn.functional as F
from bert_embedding import BertEmbedding
import pickle
# =================================  seed config ============================
SEED = 43
random.seed(SEED)
np.random.seed(seed=SEED)
torch.manual_seed(seed=SEED)
torch.cuda.manual_seed(seed=SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# =============================================================================
def transpose(x):
    return x.transpose(-2, -1)

def normalize(*xs):
    return [None if x is None else F.normalize(x, dim=-1) for x in xs]

def AVPSLoss(av_simm, soft_label):
    """audio-visual pair similarity loss for fully supervised setting,
    please refer to Eq.(8, 9) in our paper.
    """
    # av_simm: [bs, 10]
    relu_av_simm = F.relu(av_simm)
    sum_av_simm = torch.sum(relu_av_simm, dim=-1, keepdim=True)
    avg_av_simm = relu_av_simm / (sum_av_simm + 1e-8)
    loss = nn.MSELoss()(avg_av_simm, soft_label)
    return loss


bert_embedding = BertEmbedding()
with open('../../cnt.pkl', 'rb') as fp:
    id2idx = pickle.load(fp)
    
def collate_func_AT(samples):
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
    
        return {
            'query': query,
            'query_idx': query_idx,
            'query_len': query_len,
            'audio_fea': torch.from_numpy(np.asarray([sample['audio_fea'] for sample in samples])).float()
        }


def collate_func_AVT(samples):
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
            """
            There may be cases where the sample length is 0, 
            for example if your text happens to not be seen before in this BERT model. 
            If that happens, you can 
            1) clean the text before it enters BERT, 
            2) add an if statement here, 
            3) discard idx and directly import all embeddings after.
            """
            query1[i, :keep] = sample[:keep]
            query_idx[i, :keep] = query_words[i][:keep]
        query_len = np.asarray(query_len)
        query, query_len = torch.from_numpy(query1).float(), torch.from_numpy(query_len).long()
        query_idx = torch.from_numpy(query_idx).long()
    
        return {
            'query': query,
            'audio_fea': torch.from_numpy(np.asarray([sample['audio_fea'] for sample in samples])).float(),
            'video_fea': torch.from_numpy(np.asarray([sample['video_fea'] for sample in samples])).float()
        }


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
    os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
    # os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

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
    if args.dataset_name == 'ave':
        from dataset.AVE_dataset import AVEDataset as AVEDataset
    elif args.dataset_name =='vggsound':
        from dataset.VGGSOUND_dataset import VGGSoundDataset as AVEDataset 
    elif args.dataset_name =='vggsound_AT':
        from dataset.VGGSOUND_dataset import VGGSoundDataset_AT as AVEDataset
    elif args.dataset_name =='vggsound_AVT':
        from dataset.VGGSOUND_dataset import VGGSoundDataset_AVT as AVEDataset
    elif args.dataset_name =='vggsound179k' or args.dataset_name =='vggsound81k':
        from dataset.VGGSOUND_dataset179k import VGGSoundDataset as AVEDataset     
    else:
        raise NotImplementedError
    
    
    '''Dataloader selection'''
    if args.dataset_name == 'ave':
        data_root = 'data'
        train_dataloader = DataLoader(
            AVEDataset(data_root, split='train'),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=True
        )
        val_dataloader = DataLoader(
            AVEDataset(data_root, split='val'),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=8,
            pin_memory=True
        )
        test_dataloader = DataLoader(
            AVEDataset(data_root, split='test'),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=8,
            pin_memory=True
        )
    elif args.dataset_name == 'vggsound':
        meta_csv_path = 'vggsound-avel40k.csv'
        audio_fea_base_path = 'audio/zip'
        video_fea_base_path = 'video/zip'
        avc_label_base_path = 'label/zip'
        train_dataloader = DataLoader(
            AVEDataset(meta_csv_path, audio_fea_base_path, video_fea_base_path, avc_label_base_path, split='train'),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=True
        )
    elif args.dataset_name == 'vggsound_AT':
        meta_csv_path = 'vggsound-avel40k.csv'
        audio_fea_base_path = 'audio/zip'
        train_dataloader = DataLoader(
            AVEDataset(meta_csv_path, audio_fea_base_path, split='train'),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
            collate_fn=collate_func_AT
        )
    elif args.dataset_name == 'vggsound_AVT':
        meta_csv_path = 'vggsound-avel40k.csv'
        audio_fea_base_path = 'vggsound40k/feature/audio/zip'
        video_fea_base_path = 'vggsound40k/feature/video/zip'
        train_dataloader = DataLoader(
            AVEDataset(meta_csv_path, audio_fea_base_path, video_fea_base_path, split='train'),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=False,
            collate_fn=collate_func_AVT
        )
    elif args.dataset_name == 'vggsound81k':
        meta_csv_path = 'video_name_vggsound81k_checked.csv'
        audio_fea_base_path = 'audio/zip'
        video_fea_base_path = 'video/zip'
        avc_label_base_path = '...'
        train_dataloader = DataLoader(
            AVEDataset(meta_csv_path, audio_fea_base_path, video_fea_base_path, avc_label_base_path, split='train'),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=True
        )
    elif args.dataset_name == 'vggsound179k':
        meta_csv_path = 'video_name_vggsound179k_checked.csv'
        audio_fea_base_path = 'audio/zip'
        video_fea_base_path = 'video/zip'
        avc_label_base_path = '...'
        train_dataloader = DataLoader(
            AVEDataset(meta_csv_path, audio_fea_base_path, video_fea_base_path, avc_label_base_path, split='train'),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=True
        )
    else:
        raise NotImplementedError

    '''model setting'''
    video_dim = 512
    text_dim = 768
    audio_dim = 128
    text_lstm_dim = 128
    video_output_dim = 2048
    text_output_dim = 256
    audio_output_dim = 256
    n_embeddings = 400
    embedding_dim = 256
    start_epoch = -1
    model_resume = False
    total_step = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    Text_ar_lstm = nn.LSTM(text_dim, text_lstm_dim, num_layers=2, batch_first=True, bidirectional=True)
    
    if args.dataset_name == 'vggsound_AT':
        Encoder = AT_VQVAE_Encoder(text_lstm_dim*2, audio_dim, text_output_dim, audio_output_dim, n_embeddings, embedding_dim)
    if args.dataset_name == 'vggsound_AVT':
        Encoder = AVT_VQVAE_Encoder(audio_dim, video_dim, text_lstm_dim*2, audio_output_dim, video_output_dim, text_output_dim, n_embeddings, embedding_dim)
    else:
        Encoder = AV_VQVAE_Encoder(video_dim, audio_dim, video_output_dim, audio_output_dim, n_embeddings, embedding_dim)
    
    if args.dataset_name == 'vggsound_AVT':
        CPC = Cross_CPC_AVT(embedding_dim, hidden_dim=256, context_dim=256, num_layers=2)
    else:
        CPC = Cross_CPC(embedding_dim, hidden_dim=256, context_dim=256, num_layers=2)
    Video_mi_net = CLUBSample_group(x_dim=embedding_dim, y_dim=video_dim, hidden_size=256)
    Text_mi_net = CLUBSample_group(x_dim=embedding_dim, y_dim=text_output_dim, hidden_size=256)
    Audio_mi_net = CLUBSample_group(x_dim=embedding_dim, y_dim=audio_output_dim, hidden_size=256)
    
    if args.dataset_name == 'vggsound_AT':
        Decoder = AT_VQVAE_Decoder(text_lstm_dim*2, audio_dim, text_output_dim, audio_output_dim)
    if args.dataset_name == 'vggsound_AVT':
        Decoder = AVT_VQVAE_Decoder(audio_dim, video_dim, text_lstm_dim*2, audio_output_dim, video_output_dim, text_output_dim)
    else:
        Decoder = AV_VQVAE_Decoder(video_dim, audio_dim, video_output_dim, audio_output_dim)
        
    Text_ar_lstm.double()
    Encoder.double()
    CPC.double()
    Video_mi_net.double()
    Text_mi_net.double()
    Audio_mi_net.double()
    Decoder.double()
    
    '''optimizer setting'''
    Text_ar_lstm.to(device)
    Encoder.to(device)
    CPC.to(device)
    Video_mi_net.to(device)
    Text_mi_net.to(device)
    Audio_mi_net.to(device)
    Decoder.to(device)
    optimizer = torch.optim.Adam(chain(Text_ar_lstm.parameters(), \
                                       Encoder.parameters(), CPC.parameters(), Decoder.parameters()), lr=args.lr)
    optimizer_video_mi_net = torch.optim.Adam(Video_mi_net.parameters(), lr=args.mi_lr)
    optimizer_text_mi_net = torch.optim.Adam(Text_mi_net.parameters(), lr=args.mi_lr)
    optimizer_audio_mi_net = torch.optim.Adam(Audio_mi_net.parameters(), lr=args.mi_lr)
    scheduler = MultiStepLR(optimizer, milestones=[10, 20, 30], gamma=0.5)
    
    '''loss'''
    criterion = nn.BCEWithLogitsLoss().cuda()
    criterion_event = nn.CrossEntropyLoss().cuda()

    if model_resume is True:
        path_checkpoints = "..."
        checkpoints = torch.load(path_checkpoints)
        Encoder.load_state_dict(checkpoints['Encoder_parameters'])
        CPC.load_state_dict(checkpoints['CPC_parameters'])
        Video_mi_net.load_state_dict(checkpoints['Video_mi_net_parameters'])
        Audio_mi_net.load_state_dict(checkpoints['Audio_mi_net_parameters'])
        Decoder.load_state_dict(checkpoints['Decoder_parameters'])
        optimizer.load_state_dict(checkpoints['optimizer'])
        optimizer_audio_mi_net.load_state_dict(checkpoints['optimizer_audio_mi_net'])
        optimizer_video_mi_net.load_state_dict(checkpoints['optimizer_video_mi_net'])
        start_epoch = checkpoints['epoch']
        total_step = checkpoints['total_step']
        logger.info("Resume from number {}-th model.".format(start_epoch))

    '''Tensorboard and Code backup'''
    writer = SummaryWriter(args.snapshot_pref)
    recorder = Recorder(args.snapshot_pref, ignore_folder="Exps/")
    recorder.writeopt(args)

    '''Training and Evaluation'''

    for epoch in range(start_epoch+1, args.n_epoch):
        loss, total_step = train_epoch(CPC, Encoder,Text_ar_lstm, Audio_mi_net, Video_mi_net,Text_mi_net, Decoder, train_dataloader, criterion, criterion_event,
                                       optimizer, optimizer_audio_mi_net, optimizer_video_mi_net, optimizer_text_mi_net, epoch, total_step, args)
        
        save_path = os.path.join(args.model_save_path, 'your-model-{}.pt'.format(epoch))
        save_models(CPC, Encoder, Text_ar_lstm, Audio_mi_net, Video_mi_net, Text_mi_net, Decoder, optimizer, optimizer_audio_mi_net, optimizer_video_mi_net, optimizer_text_mi_net, epoch, total_step, save_path)
        logger.info(f"epoch: ******************************************* {epoch}")
        logger.info(f"loss: {loss}")
        scheduler.step()

def _export_log(epoch, total_step, batch_idx, lr, loss_meter):
    msg = 'Epoch {}, Batch {}, lr = {:.5f}, '.format(epoch, batch_idx, lr)
    for k, v in loss_meter.items():
        msg += '{} = {:.4f}, '.format(k, v)
    logger.info(msg)
    sys.stdout.flush()
    loss_meter.update({"batch": total_step})

def to_eval(all_models):
    for m in all_models:
        m.eval()


def to_train(all_models):
    for m in all_models:
        m.train()

def save_models(CPC, Encoder,Text_ar_lstm, Audio_mi_net, Video_mi_net, Text_mi_net, Decoder, optimizer, optimizer_audio_mi_net, optimizer_video_mi_net,optimizer_text_mi_net, epoch_num, total_step, path):
    state_dict = {
        'Encoder_parameters': Encoder.state_dict(),
        'CPC_parameters': CPC.state_dict(),
        'Text_ar_lstm_parameters': Text_ar_lstm.state_dict(),
        'Video_mi_net_parameters': Video_mi_net.state_dict(),
        'Text_mi_net_parameters': Text_mi_net.state_dict(),
        'Audio_mi_net_parameters': Audio_mi_net.state_dict(),
        'Decoder_parameters': Decoder.state_dict(),
        'optimizer': optimizer.state_dict(),
        'optimizer_video_mi_net': optimizer_video_mi_net.state_dict(),
        'optimizer_text_mi_net': optimizer_text_mi_net.state_dict(),
        'optimizer_audio_mi_net': optimizer_audio_mi_net.state_dict(),
        'epoch': epoch_num,
        'total_step': total_step
    }
    torch.save(state_dict, path)
    logging.info('save model to {}'.format(path))


def train_epoch_check(train_dataloader, epoch, total_step, args):
    train_dataloader = tqdm(train_dataloader)
    for n_iter, batch_data in enumerate(train_dataloader):
        
        '''Feed input to model'''
        visual_feature, audio_feature = batch_data
        visual_feature.cuda()
        audio_feature.cuda()
        
    return torch.zeros(1)

def train_epoch(CPC,Encoder,Text_ar_lstm, Audio_mi_net, Video_mi_net, Text_mi_net, Decoder,train_dataloader, criterion, criterion_event, optimizer, optimize_audio_mi_net, optimizer_video_mi_net, optimizer_text_mi_net, epoch, total_step, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    train_acc = AverageMeter()
    end_time = time.time()
    models = [CPC,Encoder,Text_ar_lstm, Audio_mi_net, Video_mi_net, Text_mi_net, Decoder]
    to_train(models)
    # Note: here we set the model to a double type precision,
    # since the extracted features are in a double type.
    # This will also lead to the size of the model double increases.

    Encoder.cuda()
    Text_ar_lstm.cuda()
    Text_mi_net.cuda()
    Audio_mi_net.cuda()
    Video_mi_net.cuda()
    Decoder.cuda()
    CPC.cuda()
    optimizer.zero_grad()
    mi_iters = 5

    # train_dataloader = tqdm(train_dataloader)

    for n_iter, batch_data in enumerate(train_dataloader):

        data_time.update(time.time() - end_time)
        '''Feed input to model'''
        
        # Adjust the input as needed according to the requirements of the model being trained.
        # vggsound_AVT
        query, audio_feature, video_feature = batch_data['query'], batch_data['audio_fea'], batch_data['video_fea']
        
        # vggsound_AV
        # visual_feature, audio_feature, labels = batch_data    # vggsound40k  
        # visual_feature, audio_feature = batch_data            # vggsound179k or vggsound81k
        query = query.double().cuda()
        audio_feature = audio_feature.to(torch.float64)
        batch_dim = query.size()[0]
        hidden_dim = 128
        num_layers = 2
        text_hidden = (torch.zeros(2*num_layers, batch_dim, hidden_dim).double().cuda(),
                  torch.zeros(2*num_layers, batch_dim, hidden_dim).double().cuda())
        text_feature, text_hidden = Text_ar_lstm(query, text_hidden)
        
        text_feature = text_feature.cuda().to(torch.float64)
        audio_feature = audio_feature.cuda().to(torch.float64)
        video_feature = video_feature.cuda().to(torch.float64)
        
        for i in range(mi_iters):
            optimizer_video_mi_net, lld_video_loss, optimizer_text_mi_net, lld_text_loss, optimize_audio_mi_net, lld_audio_loss = \
                mi_first_forward(audio_feature, video_feature, text_feature, Encoder, Audio_mi_net, Video_mi_net, Text_mi_net, optimize_audio_mi_net,optimizer_video_mi_net,optimizer_text_mi_net, epoch)

        audio_embedding_loss, video_embedding_loss,text_embedding_loss, mi_audio_loss, mi_video_loss, mi_text_loss, \
        accuracy1, accuracy2, accuracy3, accuracy4, accuracy5, accuracy6, accuracy7, accuracy8, accuracy9,\
        cpc_loss, audio_recon_loss, video_recon_loss, text_recon_loss, \
        audio_class, video_class, text_class, cmcm_loss, equal_num = mi_second_forward(CPC,audio_feature, video_feature, text_feature, Encoder,Audio_mi_net, Video_mi_net, Text_mi_net, Decoder,epoch)

        if n_iter % 20 == 0:
            logger.info("equal_num is {} in {}-th iteration.".format(equal_num, n_iter))

        loss_items = {
            "audio_recon_loss": audio_recon_loss.item(),
            "lld_audio_loss": lld_audio_loss.item(),
            "audio_embed_loss": audio_embedding_loss.item(),
            "audio_mine_loss": mi_audio_loss.item(),
            "text_recon_loss": text_recon_loss.item(),
            "lld_text_loss": lld_text_loss.item(),
            "text_embed_loss": text_embedding_loss.item(),
            "text_mine_loss": mi_text_loss.item(),
            "video_recon_loss": video_recon_loss.item(),
            "lld_video_loss": lld_video_loss.item(),
            "video_embed_loss": video_embedding_loss.item(),
            "video_mine_loss": mi_video_loss.item(),
            "acc_av": accuracy1.item(),
            "acc_at": accuracy2.item(),
            "acc_vt": accuracy3.item(),
            "acc_va": accuracy4.item(),
            "acc_ta": accuracy5.item(),
            "acc_tv": accuracy6.item(),
            "acc_aa": accuracy7.item(),
            "acc_vv": accuracy8.item(),
            "acc_tt": accuracy9.item(),
            "cpc_loss": cpc_loss.item(),
            "cmcm_loss": cmcm_loss.item()
        }

        metricsContainer.update("loss", loss_items)
        loss = audio_recon_loss + video_recon_loss + text_recon_loss + audio_embedding_loss +  video_embedding_loss\
                + text_embedding_loss+ mi_audio_loss + mi_video_loss + mi_text_loss + cpc_loss + cmcm_loss

        if n_iter % 20 == 0:
            _export_log(epoch=epoch, total_step=total_step+n_iter, batch_idx=n_iter, lr=0.0004, loss_meter=metricsContainer.calculate_average("loss"))
        
        loss.backward()

        '''Clip Gradient'''
        if args.clip_gradient is not None:
            for model in models:
                total_norm = clip_grad_norm_(model.parameters(), args.clip_gradient)

        '''Update parameters'''
        optimizer.step()
        optimizer.zero_grad()

        losses.update(loss.item(), text_feature.size(0) * 10)
        batch_time.update(time.time() - end_time)
        end_time = time.time()

        '''Add loss of a iteration in Tensorboard'''
        writer.add_scalar('Train_data/loss', losses.val, epoch * len(train_dataloader) + n_iter + 1)

        '''Add loss of an epoch in Tensorboard'''
        writer.add_scalar('Train_epoch_data/epoch_loss', losses.avg, epoch)

    return losses.avg, n_iter + total_step


def mi_first_forward(audio_feature, video_feature, text_feature, Encoder, Audio_mi_net, Video_mi_net,Text_mi_net, optimizer_audio_mi_net,optimizer_video_mi_net,optimizer_text_mi_net, epoch):

    optimizer_video_mi_net.zero_grad()
    optimizer_text_mi_net.zero_grad()
    optimizer_audio_mi_net.zero_grad()

    audio_semantic_result, video_semantic_result, text_semantic_result, \
    audio_encoder_result, video_encoder_result, video_club_feature, text_encoder_result, \
    audio_vq, video_vq, text_vq, audio_embedding_loss, video_embedding_loss, text_embedding_loss, cmcm_loss, equal_num\
    = Encoder(audio_feature, video_feature, text_feature, epoch)
               
    video_club_feature = video_club_feature.detach()
    text_encoder_result = text_encoder_result.detach()
    audio_encoder_result = audio_encoder_result.detach()
    video_vq = video_vq.detach()
    text_vq = text_vq.detach()
    audio_vq = audio_vq.detach()

    # video processing is different from audio and text modalities because video is more complex and feature extraction is more difficult.
    lld_video_loss = -Video_mi_net.loglikeli(video_vq, video_club_feature)
    lld_video_loss.backward()
    optimizer_video_mi_net.step()

    lld_text_loss = -Text_mi_net.loglikeli(text_vq, text_encoder_result)
    lld_text_loss.backward()
    optimizer_text_mi_net.step()

    lld_audio_loss = -Audio_mi_net.loglikeli(audio_vq, audio_encoder_result)
    lld_audio_loss.backward()
    optimizer_audio_mi_net.step()

    return optimizer_audio_mi_net, lld_audio_loss, optimizer_video_mi_net, lld_video_loss, optimizer_text_mi_net, lld_text_loss 


def VQ_audio_forward(audio_feature, visual_feature, Encoder, optimizer,epoch):

    audio_vq_forward_loss = Encoder.Audio_vq_forward(audio_feature, visual_feature,epoch)
    audio_vq_forward_loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return audio_vq_forward_loss, optimizer

def VQ_video_forward(audio_feature, visual_feature, Encoder, optimizer,epoch):
    optimizer.zero_grad()
    video_vq_forard_loss = Encoder.Video_vq_forward(audio_feature, visual_feature,epoch)
    video_vq_forard_loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return video_vq_forard_loss, optimizer

def mi_second_forward(CPC, audio_feature, video_feature, text_feature, Encoder, Audio_mi_net, Video_mi_net, Text_mi_net, Decoder,epoch):
    audio_semantic_result, video_semantic_result, text_semantic_result, \
    audio_encoder_result, video_encoder_result, video_club_feature, text_encoder_result, \
    audio_vq, video_vq, text_vq, audio_embedding_loss, video_embedding_loss, text_embedding_loss, cmcm_loss, equal_num \
    = Encoder(audio_feature, video_feature, text_feature, epoch)
    
    mi_video_loss = Video_mi_net.mi_est(video_vq, video_club_feature)
    mi_text_loss = Text_mi_net.mi_est(text_vq, text_encoder_result)
    mi_audio_loss = Audio_mi_net.mi_est(audio_vq, audio_encoder_result)
    
    """Cross_CPC"""
    accuracy1, accuracy2, accuracy3, accuracy4, \
    accuracy5, accuracy6, accuracy7, accuracy8, accuracy9, \
    cpc_loss = CPC(audio_semantic_result, video_semantic_result, text_semantic_result)

    audio_recon_loss, video_recon_loss, text_recon_loss, audio_class, video_class, text_class \
        = Decoder(audio_feature, video_feature, text_feature, audio_encoder_result, video_encoder_result, text_encoder_result, audio_vq, video_vq, text_vq)
    
    return audio_embedding_loss, video_embedding_loss, text_embedding_loss, mi_audio_loss, mi_video_loss, mi_text_loss, \
           accuracy1, accuracy2, accuracy3, accuracy4, accuracy5, accuracy6, accuracy7, accuracy8, accuracy9, cpc_loss,  \
           audio_recon_loss, video_recon_loss, text_recon_loss, audio_class, video_class, text_class, cmcm_loss, equal_num, zero_num, same_layer_fusion_loss, adjacent_layer_separation_loss


def compute_accuracy_supervised(event_scores, labels):
    labels_foreground = labels[:, :, :-1]
    labels_BCE, labels_evn = labels_foreground.max(-1)
    labels_event, _ = labels_evn.max(-1)
    _, event_class = event_scores.max(-1)
    correct = event_class.eq(labels_event)
    correct_num = correct.sum().double()
    acc = correct_num * (100. / correct.numel())
    return acc

def save_checkpoint(state_dict, top1, task, epoch):
    model_name = f'{args.snapshot_pref}/model_epoch_{epoch}_top1_{top1:.3f}_task_{task}_best_model.pth.tar'
    torch.save(state_dict, model_name)
    

if __name__ == '__main__':
    main()
