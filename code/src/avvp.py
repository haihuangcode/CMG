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
from model.main_model_2 import AV_VQVAE_Encoder
from model.main_model_2 import AV_VQVAE_Decoder
from model.main_model_2 import Semantic_Decoder_AVVP, AVT_VQVAE_Encoder

from utils import AverageMeter, Prepare_logger, get_and_save_args
from utils.container import metricsContainer
from utils.Recorder import Recorder

import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from utils.draw import Draw_Heatmap


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
    global best_acc,best_rec,best_f1
    best_acc=0
    best_rec=0
    best_f1=0
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
    if args.dataset_name =='avvp_av' or args.dataset_name =='avvp_va':
        from dataset.AVVP_dataset import AVVPDataset
    else: 
        raise NotImplementedError

    '''Dataloader selection'''
    if args.dataset_name == 'avvp_va':
        train_csv_path = 'AVVP/data/AVVP_eval_visual_checked_combined.csv'
        val_csv_path = 'AVVP/data/AVVP_eval_audio_checked_combined.csv'
        audio_fea_base_path = 'AVVP/feature/audio/zip'
        video_fea_base_path = 'AVVP/feature/video/zip'
        train_dataloader = DataLoader(
            AVVPDataset(train_csv_path, video_fea_base_path, split='train', modality='video'),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=False
        )
        val_dataloader = DataLoader(
            AVVPDataset(val_csv_path, audio_fea_base_path, split='val', modality='audio'),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=8,
            pin_memory=False
        ) 
    elif args.dataset_name == 'avvp_av':
        train_csv_path = 'AVVP/data/AVVP_eval_audio_checked_combined.csv'
        val_csv_path = 'AVVP/data/AVVP_eval_visual_checked_combined.csv'
        audio_fea_base_path = 'AVVP/feature/audio/zip'
        video_fea_base_path = 'AVVP/feature/video/zip'
        train_dataloader = DataLoader(
            AVVPDataset(train_csv_path, audio_fea_base_path, split='train', modality='audio'),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=False
        )
        val_dataloader = DataLoader(
            AVVPDataset(val_csv_path, video_fea_base_path, split='val', modality='video'),
            batch_size=args.batch_size,
            shuffle=False,
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
    
    Decoder = Semantic_Decoder_AVVP(input_dim=embedding_dim,class_num=26) #256对应embedding_dim，即codeword的维度
    Encoder.double()
    Decoder.double()
    Encoder.to(device)
    Decoder.to(device)
    ExpLogLoss_fn = ExpLogLoss(alpha=0.1)
    optimizer = torch.optim.Adam(chain(Encoder.parameters(), Decoder.parameters(),ExpLogLoss_fn.alpha), lr=args.lr)

    scheduler = MultiStepLR(optimizer, milestones=[10, 20, 30], gamma=0.5)
    
    '''loss'''
    criterion = nn.BCEWithLogitsLoss().cuda()
    criterion_event = nn.CrossEntropyLoss().cuda()


    if model_resume is True:
        path_checkpoints = "..."
        checkpoints = torch.load(path_checkpoints)
        Encoder.load_state_dict(checkpoints['Encoder_parameters'])
        start_epoch = checkpoints['epoch']
        logger.info("Resume from number {}-th model.".format(start_epoch))

    '''Tensorboard and Code backup'''
    writer = SummaryWriter(args.snapshot_pref)
    recorder = Recorder(args.snapshot_pref, ignore_folder="Exps/")
    recorder.writeopt(args)

    '''Training and Evaluation'''

    for epoch in range(start_epoch+1, args.n_epoch):
        
        loss, total_step = train_epoch(Encoder, Decoder,ExpLogLoss_fn, train_dataloader, criterion, criterion_event,
                                       optimizer, epoch, total_step, args)
        logger.info(f"epoch: *******************************************{epoch}")

        if ((epoch + 1) % args.eval_freq == 0) or (epoch == args.n_epoch - 1):
            loss = validate_epoch(Encoder, Decoder, ExpLogLoss_fn, val_dataloader, criterion, criterion_event, epoch, args)
            logger.info("-----------------------------")
            logger.info(f"evaluate loss:{loss}")
            logger.info("-----------------------------")
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

def save_models(Encoder, optimizer, epoch_num, total_step, path):
    state_dict = {
        'Encoder_parameters': Encoder.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch_num,
        'total_step': total_step,
    }
    torch.save(state_dict, path)
    logging.info('save model to {}'.format(path))


def train_epoch_check(train_dataloader, epoch, total_step, args):
    # train_dataloader = tqdm(train_dataloader)
    for n_iter, batch_data in enumerate(train_dataloader):
        
        '''Feed input to model'''
        feature, labels, mask = batch_data['feature'],batch_data['label'],batch_data['mask']
    return torch.zeros(1),torch.zeros(1)


def train_epoch(Encoder, Decoder,ExpLogLoss_fn, train_dataloader, criterion, criterion_event, optimizer, epoch, total_step, args):
    Sigmoid_fun = nn.Sigmoid()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    train_acc = AverageMeter()
    all_accuracy = AverageMeter()
    all_recall = AverageMeter()
    end_time = time.time()
    models = [Encoder, Decoder]
    to_train(models)
    # Note: here we set the model to a double type precision,
    # since the extracted features are in a double type.
    # This will also lead to the size of the model double increases.

    Encoder.cuda()
    Decoder.cuda()
    optimizer.zero_grad()

    for n_iter, batch_data in enumerate(train_dataloader):

        data_time.update(time.time() - end_time)
        '''Feed input to model'''
        feature, labels, video_id = batch_data
        feature = feature.to(torch.float64)
        feature.cuda()

        labels = labels.double().cuda()
        """"now"""
        B, T, C = labels.size()
        labels = labels.reshape(-1, C)# [B, T, C] -> [BxT, C] e.g.[80*10, 26]
        labels_evn = labels.to(torch.float32)
        bs = feature.size(0)

        if (args.dataset_name == 'avvp_va'):
            
            with torch.no_grad():
                video_vq = Encoder.Video_VQ_Encoder(feature)
            e_dim = video_vq.size()[2]
            video_vq = video_vq.reshape(-1, e_dim)
            video_class = Decoder(video_vq)
            loss1 = criterion(video_class, labels_evn.cuda())
            loss2 = ExpLogLoss_fn(video_class, labels_evn.cuda())
            # loss3 = distance_map_loss(Sigmoid_fun(video_class), labels_evn.cuda())+ loss3
            video_event_loss = loss1 + loss2 
            
            precision, recall = compute_accuracy_supervised_sigmoid(Sigmoid_fun(video_class), labels_evn.cuda())
            loss_items = {
                "video_event_loss":video_event_loss.item(),
                "BCELoss":loss1.item(),
                "ExpLogLoss":loss2.item(),
                # "DistanceMapLoss":loss3.item(),
                "video_precision": precision.item(),
                "video_recall": recall.item(),
                "alpha":ExpLogLoss_fn.check().item()
            }
            all_accuracy.update(precision.item(), bs * 10)
            all_recall.update(recall.item(), bs * 10)
            metricsContainer.update("loss", loss_items)
            loss = video_event_loss
        elif (args.dataset_name == 'avvp_av'):
            with torch.no_grad():
                audio_vq = Encoder.Audio_VQ_Encoder(feature)
            e_dim = audio_vq.size()[2]
            audio_vq = audio_vq.reshape(-1, e_dim)
            audio_class = Decoder(audio_vq)
            loss1 = criterion(audio_class, labels_evn.cuda())
            loss2 = ExpLogLoss_fn(audio_class, labels_evn.cuda())
            # loss3 = distance_map_loss(Sigmoid_fun(video_class), labels_evn.cuda())
            audio_event_loss = loss1 + loss2
            precision, recall = compute_accuracy_supervised_sigmoid(Sigmoid_fun(audio_class), labels_evn.cuda())
            loss_items = {
                "audio_event_loss":audio_event_loss.item(),
                "BCELoss":loss1.item(),
                "ExpLogLoss":loss2.item(),
                # "DistanceMapLoss":loss3.item(),
                "audio_precision": precision.item(),
                "audio_recall": recall.item()
            }
            all_accuracy.update(precision.item(), bs * 10)
            all_recall.update(recall.item(), bs * 10)
            metricsContainer.update("loss", loss_items)
            loss = audio_event_loss
        else: 
            raise NotImplementedError

        if n_iter % 10 == 0:
            _export_log(epoch=epoch, total_step=total_step+n_iter, batch_idx=n_iter, lr=optimizer.state_dict()['param_groups'][0]['lr'], loss_meter=metricsContainer.calculate_average("loss"))
        loss.backward()
        

        '''Clip Gradient'''
        if args.clip_gradient is not None:
            for model in models:
                total_norm = clip_grad_norm_(model.parameters(), args.clip_gradient)

        '''Update parameters'''
        optimizer.step()
        optimizer.zero_grad()

        losses.update(loss.item(), feature.size(0) * 10)
        batch_time.update(time.time() - end_time)
        end_time = time.time()

        '''Add loss of a iteration in Tensorboard'''
        writer.add_scalar('Train_data/loss', losses.val, epoch * len(train_dataloader) + n_iter + 1)

        '''Add loss of an epoch in Tensorboard'''
        writer.add_scalar('Train_epoch_data/epoch_loss', losses.avg, epoch)


    logger.info(
        f'**********************************************\t'
        f"\t Train results (accuracy and recall): {all_accuracy.avg:.4f}\t{all_recall.avg:.4f}."
    )
    return losses.avg, n_iter + total_step


@torch.no_grad()
def validate_epoch(Encoder,Decoder,ExpLogLoss_fn, val_dataloader, criterion, criterion_event, epoch, args, eval_only=False):
    Sigmoid_fun = nn.Sigmoid()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    downstream_accuracy = AverageMeter()
    downstream_recall = AverageMeter()
    end_time = time.time()

    Encoder.eval()
    Decoder.eval()
    Encoder.cuda()
    Decoder.cuda()
    
    Draw = Draw_Heatmap()
    Draw.eval().cuda()


    for n_iter, batch_data in enumerate(val_dataloader):
        data_time.update(time.time() - end_time)

        '''Feed input to model'''
        feature, labels, video_id = batch_data
        feature = feature.to(torch.float64)
        feature.cuda()
        labels = labels.double().cuda()
        
        B, T, C = labels.size()
        labels = labels.reshape(-1, C)# [B, T, C] -> [BxT, C] e.g.[80*10, 26]
        labels_evn = labels.to(torch.float32)

        bs = feature.size(0)
        
        if (args.dataset_name == 'avvp_va'):
            audio_vq = Encoder.Audio_VQ_Encoder(feature)
            B, T, e_dim = audio_vq.size()
            audio_vq = audio_vq.reshape(-1, e_dim)
            audio_class = Decoder(audio_vq)
            loss1 = criterion(audio_class, labels_evn.cuda())
            loss2 = ExpLogLoss_fn(audio_class, labels_evn.cuda())
            # loss3 = distance_map_loss(Sigmoid_fun(audio_class), labels_evn.cuda())+ loss3
            audio_event_loss = loss1 + loss2 
            
            loss = audio_event_loss
            
            precision, recall = compute_accuracy_supervised_sigmoid(Sigmoid_fun(audio_class), labels_evn.cuda())
            """draw image"""
            # if n_iter % 5 == 0:
            #     rand_choose = random.randint(0,B-1)
            #     save_img(Draw, Sigmoid_fun(audio_class.reshape(B,T,C)[rand_choose,:,:]), labels_evn.reshape(B,T,C)[rand_choose,:,:].cuda(), video_id[rand_choose],"va",epoch)
            downstream_accuracy.update(precision.item(), bs * 10)
            downstream_recall.update(recall.item(), bs * 10)
        elif (args.dataset_name == 'avvp_av'):
            video_vq = Encoder.Video_VQ_Encoder(feature)
            e_dim = video_vq.size()[2]
            video_vq = video_vq.reshape(-1, e_dim)
            video_class = Decoder(video_vq)
            loss1 = criterion(video_class, labels_evn.cuda())
            loss2 = ExpLogLoss_fn(video_class, labels_evn.cuda())
            # loss3 = distance_map_loss(Sigmoid_fun(video_class), labels_evn.cuda())
            video_event_loss = loss1 + loss2
            loss = video_event_loss
            precision, recall = compute_accuracy_supervised_sigmoid(Sigmoid_fun(video_class), labels_evn.cuda())
            """draw image"""
            # if n_iter % 5 == 0:
            #     rand_choose = random.randint(0,B-1)
            #     save_img(Draw, Sigmoid_fun(video_class.reshape(B,T,C)[rand_choose,:,:]), labels_evn.reshape(B,T,C)[rand_choose,:,:].cuda(), video_id[rand_choose],"av",epoch)
            downstream_accuracy.update(precision.item(), bs * 10)
            downstream_recall.update(recall.item(), bs * 10)
        else: 
            raise NotImplementedError
            
        batch_time.update(time.time() - end_time)
        end_time = time.time()
        losses.update(loss.item(), bs * 10)

    f1_score = 2.0*downstream_accuracy.avg*downstream_recall.avg/(downstream_accuracy.avg+downstream_recall.avg)
    global best_f1,best_acc,best_rec
    # For AVVP downstream, record the best acc. For AVE_AVVP, record the best f1-score. 
    # This setting is simple, there is no special deeper meaning.
    if downstream_accuracy.avg > best_acc:
        best_f1 = f1_score
        best_acc = downstream_accuracy.avg
        best_rec = downstream_recall.avg
    logger.info(
        f'**********************************************\t'
        f"\t Evaluation results (accuracy and recall F1-score): {downstream_accuracy.avg:.4f}\t{downstream_recall.avg:.4f}\t{f1_score:.4f}.\n"
        f'**********************************************\t'
        f"\t Best results (accuracy and recall F1-score): {best_acc:.4f}\t{best_rec:.4f}\t{best_f1:.4f}."
    )
    return losses.avg

def compute_accuracy_supervised_sigmoid(model_pred, labels):
    accuracy_th = 0.5
    pred_result = model_pred > accuracy_th
    pred_result = pred_result.float()
    pred_one_num = torch.sum(pred_result)
    if pred_one_num == 0:
        return torch.zeros(1), torch.zeros(1)
    target_one_num = torch.sum(labels)
    true_predict_num = torch.sum(pred_result * labels)
    
    precision = true_predict_num / pred_one_num
    
    recall = true_predict_num / target_one_num
    return precision, recall

# Draw three graphs: the first graph is the predicted values, the second graph is the ground truth,
# and the third graph is the intersection of the two.
def save_img(Draw, model_pred, labels, video_id, modality,epoch):
    accuracy_th = 0.5
    pred_result = model_pred > accuracy_th
    labels = labels.int()
    Draw(model_pred, labels, pred_result * labels, video_id, modality,epoch)

def distance_map_loss(y_pred, y_true):
    y_true_dist = torch.square(y_true)
    y_pred_dist = torch.square(y_pred)
    diff = torch.sqrt(torch.abs(y_true_dist - y_pred_dist))
    y_true_clone = y_true.clone() + 1
    
    for i in range(y_true_clone.size()[0]):
        for j in range(y_true_clone.size()[1]):
            if y_true_clone[i][j]==2:
                y_true_clone[i][j] = 10
                
    loss = torch.mean(diff * y_true_clone)
    return loss

# Exponential Logarithmic Loss
class ExpLogLoss(nn.Module):
    def __init__(self, alpha):
        super(ExpLogLoss, self).__init__()
        self.alpha = nn.ParameterList([nn.Parameter(alpha * torch.ones(1)) for _ in range(26)])
        
    def check(self):
        param_tensors = [p.data for p in self.alpha]
        mean = torch.mean(torch.cat(param_tensors))  
        return mean
    
    def forward(self, input, target):
        log_input = torch.log_softmax(input, dim=1).cuda()
        alpha = torch.cat([a for a in self.alpha]).cuda()
        loss = -torch.mean(torch.sum(target * log_input * torch.exp(-alpha), dim=1))
        return loss

if __name__ == '__main__':
    main()
