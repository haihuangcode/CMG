import logging
import os
import time
import random
import json
from tqdm import tqdm
import sys
# import wandb
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
from model.main_model_2 import Semantic_Decoder, AVT_VQVAE_Encoder

from utils import AverageMeter, Prepare_logger, get_and_save_args
from utils.container import metricsContainer
from utils.Recorder import Recorder

import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from dataset.UCF_VGGSOUND_datasets import VGGSoundDataset, UCFDataset

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
    
    global best_precision
    best_precision=0
    
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

    '''Dataloader selection'''
    if args.dataset_name == 'ucfv_vgga':
        train_csv_path = 'ucf2vggsound.csv'
        val_csv_path = 'vggsound2ucf.csv'
        audio_fea_base_path = 'vggsound40k/feature/audio/zip'
        video_fea_base_path = 'UCF101/feature/video/zip'
        
        train_dataloader = DataLoader(
            UCFDataset(train_csv_path, video_fea_base_path, split='train', modality='video'),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=False
        )
        val_dataloader = DataLoader(
            VGGSoundDataset(val_csv_path, audio_fea_base_path, split='val', modality='audio'),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=8,
            pin_memory=False
        ) 
    elif args.dataset_name == 'vgga_ucfv':
        train_csv_path = 'vggsound2ucf.csv'
        val_csv_path = 'ucf2vggsound.csv'
        audio_fea_base_path = 'vggsound40k/feature/audio/zip'
        video_fea_base_path = 'UCF101/feature/video/zip'
        train_dataloader = DataLoader(
            VGGSoundDataset(train_csv_path, audio_fea_base_path, split='train', modality='audio'),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=False
        )
        val_dataloader = DataLoader(
            UCFDataset(val_csv_path, video_fea_base_path, split='val', modality='video'),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=8,
            pin_memory=False
        )
    else:
        raise NotImplementedError

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
    
    Decoder = Semantic_Decoder(input_dim=embedding_dim, class_num = 16)
    Encoder.double()
    Decoder.double()
    '''optimizer setting'''
    Encoder.to(device)
    Decoder.to(device)
    optimizer = torch.optim.Adam(Decoder.parameters(), lr=args.lr)

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
        
        loss, total_step = train_epoch(Encoder, Decoder, train_dataloader, criterion, criterion_event,
                                       optimizer, epoch, total_step, args)
        logger.info(f"epoch: *******************************************{epoch}")

        if ((epoch + 1) % args.eval_freq == 0) or (epoch == args.n_epoch - 1):
            loss = validate_epoch(Encoder, Decoder, val_dataloader, criterion, criterion_event, epoch, args)
            logger.info("-----------------------------")
            logger.info(f"evaluate loss:{loss}")
            logger.info("-----------------------------")
        scheduler.step()


def _export_log(epoch, total_step, batch_idx, lr, loss_meter):
    msg = 'Epoch {}, Batch {}, lr = {:.5f}, '.format(epoch, batch_idx, lr)
    for k, v in loss_meter.items():
        msg += '{} = {:.4f}, '.format(k, v)
    # msg += '{:.3f} seconds/batch'.format(time_meter)
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


def train_epoch(Encoder, Decoder, train_dataloader, criterion, criterion_event, optimizer, epoch, total_step, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    train_precision = AverageMeter()
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
        feat, labels = batch_data
        feat = feat.to(torch.float64).cuda()
        bs = feat.size(0)
        labels = labels.double().cuda()
        labels_foreground = labels[:, :, :-1]  
        labels_BCE, labels_evn = labels_foreground.max(-1)
        labels_event, _ = labels_evn.max(-1)

        with torch.no_grad():
            if (args.dataset_name == 'ucfv_vgga'):
                vq = Encoder.Video_VQ_Encoder(feat)
            elif (args.dataset_name == 'vgga_ucfv'):
                vq = Encoder.Audio_VQ_Encoder(feat)
            else:
                raise NotImplementedError
        _class = Decoder(vq)
        event_loss = criterion_event(_class, labels_event.cuda())
        precision = compute_accuracy_supervised(_class, labels)
        loss_items = {
            "train_event_loss":event_loss.item(),
            "train_precision": precision.item(),
        }
        train_precision.update(precision.item(), bs * 10)
        metricsContainer.update("loss", loss_items)
        loss = event_loss

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

        losses.update(loss.item(), feat.size(0) * 10)
        batch_time.update(time.time() - end_time)
        end_time = time.time()

        '''Add loss of a iteration in Tensorboard'''
        writer.add_scalar('Train_data/loss', losses.val, epoch * len(train_dataloader) + n_iter + 1)

        '''Add loss of an epoch in Tensorboard'''
        writer.add_scalar('Train_epoch_data/epoch_loss', losses.avg, epoch)

    logger.info(f'Train results (precision): {train_precision.avg:.4f}')
    return losses.avg, n_iter + total_step


@torch.no_grad()
def validate_epoch(Encoder,Decoder,val_dataloader, criterion, criterion_event, epoch, args, eval_only=False):
    Sigmoid_fun = nn.Sigmoid()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    val_precision = AverageMeter()
    end_time = time.time()

    Encoder.eval()
    Decoder.eval()
    Encoder.cuda()
    Decoder.cuda()


    for n_iter, batch_data in enumerate(val_dataloader):
        data_time.update(time.time() - end_time)

        '''Feed input to model'''
        feat, labels = batch_data
        feat = feat.to(torch.float64).cuda()
        bs = feat.size(0)
        labels = labels.double().cuda()
        labels_foreground = labels[:, :, :-1]  
        labels_BCE, labels_evn = labels_foreground.max(-1)
        labels_event, _ = labels_evn.max(-1)
        
        with torch.no_grad():
            if (args.dataset_name == 'ucfv_vgga'):
                vq = Encoder.Audio_VQ_Encoder(feat)
            elif (args.dataset_name == 'vgga_ucfv'):
                vq = Encoder.Video_VQ_Encoder(feat)
            else:
                raise NotImplementedError
        _class = Decoder(vq)
        event_loss = criterion_event(_class, labels_event.cuda())
        precision = compute_accuracy_supervised(_class, labels)
        loss_items = {
            "val_event_loss":event_loss.item(),
            "val_precision": precision.item(),
        }
        val_precision.update(precision.item(), bs * 10)
        metricsContainer.update("loss", loss_items)
        loss = event_loss
        losses.update(loss.item(), bs * 10)

    global best_precision
    if val_precision.avg > best_precision:
        best_precision = val_precision.avg

    logger.info(f'Eval results (precision and loss): {val_precision.avg:.4f} {losses.avg:.4f}')
    logger.info(f"Best results (precision): {best_precision:.4f}")
    return losses.avg

def compute_accuracy_supervised(event_scores, labels):
    labels_foreground = labels[:, :, :-1]
    labels_BCE, labels_evn = labels_foreground.max(-1)
    labels_event, _ = labels_evn.max(-1)
    _, event_class = event_scores.max(-1)
    correct = event_class.eq(labels_event)
    correct_num = correct.sum().double()
    acc = correct_num * (100. / correct.numel())
    return acc

if __name__ == '__main__':
    main()
