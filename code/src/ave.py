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
from model.main_model_2 import AV_VQVAE_Encoder as AV_VQVAE_Encoder
from model.main_model_2 import AV_VQVAE_Decoder as AV_VQVAE_Decoder
from model.main_model_2 import Semantic_Decoder,AVT_VQVAE_Encoder
from utils import AverageMeter, Prepare_logger, get_and_save_args
from utils.container import metricsContainer
from utils.Recorder import Recorder
import torch.nn.functional as F

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
    if args.dataset_name == 'ave_va' or args.dataset_name == 'ave_av':
        from dataset.AVE_dataset import AVEDataset as AVEDataset
    else: 
        raise NotImplementedError
    
  
    '''Dataloader selection'''
    data_root = 'AVE-ECCV18-master/data'
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
    Decoder = Semantic_Decoder(input_dim=embedding_dim, class_num=28) 
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
            loss = validate_epoch(Encoder, Decoder, train_dataloader, criterion, criterion_event, epoch)
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


def train_epoch(Encoder, Decoder, train_dataloader, criterion, criterion_event, optimizer, epoch, total_step, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    train_acc = AverageMeter()
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
        visual_feature, audio_feature, labels = batch_data
        visual_feature.cuda()
        audio_feature.cuda()
        labels = labels.double().cuda()
        labels_foreground = labels[:, :, :-1]  
        labels_BCE, labels_evn = labels_foreground.max(-1)

        labels_event, _ = labels_evn.max(-1)
        
        if args.dataset_name == 'ave_va':
            with torch.no_grad():# Freeze Encoder
                video_vq = Encoder.Video_VQ_Encoder(visual_feature)
            video_class = Decoder(video_vq)
            video_event_loss = criterion_event(video_class, labels_event.cuda())
            video_acc = compute_accuracy_supervised(video_class, labels)
            loss_items = {
                "video_event_loss":video_event_loss.item(),
                "video_acc": video_acc.item(),
            }
            metricsContainer.update("loss", loss_items)
            loss = video_event_loss
        elif args.dataset_name == 'ave_av':
            with torch.no_grad():# Freeze Encoder
                audio_vq = Encoder.Audio_VQ_Encoder(audio_feature)
            audio_class = Decoder(audio_vq)
            audio_event_loss = criterion_event(audio_class, labels_event.cuda())
            audio_acc = compute_accuracy_supervised(audio_class, labels)
            loss_items = {
                "audio_event_loss":audio_event_loss.item(),
                "audio_acc": audio_acc.item(),
            }
            metricsContainer.update("loss", loss_items)
            loss = audio_event_loss

        if n_iter % 20 == 0:
            _export_log(epoch=epoch, total_step=total_step+n_iter, batch_idx=n_iter, lr=optimizer.state_dict()['param_groups'][0]['lr'], loss_meter=metricsContainer.calculate_average("loss"))
        loss.backward()


        '''Clip Gradient'''
        if args.clip_gradient is not None:
            for model in models:
                total_norm = clip_grad_norm_(model.parameters(), args.clip_gradient)

        '''Update parameters'''
        optimizer.step()
        optimizer.zero_grad()

        losses.update(loss.item(), audio_feature.size(0) * 10)
        batch_time.update(time.time() - end_time)
        end_time = time.time()

        '''Add loss of a iteration in Tensorboard'''
        writer.add_scalar('Train_data/loss', losses.val, epoch * len(train_dataloader) + n_iter + 1)

        '''Add loss of an epoch in Tensorboard'''
        writer.add_scalar('Train_epoch_data/epoch_loss', losses.avg, epoch)

    return losses.avg, n_iter + total_step


@torch.no_grad()
def validate_epoch(Encoder,Decoder, val_dataloader, criterion, criterion_event, epoch, eval_only=False):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    audio_accuracy = AverageMeter()
    video_accuracy = AverageMeter()
    end_time = time.time()

    Encoder.eval()
    Decoder.eval()
    Encoder.cuda()
    Decoder.cuda()

    for n_iter, batch_data in enumerate(val_dataloader):
        data_time.update(time.time() - end_time)

        '''Feed input to model'''
        visual_feature, audio_feature, labels = batch_data
        visual_feature.cuda()
        audio_feature.cuda()
        audio_feature = audio_feature.to(torch.float64)

        labels = labels.double().cuda()
        labels_foreground = labels[:, :, :-1]
        labels_BCE, labels_evn = labels_foreground.max(-1)
        labels_event, _ = labels_evn.max(-1)

        bs = visual_feature.size(0)
        
        audio_vq = Encoder.Audio_VQ_Encoder(audio_feature)
        video_vq = Encoder.Video_VQ_Encoder(visual_feature)
        audio_class = Decoder(audio_vq)
        video_class = Decoder(video_vq)
        
        audio_event_loss = criterion_event(audio_class, labels_event.cuda())
        video_event_loss = criterion_event(video_class, labels_event.cuda())

        loss = audio_event_loss

        audio_acc = compute_accuracy_supervised(audio_class, labels)
        video_acc = compute_accuracy_supervised(video_class, labels)
        audio_accuracy.update(audio_acc.item(), bs * 10)
        video_accuracy.update(video_acc.item(), bs * 10)
        batch_time.update(time.time() - end_time)
        end_time = time.time()
        losses.update(loss.item(), bs * 10)

    logger.info(
        f'**************************************************************************\t'
        f"\t Audio Evaluation results (acc): {audio_accuracy.avg:.4f}%."
        f"\t Video Evaluation results (acc): {video_accuracy.avg:.4f}%."
    )
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