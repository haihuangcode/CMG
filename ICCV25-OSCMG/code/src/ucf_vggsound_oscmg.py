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
from torch.optim.lr_scheduler import StepLR, MultiStepLR

import numpy as np
from configs.opts import parser
from model.main_model_2 import AV_VQVAE_Encoder
from model.main_model_2 import AV_VQVAE_Decoder
from model.main_model_2 import Semantic_Decoder, AVT_VQVAE_Encoder,AVT_RQVAE_Encoder,AVT_MLVQVAE_Encoder

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
    global args, logger, dataset_configs
    # statistics variable
    
    global best_precision
    best_precision=0
    
    global best_overall_Hscore, best_thred_Hscore, best_acc_insider, best_acc_outsider
    best_overall_Hscore, best_thred_Hscore, best_acc_insider, best_acc_outsider = 0,0,0,0
    
    # configs
    dataset_configs = get_and_save_args(parser)
    parser.set_defaults(**dataset_configs)
    args = parser.parse_args()
    # select GPUs
    # os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu#"0"
    
    

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
        train_csv_path = 'data/OSCMG(cvpr25)/data/UCF_VGG_OSCMG/ucf2vggsound_' + str(args.source_class_num) + '.csv'
        val_csv_path = 'data/feature_extractor/feature_extractor/vggsound2ucf.csv'
        audio_fea_base_path = 'data/vggsound40k/feature/audio/zip'
        video_fea_base_path = 'data/UCF101/feature/video/zip'
        
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
        train_csv_path = 'data/OSCMG(cvpr25)/data/UCF_VGG_OSCMG/vggsound2ucf_' + str(args.source_class_num) + '.csv'
        val_csv_path = 'data/feature_extractor/feature_extractor/ucf2vggsound.csv'
        audio_fea_base_path = 'data/vggsound40k/feature/audio/zip'
        video_fea_base_path = 'data/UCF101/feature/video/zip'
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
    
    Decoder = Semantic_Decoder(input_dim=embedding_dim, class_num = args.source_class_num)
    Encoder.double()
    Decoder.double()
    '''optimizer setting'''
    Encoder.to(device)
    Decoder.to(device)
    optimizer = torch.optim.Adam(chain(Encoder.parameters(), Decoder.parameters()), lr=args.lr)

    scheduler = MultiStepLR(optimizer, milestones=[10, 20, 30], gamma=0.5)
    
    '''loss'''
    criterion = nn.BCEWithLogitsLoss().cuda()
    criterion_event = nn.CrossEntropyLoss().cuda()


    if model_resume is True:
        path_checkpoints = "MICU.pt"# MICU
        checkpoints = torch.load(path_checkpoints)
        Encoder.load_state_dict(checkpoints['Encoder_parameters'])
        start_epoch = checkpoints['epoch']
        logger.info("Resume from number {}-th model.".format(start_epoch))


    '''Training and Evaluation'''

    for epoch in range(start_epoch+1, args.n_epoch):
        
        loss, total_step = train_epoch(Encoder, Decoder, train_dataloader, criterion, criterion_event,
                                       optimizer, epoch, total_step, args)
        logger.info(f"epoch: *******************************************{epoch}")

        if ((epoch + 1) % args.eval_freq == 0) or (epoch == args.n_epoch - 1):
            validate_epoch(Encoder, Decoder, val_dataloader, criterion, criterion_event, epoch, args)
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

        if n_iter % 30 == 0:
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

    output_sum = []
    target_sum = []

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
                vq, indices = Encoder.Audio_VQ_Encoder(feat)
            elif (args.dataset_name == 'vgga_ucfv'):
                vq, indices = Encoder.Video_VQ_Encoder(feat)
            else:
                raise NotImplementedError
        _class = Decoder(vq)
        
        outlier_flag = (labels_event > (args.source_class_num - 1)).float()
        target = labels_event * (1 - outlier_flag) + args.source_class_num * outlier_flag
        target = target.long()
        
        output_sum.append(_class)
        target_sum.append(target)
        
        # event_loss = criterion_event(_class, labels_event.cuda())
        # precision = compute_accuracy_supervised(_class, labels)
        # loss_items = {
        #     "val_event_loss":event_loss.item(),
        #     "val_precision": precision.item(),
        # }
        # val_precision.update(precision.item(), bs * 10)
        # metricsContainer.update("loss", loss_items)
        # loss = event_loss
        # losses.update(loss.item(), bs * 10)

    output_sum = torch.cat(output_sum)
    target_sum = torch.cat(target_sum)
    
    # import pdb
    # pdb.set_trace()
    
    tsm_output = F.softmax(output_sum, dim=1)
    outlier_indis, max_index = torch.max(tsm_output, 1)
    thd_min = torch.min(outlier_indis)
    thd_max = torch.max(outlier_indis)
    outlier_range = [thd_min + (thd_max - thd_min) * k / 9 for k in range(10)]
    
    best_overall_acc = 0.0
    best_thred_acc = 0.0
    global best_overall_Hscore, best_thred_Hscore, best_acc_insider, best_acc_outsider
    
    for outlier_thred in outlier_range:
        outlier_pred = (outlier_indis < outlier_thred).double()
        outlier_pred = outlier_pred.view(-1, 1)
        output = torch.cat((tsm_output, outlier_pred.cuda()), dim=1)

        _, predict = torch.max(output.detach(), dim=1)
        overall_acc = (predict == target_sum).sum().item() / target_sum.shape[0]

        indices_outsider = torch.where(target_sum == args.source_class_num)[0]
        indices_insider = torch.where(target_sum != args.source_class_num)[0]
        acc_insider = (predict[indices_insider] == target_sum[indices_insider]).sum().item() / target_sum[indices_insider].shape[0]
        acc_outsider = (predict[indices_outsider] == target_sum[indices_outsider]).sum().item() / target_sum[indices_outsider].shape[0]
        overall_Hscore = 2.0 * acc_insider * acc_outsider / (acc_insider + acc_outsider)

        if overall_Hscore > best_overall_Hscore:
            best_overall_Hscore = overall_Hscore
            best_thred_Hscore = outlier_thred
            best_acc_insider = acc_insider
            best_acc_outsider = acc_outsider
            
            # 保存绘图所需的数据
            # import pandas as pd
            # import os
            # plot_data = []
            # for i in range(target_sum.shape[0]):
            #     item = {
            #         "index": i,
            #         "true_class": int(target_sum[i].item()),
            #         "pred_class": int(predict[i].item()),
            #         "correct": int(predict[i] == target_sum[i]),
            #         "is_unknown": int(target_sum[i] == args.source_class_num)
            #     }
            #     plot_data.append(item)
            # df = pd.DataFrame(plot_data)
            # df.to_csv(os.path.join('data/OSCMG(cvpr25)/draw/openset_draw', f"ucf_vgg_oscmg_data.csv"), index=False)
            
            save_dir = os.path.join('data/OSCMG(cvpr25)/draw', "openset_draw")
            os.makedirs(save_dir, exist_ok=True)

            # (1) 保存 codeword 的使用统计和多数类投票信息
            code_usage = defaultdict(list)  # code_id -> list of (true_class, pred_class, correct)
            for i in range(vq_codes.shape[0]):
                code_ids = vq_codes[i].detach().cpu().tolist()
                true_cls = int(target_sum[i].item())
                pred_cls = int(predict[i].item())
                correct = int(pred_cls == true_cls)
                for code_id in code_ids:
                    code_usage[code_id].append((true_cls, pred_cls, correct))

            records = []
            for code_id, entries in code_usage.items():
                true_classes = [e[0] for e in entries]
                pred_classes = [e[1] for e in entries]
                correct_flags = [e[2] for e in entries]
                majority_true = Counter(true_classes).most_common(1)[0][0]
                majority_pred = Counter(pred_classes).most_common(1)[0][0]
                majority_correct = int(majority_true == majority_pred)
                records.append({
                    'code_id': code_id,
                    'true_class': majority_true,
                    'pred_class': majority_pred,
                    'correct': majority_correct,
                    'used_count': len(entries),
                    'is_unknown': int(majority_true == args.source_class_num)
                })
            df_codebook = pd.DataFrame(records)
            df_codebook.to_csv(os.path.join(save_dir, f"codebook_epoch_{epoch}.csv"), index=False)

            # (2) 保存 codebook 本身用于 t-SNE 可视化
            codebook_np = Encoder.codebook.detach().cpu().numpy()  # shape: [num_codes, dim]
            np.save(os.path.join(save_dir, f"codebook_epoch_{epoch}.npy"), codebook_np)
        
    logger.info(
        f'**************************************************************************\t'
        f"\t task: {args.dataset_name}"
        f"\t best_overall_Hscore: {best_overall_Hscore*100.0:.4f}%."
        # f"\t best_thred_Hscore: {best_thred_Hscore*100.0:.4f}%."
        f"\t best_acc_insider: {best_acc_insider*100.0:.4f}%."
        f"\t best_acc_outsider: {best_acc_outsider*100.0:.4f}%."
    )
    
    return None

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
