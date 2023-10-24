import os
import time
import random
import shutil
import torch
import torch.nn as nn
import numpy as np
import argparse
import logging
from itertools import chain
from config import cfg
from dataloader import S4Dataset
# from torchvggish import vggish
from loss import IouSemanticAwareLoss

from utils import pyutils
from utils.utility import logger, mask_iou, save_mask
from utils.system import setup_logging
import pdb

from model.main_model_2 import AT_VQVAE_Encoder,AVT_VQVAE_Encoder
from itertools import chain


if __name__ == "__main__":
    parser = argparse.ArgumentParser()    
    parser.add_argument("--session_name", default="S4", type=str, help="the S4 setting")
    parser.add_argument("--visual_backbone", default="resnet", type=str, help="use resnet50 or pvt-v2 as the visual backbone")

    parser.add_argument("--train_batch_size", default=4, type=int)
    parser.add_argument("--val_batch_size", default=1, type=int)
    parser.add_argument("--max_epoches", default=1, type=int)
    parser.add_argument("--lr", default=0.0001, type=float)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--wt_dec", default=5e-4, type=float)


    parser.add_argument('--sa_loss_flag', action='store_true', default=False, help='additional loss for last four frames')
    parser.add_argument("--lambda_1", default=0, type=float, help='weight for balancing l4 loss')
    parser.add_argument("--sa_loss_stages", default=[], nargs='+', type=int, help='compute sa loss in which stages: [0, 1, 2, 3')
    parser.add_argument("--mask_pooling_type", default='avg', type=str, help='the manner to downsample predicted masks')

    parser.add_argument("--tpavi_stages", default=[], nargs='+', type=int, help='add tpavi block in which stages: [0, 1, 2, 3')
    parser.add_argument("--tpavi_vv_flag", action='store_true', default=False, help='visual-visual self-attention')
    parser.add_argument("--tpavi_va_flag", action='store_true', default=False, help='visual-audio cross-attention')

    parser.add_argument("--weights", type=str, default='', help='path of trained model')
    parser.add_argument('--log_dir', default='./train_logs', type=str)

    args = parser.parse_args()

    if (args.visual_backbone).lower() == "resnet":
        from model import ResNet_AVSModel as AVSModel
        print('==> Use ResNet50 as the visual backbone...')
    elif (args.visual_backbone).lower() == "pvt":
        from model import PVT_AVSModel as AVSModel
        print('==> Use pvt-v2 as the visual backbone...')
    else:
        raise NotImplementedError("only support the resnet50 and pvt-v2")

    '''upstream_model setting'''
    text_dim = 768
    video_dim = 512
    audio_dim = 128
    text_lstm_dim = 128
    text_output_dim = 256
    video_output_dim = 2048
    audio_output_dim = 256
    n_embeddings = 400
    embedding_dim = 256
    start_epoch = -1
    model_resume = False
    total_step = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Text_ar_lstm = nn.LSTM(text_dim, text_lstm_dim, num_layers=2, batch_first=True, bidirectional=True)
    
    # AT
    Encoder = AT_VQVAE_Encoder(text_lstm_dim*2, audio_dim, text_output_dim, audio_output_dim, n_embeddings, embedding_dim)
    
    # AVT
    # Encoder = AVT_VQVAE_Encoder(audio_dim, video_dim, text_lstm_dim*2, audio_output_dim, video_output_dim, text_output_dim, n_embeddings, embedding_dim)
    
    AT_10_5_Linear = nn.Linear(10, 5)
        
    Text_ar_lstm.double()
    Encoder.double()
    
    Text_ar_lstm.cuda()
    Encoder.cuda()
    AT_10_5_Linear.cuda()
    
    if model_resume is True:
        path_checkpoints = "..."
        
        checkpoints = torch.load(path_checkpoints)
        Text_ar_lstm.load_state_dict(checkpoints['Text_ar_lstm_parameters'])
        Encoder.load_state_dict(checkpoints['Encoder_parameters'])
        start_epoch = checkpoints['epoch']
        print("Resume from number {}-th model.".format(start_epoch))

    # Fix seed
    FixSeed = 123
    random.seed(FixSeed)
    np.random.seed(FixSeed)
    torch.manual_seed(FixSeed)
    torch.cuda.manual_seed(FixSeed)

    # Log directory
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir, exist_ok=True)
    # Logs
    prefix = args.session_name
    log_dir = os.path.join(args.log_dir, '{}'.format(time.strftime(prefix + '_%Y%m%d-%H%M%S')))
    args.log_dir = log_dir

    # Save scripts
    script_path = os.path.join(log_dir, 'scripts')
    if not os.path.exists(script_path):
        os.makedirs(script_path, exist_ok=True)

    scripts_to_save = ['train.sh', 'train.py', 'test.sh', 'test.py', 'config.py', 'dataloader.py', './model/ResNet_AVSModel.py', './model/PVT_AVSModel.py', 'loss.py']
    for script in scripts_to_save:
        dst_path = os.path.join(script_path, script)
        try:
            shutil.copy(script, dst_path)
        except IOError:
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            shutil.copy(script, dst_path)

    # Checkpoints directory
    checkpoint_dir = os.path.join(log_dir, 'checkpoints')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True)
    args.checkpoint_dir = checkpoint_dir

    # Set logger
    log_path = os.path.join(log_dir, 'log')
    if not os.path.exists(log_path):
        os.makedirs(log_path, exist_ok=True)

    setup_logging(filename=os.path.join(log_path, 'log.txt'))
    logger = logging.getLogger(__name__)
    logger.info('==> Config: {}'.format(cfg))
    logger.info('==> Arguments: {}'.format(args))
    logger.info('==> Experiment: {}'.format(args.session_name))

    # Model
    model = AVSModel.Pred_endecoder(channel=256, \
                                        config=cfg, \
                                        tpavi_stages=args.tpavi_stages, \
                                        tpavi_vv_flag=args.tpavi_vv_flag, \
                                        tpavi_va_flag=args.tpavi_va_flag)
    model = model.cuda()
    model.train()

    # Data
    train_dataset = S4Dataset('train')
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                        batch_size=args.train_batch_size,
                                                        shuffle=True,
                                                        num_workers=args.num_workers,
                                                        pin_memory=True,
                                                        collate_fn=train_dataset.collate_func)
    max_step = (len(train_dataset) // args.train_batch_size) * args.max_epoches

    val_dataset = S4Dataset('val')
    val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                                        batch_size=args.val_batch_size,
                                                        shuffle=False,
                                                        num_workers=args.num_workers,
                                                        pin_memory=True,
                                                        collate_fn=val_dataset.collate_func)

    # Optimizer
    model_params = model.parameters()
    optimizer = torch.optim.Adam(chain(model_params,AT_10_5_Linear.parameters()), args.lr)
    avg_meter_total_loss = pyutils.AverageMeter('total_loss')
    avg_meter_iou_loss = pyutils.AverageMeter('iou_loss')
    avg_meter_sa_loss = pyutils.AverageMeter('sa_loss')
    avg_meter_miou = pyutils.AverageMeter('miou')
    avg_meter_val_loss = pyutils.AverageMeter('val_loss')

    # Train
    best_epoch = 0
    global_step = 0
    global_step_val = 0
    miou_list = []
    max_miou = 0
    for epoch in range(args.max_epoches):
        for n_iter, batch_data in enumerate(train_dataloader):
            # [bs, 5, 3, 224, 224], [bs, 5, 128], [bs, 1, 1, 224, 224]
            imgs, audio_feature, mask = batch_data['imgs_tensor'],batch_data['audio_fea'],batch_data['masks_tensor']
            category_list, video_name_list = batch_data['category'],batch_data['video_name']
            
            imgs = imgs.cuda()
            audio_feature = audio_feature.cuda()
            mask = mask.cuda()
            B, frame, C, H, W = imgs.shape
            imgs = imgs.view(B*frame, C, H, W)
            mask = mask.view(B, H, W)
            
            audio_feature = audio_feature.to(torch.float64)
            # train_data:vggsound 10s but avs only 5s
            audio_feature = audio_feature.repeat(1, 2, 1)# [B, 5, audio_dim] -> [B, 10, audio_dim]
            
            audio_feature = audio_feature.transpose(2, 1).contiguous()  # [batch, audio_dim, length:10]
                
            audio_feature = AT_10_5_Linear(audio_feature.to(torch.float32))
            with torch.no_grad():
                audio_feature = audio_feature.transpose(2, 1).contiguous().to(torch.float64)  # [batch, length:3, audio_dim]
            audio_vq = Encoder.Audio_VQ_Encoder(audio_feature)# [B, T, 256]

            audio_vq = audio_vq.reshape(-1, audio_vq.shape[-1])
            output, visual_map_list, a_fea_list = model(imgs, audio_vq.to(torch.float32)) # [bs*5, 1, 224, 224]
            loss, loss_dict = IouSemanticAwareLoss(output, mask.unsqueeze(1).unsqueeze(1), \
                                                a_fea_list, visual_map_list, \
                                                lambda_1=args.lambda_1, \
                                                count_stages=args.sa_loss_stages, \
                                                sa_loss_flag=args.sa_loss_flag, \
                                                mask_pooling_type=args.mask_pooling_type)

            if (global_step-1) % 50 == 0:
                mask_save_path = "..."
                save_mask(output.squeeze(1),mask_save_path, category_list, video_name_list)


            avg_meter_total_loss.add({'total_loss': loss.item()})
            avg_meter_iou_loss.add({'iou_loss': loss_dict['iou_loss']})
            avg_meter_sa_loss.add({'sa_loss': loss_dict['sa_loss']})

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            global_step += 1

            if (global_step-1) % 100 == 0:
                train_log = 'Iter:%5d/%5d, Total_Loss:%.4f, iou_loss:%.4f, sa_loss:%.4f, lambda_1:%.4f, lr: %.4f'%(
                            global_step-1, max_step, avg_meter_total_loss.pop('total_loss'), avg_meter_iou_loss.pop('iou_loss'), avg_meter_sa_loss.pop('sa_loss'), args.lambda_1, optimizer.param_groups[0]['lr'])
                logger.info(train_log)


        # Validation:
        model.eval()
        with torch.no_grad():
            for n_iter, batch_data in enumerate(val_dataloader):
                query, imgs, audio_feature, mask = batch_data['query'],batch_data['imgs_tensor'],batch_data['audio_fea'],batch_data['masks_tensor']
                category_list, video_name_list = batch_data['category'],batch_data['video_name']
                query = query.double().cuda()
                imgs = imgs.cuda()
                audio_feature = audio_feature.cuda()
                mask = mask.cuda()
                B, frame, C, H, W = imgs.shape
                imgs = imgs.view(B*frame, C, H, W)
                mask = mask.view(B*frame, H, W)

                batch_dim = query.size()[0]
                hidden_dim = 128
                num_layers = 2
                text_hidden = (torch.zeros(2*num_layers, batch_dim, hidden_dim).double().cuda(),
                  torch.zeros(2*num_layers, batch_dim, hidden_dim).double().cuda())
                text_feature, text_hidden = Text_ar_lstm(query, text_hidden)
                text_feature = text_feature.cuda()
                
                text_feature = text_feature.transpose(2, 1).contiguous()  # [batch, text_dim, length:10]
                text_feature = AT_10_5_Linear(text_feature.to(torch.float32))
                text_feature = text_feature.transpose(2, 1).contiguous().to(torch.float64)  # [batch, length:3, text_dim]
                
                text_vq = Encoder.Text_VQ_Encoder(text_feature)# [B, T, 256]
                
                text_vq = text_vq.reshape(-1, text_vq.shape[-1])
                
                output, visual_map_list, a_fea_list = model(imgs, text_vq.to(torch.float32)) # [bs*5, 1, 224, 224]
                
                if (global_step_val-1) % 20 == 0:
                    mask_save_path = "..."
                    save_mask(output.squeeze(1),mask_save_path, category_list, video_name_list)
                
                global_step_val += 1

                miou = mask_iou(output.squeeze(1), mask)
                avg_meter_miou.add({'miou': miou})

            miou = (avg_meter_miou.pop('miou'))
            if miou > max_miou:
                model_save_path = os.path.join("...", '%s_best.pth'%(args.session_name))
                torch.save(model.state_dict(), model_save_path)
                best_epoch = epoch
                logger.info('save best model to %s'%model_save_path)

            miou_list.append(miou)
            max_miou = max(miou_list)

            val_log = 'Epoch: {}, Miou: {}, maxMiou: {}'.format(epoch, miou, max_miou)
            logger.info(val_log)

        model.train()
    logger.info('best val Miou {} at peoch: {}'.format(max_miou, best_epoch))