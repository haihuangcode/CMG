import os
import time
import random
import shutil
import torch
import torch.nn as nn
import numpy as np
import argparse
import logging

from config import cfg
from dataloader import S4Dataset
# from torchvggish import vggish

from utils import pyutils
from utils.utility import logger, mask_iou, Eval_Fmeasure, save_mask
from utils.system import setup_logging
import pdb

from model.main_model_2 import AT_VQVAE_Encoder,AVT_VQVAE_Encoder


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--session_name", default="S4", type=str, help="the S4 setting")
    parser.add_argument("--visual_backbone", default="resnet", type=str, help="use resnet50 or pvt-v2 as the visual backbone")

    parser.add_argument("--test_batch_size", default=1, type=int)
    parser.add_argument("--max_epoches", default=15, type=int)
    parser.add_argument("--lr", default=0.0001, type=float)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--wt_dec", default=5e-4, type=float)

    parser.add_argument("--tpavi_stages", default=[], nargs='+', type=int, help='add tpavi block in which stages: [0, 1, 2, 3')
    parser.add_argument("--tpavi_vv_flag", action='store_true', default=False, help='visual-visual self-attention')
    parser.add_argument("--tpavi_va_flag", action='store_true', default=False, help='visual-audio cross-attention')

    parser.add_argument("--weights",type=str)
    parser.add_argument("--save_pred_mask", action='store_true', default=False, help="save predited masks or not")
    parser.add_argument('--log_dir', default='./test_logs', type=str)

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
    # Encoder = AT_VQVAE_Encoder(text_lstm_dim*2, audio_dim, text_output_dim, audio_output_dim, n_embeddings, embedding_dim)
    
    # AVT
    Encoder = AVT_VQVAE_Encoder(audio_dim, video_dim, text_lstm_dim*2, audio_output_dim, video_output_dim, text_output_dim, n_embeddings, embedding_dim)
    
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
        os.makedirs(args.log_dir)
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
    model.load_state_dict(torch.load(args.weights))
    model = model.cuda()
    # model = torch.nn.DataParallel(model).cuda()
    logger.info('=> Load trained model %s'%args.weights)

    # audio backbone
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Test data
    split = 'test'
    test_dataset = S4Dataset(split)
    test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                        batch_size=args.test_batch_size,
                                                        shuffle=False,
                                                        num_workers=args.num_workers,
                                                        pin_memory=True,
                                                        collate_fn=test_dataset.collate_func)

    avg_meter_miou = pyutils.AverageMeter('miou')
    avg_meter_F = pyutils.AverageMeter('F_score')

    # Test
    model.eval()
    with torch.no_grad():
        for n_iter, batch_data in enumerate(test_dataloader):
            query, imgs, audio_feature, mask = batch_data['query'],batch_data['imgs_tensor'],batch_data['audio_fea'],batch_data['masks_tensor']
            category_list, video_name_list = batch_data['category'],batch_data['video_name']
            query = query.double().cuda()
            imgs = imgs.cuda()
            audio_feature = audio_feature.cuda()
            mask = mask.cuda()
            B, frame, C, H, W = imgs.shape
            imgs = imgs.view(B*frame, C, H, W)
            mask = mask.view(B*frame, H, W)

            audio_feature = audio_feature.to(torch.float64)
            audio_feature = audio_feature.repeat(1, 2, 1)# [B, 5, audio_dim] -> [B, 10, audio_dim]
            
            audio_feature = audio_feature.transpose(2, 1).contiguous()  # [batch, audio_dim, length:10]
            audio_feature = AT_10_5_Linear(audio_feature.to(torch.float32))
            audio_feature = audio_feature.transpose(2, 1).contiguous().to(torch.float64)  # [batch, length:3, audio_dim]
            
            audio_vq = Encoder.Audio_VQ_Encoder(audio_feature)# [B, T, 256]
            
            audio_vq = audio_vq.reshape(-1, audio_vq.shape[-1])
            output, visual_map_list, a_fea_list = model(imgs, audio_vq.to(torch.float32)) # [bs*5, 1, 224, 224]
            
            
            if args.save_pred_mask:
                mask_save_path = os.path.join(log_dir, 'pred_masks')
                save_mask(output.squeeze(1), mask_save_path, category_list, video_name_list)

            miou = mask_iou(output.squeeze(1), mask)
            avg_meter_miou.add({'miou': miou})
            F_score = Eval_Fmeasure(output.squeeze(1), mask, log_dir)
            avg_meter_F.add({'F_score': F_score})
            print('n_iter: {}, iou: {}, F_score: {}'.format(n_iter, miou, F_score))


        miou = (avg_meter_miou.pop('miou'))
        F_score = (avg_meter_F.pop('F_score'))
        print('test miou:', miou.item())
        print('test F_score:', F_score)
        logger.info('test miou: {}, F_score: {}'.format(miou.item(), F_score))