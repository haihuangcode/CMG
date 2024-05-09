import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from tqdm import tqdm
import numpy as np
# from preprocess import mulaw_decode
import math
from torch.nn import MultiheadAttention
from model.models import EncoderLayer, Encoder, DecoderLayer
from torch import Tensor
# The model is testing
from model.mine import MINE
from info_nce import InfoNCE
import random
random.seed(123)

class InternalTemporalRelationModule(nn.Module):
    def __init__(self, input_dim, d_model):
        super(InternalTemporalRelationModule, self).__init__()
        self.encoder_layer = EncoderLayer(d_model=d_model, nhead=4)
        self.encoder = Encoder(self.encoder_layer, num_layers=2)

        self.affine_matrix = nn.Linear(input_dim, d_model)
        self.relu = nn.ReLU(inplace=True)
        # add relu here?

    def forward(self, feature):
        # feature: [seq_len, batch, dim]
        feature = self.affine_matrix(feature)
        feature = self.encoder(feature)
        return feature


class ResidualLayer(nn.Module):
    """
    One residual layer inputs:
    - in_dim : the input dimension
    - h_dim : the hidden layer dimension
    - res_h_dim : the hidden dimension of the residual block
    """

    def __init__(self, in_dim, h_dim, res_h_dim):
        super(ResidualLayer, self).__init__()
        self.res_block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(in_dim, res_h_dim, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(res_h_dim, h_dim, kernel_size=1,
                      stride=1, bias=False)
        )

    def forward(self, x):
        x = x + self.res_block(x)
        return x


class ResidualStack(nn.Module):
    """
    A stack of residual layers inputs:
    - in_dim : the input dimension
    - h_dim : the hidden layer dimension
    - res_h_dim : the hidden dimension of the residual block
    - n_res_layers : number of layers to stack
    """

    def __init__(self, in_dim, h_dim, res_h_dim, n_res_layers):
        super(ResidualStack, self).__init__()
        self.n_res_layers = n_res_layers
        self.stack = nn.ModuleList(
            [ResidualLayer(in_dim, h_dim, res_h_dim)] * n_res_layers)

    def forward(self, x):
        for layer in self.stack:
            x = layer(x)
        x = F.relu(x)
        return x


class Video_Semantic_Encoder(nn.Module):
    def __init__(self, video_dim):
        super(Video_Semantic_Encoder, self).__init__()
        self.reduction = 8
        self.aver_pool = nn.AdaptiveAvgPool2d(1)
        self.se_layer = nn.Sequential(
            nn.Linear(video_dim, video_dim // self.reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(video_dim // self.reduction, video_dim, bias=False),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU()
        self.affine_video_ave = nn.Linear(video_dim, video_dim // 2)
        self.affine_video_self = nn.Linear(video_dim, video_dim // 2)
        self.ave_v_att = nn.Linear(video_dim // 2, 1)
        self.softmax = nn.Softmax(dim=-1)
        self.tanh = nn.Tanh()

    def forward(self, video_feat):
        batch, length, h, w, v_dim = video_feat.size()
        video_feat = video_feat.reshape(batch * length, h, w, v_dim)
        average_video_feat = video_feat.permute(0, 3, 1, 2)
        average_video_feat = self.aver_pool(average_video_feat).view(batch * length, v_dim)
        average_attention = self.se_layer(average_video_feat).view(batch * length, v_dim, 1, 1).permute(0, 2, 3, 1)
        video_channel_att = video_feat * average_attention.expand_as(video_feat) + video_feat

        video_average = self.relu(self.affine_video_ave(average_video_feat)).unsqueeze(-2)
        self_video_att_feat = video_channel_att.reshape((batch * length, -1, v_dim))
        self_video_att_query = self.relu(self.affine_video_self(self_video_att_feat))
        self_query = self_video_att_query * video_average
        self_spatial_att_maps = self.softmax(self.tanh(self.ave_v_att(self_query))).transpose(2, 1)

        self_att_feat = torch.bmm(self_spatial_att_maps,
                                  video_channel_att.view(batch * length, -1, v_dim)).squeeze().reshape(batch, length,
                                                                                                       v_dim)

        return self_att_feat

""" class_num AVE:28  VGGSOUND:141  UCF_VGGSOUND:16 """
class Semantic_Decoder(nn.Module):
    def __init__(self, input_dim, class_num):
        super(Semantic_Decoder, self).__init__()
        self.linear = nn.Linear(input_dim, input_dim)
        self.event_classifier = nn.Linear(input_dim, class_num)  

    def forward(self, input_vq):
        input_feat = self.linear(input_vq)
        input_feat, _ = input_feat.max(1)
        class_logits = self.event_classifier(input_feat)
        return class_logits

""" class_num AVVP:25+1(negative label) AVE_AVVP:12+1 """
class Semantic_Decoder_AVVP(nn.Module):
    def __init__(self, input_dim, class_num):
        super(Semantic_Decoder_AVVP, self).__init__()
        self.linear = nn.Linear(input_dim, input_dim)
        self.event_classifier = nn.Linear(input_dim, class_num)  

    def forward(self, input_vq):
        input_feat = self.linear(input_vq)
        # input_feat, _ = input_feat.max(1)
        class_logits = self.event_classifier(input_feat)
        return class_logits

class Video_Encoder(nn.Module):
    def __init__(self, video_dim, hidden_dim):
        super(Video_Encoder, self).__init__()
        self.video_dim = video_dim
        self.hidden_dim = hidden_dim
        kernel = 3
        stride = 1
        self.conv_stack = nn.Sequential(
            nn.Conv2d(video_dim, hidden_dim // 2, kernel_size=kernel, stride=stride, padding=0),
            nn.ReLU(),
            nn.Conv2d(hidden_dim // 2, hidden_dim, kernel_size=kernel, stride=stride, padding=0),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1, stride=1, padding=0),
            ResidualStack(
                hidden_dim, hidden_dim, hidden_dim, 1)
        )
        self.video_pool = nn.AvgPool2d(kernel_size=3, stride=1)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, video_dim)
        )

    def forward(self, video_feat):
        batch, length, h, w, channel = video_feat.size()
        video_feat = video_feat.reshape(batch * length, h, w, channel)
        video_feat = video_feat.permute(0, 3, 1, 2)
        result = self.conv_stack(video_feat)
        result = result.permute(0, 2, 3, 1)
        _, h1, w1, _ = result.size()
        mine_result = self.video_pool(result.permute(0, 3, 1, 2))
        mine_result = mine_result.permute(0, 2, 3, 1).reshape(batch, length, -1)
        mine_result = self.feed_forward(mine_result)
        result = result.reshape(batch, length, h1, w1, -1)
        return result, mine_result


class Audio_Encoder(nn.Module):
    def __init__(self, audio_dim, hidden_dim):
        super(Audio_Encoder, self).__init__()
        self.audio_dim = audio_dim
        self.hidden_dim = hidden_dim
        self.audio_linear = nn.Linear(audio_dim, hidden_dim)
        self.relu = nn.ReLU()

    def forward(self, audio_feat):
        return self.relu(self.audio_linear(audio_feat))

class Text_Encoder(nn.Module):
    def __init__(self, text_dim, hidden_dim):
        super(Text_Encoder, self).__init__()
        self.text_dim = text_dim
        self.hidden_dim = hidden_dim
        self.text_linear = nn.Linear(text_dim, hidden_dim)
        self.relu = nn.ReLU()

    def forward(self, text_feat):
        return self.relu(self.text_linear(text_feat))

class Video_Decoder(nn.Module):
    def __init__(self, input_dim, output_dim, vq_dim):
        super(Video_Decoder, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        kernel = 3
        stride = 1

        self.inverse_conv_block = nn.Sequential(
            nn.ConvTranspose2d(input_dim + vq_dim, input_dim // 2, kernel_size=kernel, stride=stride, padding=0),
            ResidualStack(input_dim // 2, input_dim // 2, input_dim // 2, 1),
            nn.ConvTranspose2d(input_dim // 2, output_dim, kernel_size=kernel, stride=stride, padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(output_dim, output_dim, kernel_size=1, stride=1, padding=0)
        )
        self.video_linear = nn.Linear(vq_dim, vq_dim)

    def forward(self, video_encoder_result, video_vq):
        batch, length, h1, w1, dim = video_encoder_result.size()
        video_vq_result = self.video_linear(video_vq).unsqueeze(2).unsqueeze(3)
        video_vq_result = video_vq_result.repeat(1, 1, h1, w1, 1).reshape(batch * length, h1, w1, -1)
        video_encoder_result = video_encoder_result.reshape(batch * length, h1, w1, dim)
        video_encoder_result = torch.cat([video_vq_result, video_encoder_result], dim=3)
        video_encoder_result = video_encoder_result.permute(0, 3, 1, 2)

        video_recon_result = self.inverse_conv_block(video_encoder_result)
        _, dim, H, W = video_recon_result.size()
        video_recon_result = video_recon_result.reshape(batch, length, dim, H, W)
        video_recon_result = video_recon_result.permute(0, 1, 3, 4, 2)

        return video_recon_result

class Audio_Decoder(nn.Module):
    def __init__(self, input_dim, output_dim, vq_dim):
        super(Audio_Decoder, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.relu = nn.ReLU()
        self.audio_rec = nn.Linear(input_dim * 2, output_dim)
        self.audio_linear = nn.Linear(vq_dim, input_dim)

    def forward(self, audio_encoder_result, audio_vq):
        audio_vq_result = self.audio_linear(audio_vq)
        audio_encoder_result = torch.cat([audio_vq_result, audio_encoder_result], dim=2)
        audio_decoder_result = self.audio_rec(audio_encoder_result)
        return audio_decoder_result
    
class Text_Decoder(nn.Module):
    def __init__(self, input_dim, output_dim, vq_dim):
        super(Text_Decoder, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.relu = nn.ReLU()
        self.text_rec = nn.Linear(input_dim * 2, output_dim)
        self.text_linear = nn.Linear(vq_dim, input_dim)

    def forward(self, text_encoder_result, text_vq):
        text_vq_result = self.text_linear(text_vq)
        text_encoder_result = torch.cat([text_vq_result, text_encoder_result], dim=2)
        text_decoder_result = self.text_rec(text_encoder_result)
        return text_decoder_result

class AVT_VQVAE_Encoder(nn.Module):
    def __init__(self, audio_dim, video_dim, text_dim, audio_output_dim, video_output_dim, text_output_dim, n_embeddings, embedding_dim):
        super(AVT_VQVAE_Encoder, self).__init__()
        self.video_dim = video_dim
        self.text_dim = text_dim
        self.audio_dim = audio_dim
        self.hidden_dim = 256
        self.Video_encoder = Video_Encoder(video_dim, video_output_dim)
        self.Audio_encoder = Audio_Encoder(audio_dim, audio_output_dim)
        self.Text_encoder = Text_Encoder(text_dim, text_output_dim)
        
        self.Cross_quantizer = Cross_VQEmbeddingEMA_AVT(n_embeddings, self.hidden_dim)
        self.video_semantic_encoder = Video_Semantic_Encoder(video_dim)
        self.video_self_att = InternalTemporalRelationModule(input_dim=video_dim, d_model=self.hidden_dim)
        self.text_self_att = InternalTemporalRelationModule(input_dim=text_dim, d_model=self.hidden_dim)
        self.audio_self_att = InternalTemporalRelationModule(input_dim=audio_dim, d_model=self.hidden_dim)

    def Audio_VQ_Encoder(self, audio_feat):
        audio_feat = audio_feat.cuda()
        audio_semantic_result = audio_feat.transpose(0, 1).contiguous()
        audio_semantic_result = self.audio_self_att(audio_semantic_result)
        audio_semantic_result = audio_semantic_result.transpose(0, 1).contiguous()
        audio_vq = self.Cross_quantizer.Audio_vq_embedding(audio_semantic_result)
        return audio_vq

    def Video_VQ_Encoder(self, video_feat):
        video_feat = video_feat.cuda()
        video_semantic_result = self.video_semantic_encoder(video_feat).transpose(0, 1).contiguous()
        video_semantic_result = self.video_self_att(video_semantic_result)
        video_semantic_result = video_semantic_result.transpose(0, 1).contiguous()  # [batch, length, hidden_dim]
        video_vq = self.Cross_quantizer.Video_vq_embedding(video_semantic_result)
        return video_vq

    def Text_VQ_Encoder(self, text_feat):
        text_feat = text_feat.cuda()
        text_semantic_result = text_feat.transpose(0, 1).contiguous()
        text_semantic_result = self.text_self_att(text_semantic_result)
        text_semantic_result = text_semantic_result.transpose(0, 1).contiguous()
        text_vq = self.Cross_quantizer.Text_vq_embedding(text_semantic_result)
        return text_vq

    def forward(self, audio_feat, video_feat, text_feat, epoch):
        video_feat = video_feat.cuda()
        text_feat = text_feat.cuda()
        audio_feat = audio_feat.cuda()
        video_encoder_result, video_club_feature = self.Video_encoder(video_feat)
        
        video_semantic_result = self.video_semantic_encoder(video_feat).transpose(0, 1).contiguous()
        video_semantic_result = self.video_self_att(video_semantic_result)
        video_semantic_result = video_semantic_result.transpose(0, 1).contiguous()  # [batch, length, hidden_dim]
        
        text_semantic_result = text_feat.transpose(0, 1).contiguous()
        text_semantic_result = self.text_self_att(text_semantic_result)# [length, batch, hidden_dim]
        text_semantic_result = text_semantic_result.transpose(0, 1).contiguous()  # [batch, length, hidden_dim]
        
        audio_semantic_result = audio_feat.transpose(0, 1).contiguous()
        audio_semantic_result = self.audio_self_att(audio_semantic_result)# [length, batch, hidden_dim]
        audio_semantic_result = audio_semantic_result.transpose(0, 1).contiguous()  # [batch, length, hidden_dim]

        text_encoder_result = self.Text_encoder(text_feat)  # [batch, length, audio_output_dim]
        audio_encoder_result = self.Audio_encoder(audio_feat)  # [batch, length, audio_output_dim]
        
        audio_vq, video_vq, text_vq, audio_embedding_loss, video_embedding_loss, text_embedding_loss, audio_perplexity, video_perplexity, text_perplexity, cmcm_loss, equal_num \
            = self.Cross_quantizer(audio_semantic_result, video_semantic_result, text_semantic_result, epoch)

        return audio_semantic_result, video_semantic_result, text_semantic_result, \
               audio_encoder_result, video_encoder_result, video_club_feature, text_encoder_result, \
               audio_vq, video_vq, text_vq, audio_embedding_loss, video_embedding_loss, text_embedding_loss, cmcm_loss, equal_num


class AT_VQVAE_Encoder(nn.Module):
    def __init__(self, text_dim, audio_dim, text_output_dim, audio_output_dim, n_embeddings, embedding_dim):
        super(AT_VQVAE_Encoder, self).__init__()
        self.text_dim = text_dim
        self.audio_dim = audio_dim
        self.hidden_dim = 256
        self.Audio_encoder = Audio_Encoder(audio_dim, audio_output_dim)
        self.Text_encoder = Text_Encoder(text_dim, text_output_dim)
        
        self.Cross_quantizer = Cross_VQEmbeddingEMA_AT(n_embeddings, self.hidden_dim)
        self.text_self_att = InternalTemporalRelationModule(input_dim=text_dim, d_model=self.hidden_dim)
        self.audio_self_att = InternalTemporalRelationModule(input_dim=audio_dim, d_model=self.hidden_dim)

    def Audio_VQ_Encoder(self, audio_feat):
        audio_feat = audio_feat.cuda()
        audio_semantic_result = audio_feat.transpose(0, 1).contiguous()
        audio_semantic_result = self.audio_self_att(audio_semantic_result)
        audio_semantic_result = audio_semantic_result.transpose(0, 1).contiguous()
        audio_vq = self.Cross_quantizer.Audio_vq_embedding(audio_semantic_result)
        return audio_vq

    def Text_VQ_Encoder(self, text_feat):
        text_feat = text_feat.cuda()
        text_semantic_result = text_feat.transpose(0, 1).contiguous()
        text_semantic_result = self.text_self_att(text_semantic_result)
        text_semantic_result = text_semantic_result.transpose(0, 1).contiguous()
        text_vq = self.Cross_quantizer.Text_vq_embedding(text_semantic_result)
        return text_vq

    def Audio_vq_forward(self, audio_feat, text_feat):
        text_vq = self.Text_VQ_Encoder(text_feat)
        audio_feat = audio_feat.cuda()
        audio_semantic_result = audio_feat.transpose(0, 1).contiguous()
        audio_semantic_result = self.audio_self_att(audio_semantic_result)
        audio_semantic_result = audio_semantic_result.transpose(0, 1).contiguous()
        audio_vq = self.Cross_quantizer.Audio_vq_embedding(audio_semantic_result)
        audio_vq_forward_loss =  F.mse_loss(audio_semantic_result, audio_vq.detach()) + 0.25*F.mse_loss(audio_semantic_result, text_vq.detach())
        return audio_vq_forward_loss

    def Text_vq_forward(self, audio_feat, text_feat):
        audio_vq = self.Audio_VQ_Encoder(audio_feat)
        text_feat = text_feat.cuda()
        text_semantic_result = self.text_semantic_encoder(text_feat).transpose(0, 1).contiguous()
        text_semantic_result = self.text_self_att(text_semantic_result)
        text_semantic_result = text_semantic_result.transpose(0, 1).contiguous()  # [batch, length, hidden_dim]
        text_vq = self.Cross_quantizer.text_vq_embedding(text_semantic_result)
        text_vq_forward_loss = F.mse_loss(text_semantic_result, text_vq.detach()) + 0.25*F.mse_loss(text_semantic_result, audio_vq.detach())
        return text_vq_forward_loss

    
    def forward(self, audio_feat, text_feat, epoch):
        text_feat = text_feat.cuda()
        audio_feat = audio_feat.cuda()
        
        text_semantic_result = text_feat.transpose(0, 1).contiguous()
        text_semantic_result = self.text_self_att(text_semantic_result)# [length, batch, hidden_dim]
        text_semantic_result = text_semantic_result.transpose(0, 1).contiguous()  # [batch, length, hidden_dim]
        
        audio_semantic_result = audio_feat.transpose(0, 1).contiguous()
        audio_semantic_result = self.audio_self_att(audio_semantic_result)# [length, batch, hidden_dim]
        audio_semantic_result = audio_semantic_result.transpose(0, 1).contiguous()  # [batch, length, hidden_dim]

        text_encoder_result = self.Text_encoder(text_feat)  # [batch, length, audio_output_dim]
        audio_encoder_result = self.Audio_encoder(audio_feat)  # [batch, length, audio_output_dim]

        audio_vq, text_vq, audio_embedding_loss, text_embedding_loss, audio_perplexity, text_perplexity, cmcm_loss, equal_num \
            = self.Cross_quantizer(audio_semantic_result, text_semantic_result, epoch)

        return text_semantic_result, audio_semantic_result, text_encoder_result, audio_encoder_result, \
               text_vq, audio_vq, audio_embedding_loss, text_embedding_loss, cmcm_loss, equal_num


class AV_VQVAE_Encoder(nn.Module):
    def __init__(self, video_dim, audio_dim, video_output_dim, audio_output_dim, n_embeddings, embedding_dim):
        super(AV_VQVAE_Encoder, self).__init__()
        self.video_dim = video_dim
        self.audio_dim = audio_dim
        self.hidden_dim = embedding_dim
        self.Audio_encoder = Audio_Encoder(audio_dim, audio_output_dim)
        self.Video_encoder = Video_Encoder(video_dim, video_output_dim)


        self.Cross_quantizer = Cross_VQEmbeddingEMA(n_embeddings, self.hidden_dim)

        self.video_semantic_encoder = Video_Semantic_Encoder(video_dim)
        self.video_self_att = InternalTemporalRelationModule(input_dim=video_dim, d_model=self.hidden_dim)
        self.audio_self_att = InternalTemporalRelationModule(input_dim=audio_dim, d_model=self.hidden_dim)

    def Audio_VQ_Encoder(self, audio_feat):
        audio_feat = audio_feat.cuda()
        audio_semantic_result = audio_feat.transpose(0, 1).contiguous()
        audio_semantic_result = self.audio_self_att(audio_semantic_result)
        audio_semantic_result = audio_semantic_result.transpose(0, 1).contiguous()
        audio_vq = self.Cross_quantizer.Audio_vq_embedding(audio_semantic_result)
        return audio_vq

    def Video_VQ_Encoder(self, video_feat):
        video_feat = video_feat.cuda()
        video_semantic_result = self.video_semantic_encoder(video_feat).transpose(0, 1).contiguous()
        video_semantic_result = self.video_self_att(video_semantic_result)
        video_semantic_result = video_semantic_result.transpose(0, 1).contiguous()  # [batch, length, hidden_dim]
        video_vq = self.Cross_quantizer.Video_vq_embedding(video_semantic_result)
        return video_vq

    def Audio_vq_forward(self, audio_feat, video_feat):
        video_vq = self.Video_VQ_Encoder(video_feat)
        audio_feat = audio_feat.cuda()
        audio_semantic_result = audio_feat.transpose(0, 1).contiguous()
        audio_semantic_result = self.audio_self_att(audio_semantic_result)
        audio_semantic_result = audio_semantic_result.transpose(0, 1).contiguous()
        audio_vq = self.Cross_quantizer.Audio_vq_embedding(audio_semantic_result)
        audio_vq_forward_loss =  F.mse_loss(audio_semantic_result, audio_vq.detach()) + 0.25*F.mse_loss(audio_semantic_result, video_vq.detach())
        return audio_vq_forward_loss

    def Video_vq_forward(self, audio_feat, video_feat):
        audio_vq = self.Audio_VQ_Encoder(audio_feat)
        video_feat = video_feat.cuda()
        video_semantic_result = self.video_semantic_encoder(video_feat).transpose(0, 1).contiguous()
        video_semantic_result = self.video_self_att(video_semantic_result)
        video_semantic_result = video_semantic_result.transpose(0, 1).contiguous()  # [batch, length, hidden_dim]
        video_vq = self.Cross_quantizer.Video_vq_embedding(video_semantic_result)
        video_vq_forward_loss = F.mse_loss(video_semantic_result, video_vq.detach()) + 0.25*F.mse_loss(video_semantic_result, audio_vq.detach())
        return video_vq_forward_loss

    
    def forward(self, audio_feat, video_feat, epoch):
        video_feat = video_feat.cuda()
        audio_feat = audio_feat.cuda()
        video_encoder_result, video_club_feature = self.Video_encoder(
            video_feat)  # [batch, length, 3, 3, video_output_dim]

        video_semantic_result = self.video_semantic_encoder(video_feat).transpose(0, 1).contiguous()

        video_semantic_result = self.video_self_att(video_semantic_result)
        video_semantic_result = video_semantic_result.transpose(0, 1).contiguous()  # [batch, length, hidden_dim]

        audio_semantic_result = audio_feat.transpose(0, 1).contiguous()
        audio_semantic_result = self.audio_self_att(audio_semantic_result)
        audio_semantic_result = audio_semantic_result.transpose(0, 1).contiguous()  # [batch, length, hidden_dim]

        audio_encoder_result = self.Audio_encoder(audio_feat)  # [batch, length, audio_output_dim]

        audio_vq, video_vq, audio_embedding_loss, video_embedding_loss, audio_perplexity, video_perplexity, cmcm_loss, equal_num \
            = self.Cross_quantizer(audio_semantic_result, video_semantic_result, epoch)

        return video_semantic_result, audio_semantic_result, video_encoder_result, video_club_feature, audio_encoder_result, \
               video_vq, audio_vq, audio_embedding_loss, video_embedding_loss, cmcm_loss, equal_num


class AVT_VQVAE_Decoder(nn.Module):
    def __init__(self, audio_dim, video_dim, text_dim, audio_output_dim, video_output_dim, text_output_dim):
        super(AVT_VQVAE_Decoder, self).__init__()
        self.hidden_dim = 256 #embedding_dim
        self.video_dim = video_dim
        self.text_dim = text_dim
        self.audio_dim = audio_dim
        self.video_output_dim = video_output_dim
        self.text_output_dim = text_output_dim
        self.audio_output_dim = audio_output_dim
        self.Video_decoder = Video_Decoder(video_output_dim, video_dim, self.hidden_dim)
        self.Audio_decoder = Audio_Decoder(audio_output_dim, audio_dim, self.hidden_dim)
        self.Text_decoder = Text_Decoder(text_output_dim, text_dim, self.hidden_dim)
        self.video_semantic_decoder = Semantic_Decoder(self.hidden_dim, class_num=141)
        self.text_semantic_decoder = Semantic_Decoder(self.hidden_dim, class_num=141)
        self.audio_semantic_decoder = Semantic_Decoder(self.hidden_dim, class_num=141)

    def forward(self, audio_feat, video_feat, text_feat, audio_encoder_result, video_encoder_result, text_encoder_result, audio_vq, video_vq, text_vq):
        video_feat = video_feat.cuda()
        text_feat = text_feat.cuda()
        audio_feat = audio_feat.cuda()
        video_recon_result = self.Video_decoder(video_encoder_result, video_vq)
        text_recon_result = self.Text_decoder(text_encoder_result, text_vq)
        audio_recon_result = self.Audio_decoder(audio_encoder_result, audio_vq)
        video_recon_loss = F.mse_loss(video_recon_result, video_feat)
        text_recon_loss = F.mse_loss(text_recon_result, text_feat)
        audio_recon_loss = F.mse_loss(audio_recon_result, audio_feat)
        video_class = self.video_semantic_decoder(video_vq)
        text_class = self.text_semantic_decoder(text_vq)
        audio_class = self.audio_semantic_decoder(audio_vq)

        return audio_recon_loss, video_recon_loss, text_recon_loss, audio_class, video_class, text_class

class AT_VQVAE_Decoder(nn.Module):
    def __init__(self, text_dim, audio_dim, text_output_dim, audio_output_dim):
        super(AT_VQVAE_Decoder, self).__init__()
        self.hidden_dim = 256#embedding_dim
        self.text_dim = text_dim
        self.audio_dim = audio_dim
        self.text_output_dim = text_output_dim
        self.audio_output_dim = audio_output_dim
        self.Audio_decoder = Audio_Decoder(audio_output_dim, audio_dim, self.hidden_dim)
        self.Text_decoder = Text_Decoder(text_output_dim, text_dim, self.hidden_dim)
        self.text_semantic_decoder = Semantic_Decoder(self.hidden_dim, class_num=141)
        self.audio_semantic_decoder = Semantic_Decoder(self.hidden_dim, class_num=141)

    def forward(self, text_feat, audio_feat, text_encoder_result, audio_encoder_result, text_vq, audio_vq):
        text_feat = text_feat.cuda()
        audio_feat = audio_feat.cuda()
        text_recon_result = self.Text_decoder(text_encoder_result, text_vq)
        audio_recon_result = self.Audio_decoder(audio_encoder_result, audio_vq)
        text_recon_loss = F.mse_loss(text_recon_result, text_feat)
        audio_recon_loss = F.mse_loss(audio_recon_result, audio_feat)
        text_class = self.text_semantic_decoder(text_vq)
        audio_class = self.audio_semantic_decoder(audio_vq)

        return text_recon_loss, audio_recon_loss, text_class, audio_class

class AV_VQVAE_Decoder(nn.Module):
    def __init__(self, video_dim, audio_dim, video_output_dim, audio_output_dim):
        super(AV_VQVAE_Decoder, self).__init__()
        self.hidden_dim = 256#embedding_dim
        self.video_dim = video_dim
        self.audio_dim = audio_dim
        self.video_output_dim = video_output_dim
        self.audio_output_dim = audio_output_dim
        self.Audio_decoder = Audio_Decoder(audio_output_dim, audio_dim, self.hidden_dim)
        self.Video_decoder = Video_Decoder(video_output_dim, video_dim, self.hidden_dim)
        self.video_semantic_decoder = Semantic_Decoder(self.hidden_dim, class_num=141)
        self.audio_semantic_decoder = Semantic_Decoder(self.hidden_dim, class_num=141)

    def forward(self, video_feat, audio_feat, video_encoder_result, audio_encoder_result, video_vq, audio_vq):
        video_feat = video_feat.cuda()
        audio_feat = audio_feat.cuda()
        video_recon_result = self.Video_decoder(video_encoder_result, video_vq)
        audio_recon_result = self.Audio_decoder(audio_encoder_result, audio_vq)
        video_recon_loss = F.mse_loss(video_recon_result, video_feat)
        audio_recon_loss = F.mse_loss(audio_recon_result, audio_feat)
        video_class = self.video_semantic_decoder(video_vq)
        audio_class = self.audio_semantic_decoder(audio_vq)

        return video_recon_loss, audio_recon_loss, video_class, audio_class

class Cross_VQEmbeddingEMA_AVT(nn.Module):
    def __init__(self, n_embeddings, embedding_dim, commitment_cost=0.25, decay=0.99, epsilon=1e-5):
        super(Cross_VQEmbeddingEMA_AVT, self).__init__()
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon

        init_bound = 1 / 400
        embedding = torch.Tensor(n_embeddings, embedding_dim)
        
        embedding.uniform_(-init_bound, init_bound)
        self.register_buffer("embedding", embedding)
        self.register_buffer("ema_count", torch.zeros(n_embeddings))
        self.register_buffer("ema_weight", self.embedding.clone())
        self.register_buffer("unactivated_count", -torch.ones(n_embeddings))# unactivated:-1


    def Audio_vq_embedding(self, audio_semantic):

        B, T, D = audio_semantic.size()
        a_flat = audio_semantic.detach().reshape(-1, D)  # [B, T, D] -> [BxT, D]
        a_distances = torch.addmm(torch.sum(self.embedding ** 2, dim=1) +
                                  torch.sum(a_flat ** 2, dim=1, keepdim=True),
                                  a_flat, self.embedding.t(),
                                  alpha=-2.0, beta=1.0)  # [BxT, M]
        a_indices = torch.argmin(a_distances.double(), dim=-1)  # [BxT,1]
        a_quantized = F.embedding(a_indices, self.embedding)  
        a_quantized = a_quantized.view_as(audio_semantic)  # [BxT,D]->[B,T,D]
        a_quantized = audio_semantic + (a_quantized - audio_semantic).detach()
        return a_quantized

    def Text_vq_embedding(self, text_semantic):
        B, T, D = text_semantic.size()
        t_flat = text_semantic.detach().reshape(-1, D)
        t_distance = torch.addmm(torch.sum(self.embedding ** 2, dim=1) +
                                 torch.sum(t_flat**2, dim=1, keepdim=True),
                                 t_flat, self.embedding.t(),
                                 alpha=-2.0, beta=1.0)
        t_indices = torch.argmin(t_distance.double(), dim=-1)
        t_quantized = F.embedding(t_indices, self.embedding)
        t_quantized = t_quantized.view_as(text_semantic)
        t_quantized = text_semantic + (t_quantized - text_semantic).detach()
        return t_quantized

    def Video_vq_embedding(self, video_semantic):

        B, T, D = video_semantic.size()
        v_flat = video_semantic.detach().reshape(-1, D)
        v_distance = torch.addmm(torch.sum(self.embedding ** 2, dim=1) +
                                 torch.sum(v_flat**2, dim=1, keepdim=True),
                                 v_flat, self.embedding.t(),
                                 alpha=-2.0, beta=1.0)
        v_indices = torch.argmin(v_distance.double(), dim=-1)
        v_quantized = F.embedding(v_indices, self.embedding)
        v_quantized = v_quantized.view_as(video_semantic)
        v_quantized = video_semantic + (v_quantized - video_semantic).detach()
        return v_quantized

    def forward(self, audio_semantic, video_semantic, text_semantic, epoch):
        M, D = self.embedding.size()
        B, T, D = audio_semantic.size()
        a_flat = audio_semantic.detach().reshape(-1, D)  # [B, T, D] -> [BxT, D]
        v_flat = video_semantic.detach().reshape(-1, D)  # [B, T, D] -> [BxT, D]
        t_flat = text_semantic.detach().reshape(-1, D)  # [B, T, D] -> [BxT, D]

        # M:512  B:batchsize  T:10
        # b * mat + a * (mat1@mat2) ([M,] + [BxT,1]) - 2*([BxT,D]@[D,M]) =
        a_distances = torch.addmm(torch.sum(self.embedding ** 2, dim=1) +
                                  torch.sum(a_flat ** 2, dim=1, keepdim=True),
                                  a_flat, self.embedding.t(),
                                  alpha=-2.0, beta=1.0)  # [BxT, M]

        v_distances = torch.addmm(torch.sum(self.embedding ** 2, dim=1) +
                                  torch.sum(v_flat ** 2, dim=1, keepdim=True),
                                  v_flat, self.embedding.t(),
                                  alpha=-2.0, beta=1.0)  # [BxT, M]
        
        t_distances = torch.addmm(torch.sum(self.embedding ** 2, dim=1) +
                                  torch.sum(t_flat ** 2, dim=1, keepdim=True),
                                  t_flat, self.embedding.t(),
                                  alpha=-2.0, beta=1.0)  # [BxT, M]

        a_distances_gradient = torch.addmm(torch.sum(self.embedding ** 2, dim=1) +
                                  torch.sum(audio_semantic.reshape(-1, D) ** 2, dim=1, keepdim=True),
                                  audio_semantic.reshape(-1, D), self.embedding.t(),
                                  alpha=-2.0, beta=1.0)  # [BxT, M]

        v_distances_gradient = torch.addmm(torch.sum(self.embedding ** 2, dim=1) +
                                           torch.sum(video_semantic.reshape(-1, D) ** 2, dim=1, keepdim=True),
                                           video_semantic.reshape(-1, D), self.embedding.t(),
                                           alpha=-2.0, beta=1.0)  # [BxT, M]
        
        t_distances_gradient = torch.addmm(torch.sum(self.embedding ** 2, dim=1) +
                                           torch.sum(text_semantic.reshape(-1, D) ** 2, dim=1, keepdim=True),
                                           text_semantic.reshape(-1, D), self.embedding.t(),
                                           alpha=-2.0, beta=1.0)  # [BxT, M]

        a_ph = F.softmax(-torch.sqrt(a_distances_gradient), dim=1)  # [BxT, M] torch.Size([160, 512])
        v_ph = F.softmax(-torch.sqrt(v_distances_gradient), dim=1)  # [BxT, M] torch.Size([160, 512])
        t_ph = F.softmax(-torch.sqrt(t_distances_gradient), dim=1)  # [BxT, M] torch.Size([160, 512])

        a_ph = torch.reshape(a_ph, ((B, T, M)))  # [BxT, M] -> [B, T, M]
        a_pH = torch.mean(a_ph, dim=1)  # [B, T, M] -> [B, M]
        v_ph = torch.reshape(v_ph, ((B, T, M)))  # [BxT, M] -> [B, T, M]
        v_pH = torch.mean(v_ph, dim=1)  # [B, T, M] -> [B, M]
        t_ph = torch.reshape(t_ph, ((B, T, M)))  # [BxT, M] -> [B, T, M]
        t_pH = torch.mean(t_ph, dim=1)  # [B, T, M] -> [B, M]

        Scode_av = a_pH @ torch.log(v_pH.t() + 1e-10) + v_pH @ torch.log(a_pH.t() + 1e-10)
        Scode_at = a_pH @ torch.log(t_pH.t() + 1e-10) + t_pH @ torch.log(a_pH.t() + 1e-10)
        Scode_tv = t_pH @ torch.log(v_pH.t() + 1e-10) + v_pH @ torch.log(t_pH.t() + 1e-10)

        # caculate Lcmcm
        # If the numerical values in the calculation process of exp are too large, 
        # you can add a logC to each item in the matrix, where logC = -Scode.
        MaxScode_av = torch.max(-Scode_av)
        EScode_av = torch.exp(Scode_av + MaxScode_av)
        
        MaxScode_at = torch.max(-Scode_at)
        EScode_at = torch.exp(Scode_at + MaxScode_at)
        
        MaxScode_tv = torch.max(-Scode_tv)
        EScode_tv = torch.exp(Scode_tv + MaxScode_tv)

        EScode_sumdim1_av = torch.sum(EScode_av, dim=1)
        Lcmcm_av = 0
        
        EScode_sumdim1_at = torch.sum(EScode_at, dim=1)
        Lcmcm_at = 0
        
        EScode_sumdim1_tv = torch.sum(EScode_tv, dim=1)
        Lcmcm_tv = 0
        
        for i in range(B):
            Lcmcm_av -= torch.log(EScode_av[i, i] / (EScode_sumdim1_av[i] + self.epsilon))
            Lcmcm_at -= torch.log(EScode_at[i, i] / (EScode_sumdim1_at[i] + self.epsilon))
            Lcmcm_tv -= torch.log(EScode_tv[i, i] / (EScode_sumdim1_tv[i] + self.epsilon))
            
            
        Lcmcm_av /= B
        Lcmcm_at /= B
        Lcmcm_tv /= B

        a_indices = torch.argmin(a_distances.double(), dim=-1)  # [BxT,1]
        a_encodings = F.one_hot(a_indices, M).double()  # [BxT, M]
        a_quantized = F.embedding(a_indices, self.embedding)  
        a_quantized = a_quantized.view_as(audio_semantic)  # [BxT,D]->[B,T,D]

        v_indices = torch.argmin(v_distances.double(), dim=-1)  # [BxT,1]
        v_encodings = F.one_hot(v_indices, M).double()  # [BxT, M]
        v_quantized = F.embedding(v_indices, self.embedding)  
        v_quantized = v_quantized.view_as(video_semantic)  # [BxT,D]->[B,T,D]
        
        t_indices = torch.argmin(t_distances.double(), dim=-1)  # [BxT,1]
        t_encodings = F.one_hot(t_indices, M).double()  # [BxT, M]
        t_quantized = F.embedding(t_indices, self.embedding)  
        t_quantized = t_quantized.view_as(text_semantic)  # [BxT,D]->[B,T,D]


        if True:
            a_indices_reshape = a_indices.reshape(B, T)
            v_indices_reshape = v_indices.reshape(B, T)
            t_indices_reshape = t_indices.reshape(B, T)
            a_indices_mode = torch.mode(a_indices_reshape, dim=-1, keepdim=False)
            v_indices_mode = torch.mode(v_indices_reshape, dim=-1, keepdim=False)
            t_indices_mode = torch.mode(t_indices_reshape, dim=-1, keepdim=False)

            equal_item = (a_indices_mode.values == v_indices_mode.values) & (a_indices_mode.values == t_indices_mode.values)
            equal_num = equal_item.sum()
            
        if self.training:
            # audio
            self.ema_count = self.decay * self.ema_count + (1 - self.decay) * torch.sum(a_encodings, dim=0)
            n = torch.sum(self.ema_count)
            self.ema_count = (self.ema_count + self.epsilon) / (n + M * self.epsilon) * n
            a_dw = torch.matmul(a_encodings.t(), a_flat)
            av_dw = torch.matmul(a_encodings.t(), v_flat)
            at_dw = torch.matmul(a_encodings.t(), t_flat)

            self.ema_weight = self.decay * self.ema_weight + 0.5*(1 - self.decay) * a_dw + 0.25*(1 - self.decay) * av_dw + 0.25*(1 - self.decay) * at_dw
            self.embedding = self.ema_weight / self.ema_count.unsqueeze(-1)

            # video
            self.ema_count = self.decay * self.ema_count + (1 - self.decay) * torch.sum(v_encodings, dim=0)
            n = torch.sum(self.ema_count)
            self.ema_count = (self.ema_count + self.epsilon) / (n + M * self.epsilon) * n
            v_dw = torch.matmul(v_encodings.t(), v_flat)
            va_dw = torch.matmul(v_encodings.t(), a_flat)
            vt_dw = torch.matmul(v_encodings.t(), t_flat)

            self.ema_weight = self.decay * self.ema_weight + 0.5*(1 - self.decay) * v_dw + 0.25*(1 - self.decay) * va_dw + 0.25*(1 - self.decay) * vt_dw
            self.embedding = self.ema_weight / self.ema_count.unsqueeze(-1)
            
            # text
            self.ema_count = self.decay * self.ema_count + (1 - self.decay) * torch.sum(t_encodings, dim=0)
            n = torch.sum(self.ema_count)
            self.ema_count = (self.ema_count + self.epsilon) / (n + M * self.epsilon) * n
            t_dw = torch.matmul(t_encodings.t(), t_flat)
            ta_dw = torch.matmul(t_encodings.t(), a_flat)
            tv_dw = torch.matmul(t_encodings.t(), v_flat)
            
            self.ema_weight = self.decay * self.ema_weight + 0.5*(1 - self.decay) * t_dw + 0.25*(1 - self.decay) * ta_dw + 0.25*(1 - self.decay) * tv_dw
            self.embedding = self.ema_weight / self.ema_count.unsqueeze(-1)

        self.unactivated_count = self.unactivated_count + 1
        for indice in a_indices:
            self.unactivated_count[indice.item()] = 0
        for indice in v_indices:
            self.unactivated_count[indice.item()] = 0
        for indice in t_indices:
            self.unactivated_count[indice.item()] = 0
        activated_indices = []
        unactivated_indices = []
        for i, x in enumerate(self.unactivated_count):
            if x > 300:
                unactivated_indices.append(i)
                self.unactivated_count[i] = 0
            elif x >= 0 and x < 100:
                activated_indices.append(i)
        activated_quantized = F.embedding(torch.tensor(activated_indices,dtype=torch.int32).cuda(), self.embedding)
        for i in unactivated_indices:
            self.embedding[i] = activated_quantized[random.randint(0,len(activated_indices)-1)] + torch.Tensor(256).uniform_(-1/1024, 1/1024).cuda()

        cmcm_loss = 0.5 * (Lcmcm_av + Lcmcm_at + Lcmcm_tv)

        a_e_latent_loss = F.mse_loss(audio_semantic, a_quantized.detach())
        av_e_latent_loss = F.mse_loss(audio_semantic, v_quantized.detach())
        at_e_latent_loss = F.mse_loss(audio_semantic, t_quantized.detach())
        #a_loss = self.commitment_cost * 1.0 * a_e_latent_loss
        a_loss = self.commitment_cost * 2.0 * a_e_latent_loss + 0.5*self.commitment_cost * av_e_latent_loss + 0.5*self.commitment_cost * at_e_latent_loss
        
        v_e_latent_loss = F.mse_loss(video_semantic, v_quantized.detach())
        va_e_latent_loss = F.mse_loss(video_semantic, a_quantized.detach())
        vt_e_latent_loss = F.mse_loss(video_semantic, t_quantized.detach())
        #v_loss = self.commitment_cost * 1.0 * v_e_latent_loss
        v_loss = self.commitment_cost * 2.0 * v_e_latent_loss + 0.5*self.commitment_cost * va_e_latent_loss + 0.5*self.commitment_cost * vt_e_latent_loss
        
        t_e_latent_loss = F.mse_loss(text_semantic, t_quantized.detach())
        ta_e_latent_loss = F.mse_loss(text_semantic, a_quantized.detach())
        tv_e_latent_loss = F.mse_loss(text_semantic, v_quantized.detach())
        #v_loss = self.commitment_cost * 1.0 * v_e_latent_loss
        t_loss = self.commitment_cost * 2.0 * t_e_latent_loss + 0.5*self.commitment_cost * ta_e_latent_loss + 0.5*self.commitment_cost * tv_e_latent_loss

        a_quantized = audio_semantic + (a_quantized - audio_semantic).detach()
        v_quantized = video_semantic + (v_quantized - video_semantic).detach()
        t_quantized = text_semantic + (t_quantized - text_semantic).detach()

        a_avg_probs = torch.mean(a_encodings, dim=0)
        a_perplexity = torch.exp(-torch.sum(a_avg_probs * torch.log(a_avg_probs + 1e-10)))
        v_avg_probs = torch.mean(v_encodings, dim=0)
        v_perplexity = torch.exp(-torch.sum(v_avg_probs * torch.log(v_avg_probs + 1e-10)))
        t_avg_probs = torch.mean(t_encodings, dim=0)
        t_perplexity = torch.exp(-torch.sum(t_avg_probs * torch.log(t_avg_probs + 1e-10)))
        
        return a_quantized, v_quantized, t_quantized, a_loss, v_loss, t_loss, a_perplexity, v_perplexity, t_perplexity, cmcm_loss, equal_num

class Cross_VQEmbeddingEMA_AT(nn.Module):
    def __init__(self, n_embeddings, embedding_dim, commitment_cost=0.25, decay=0.99, epsilon=1e-5):
        super(Cross_VQEmbeddingEMA_AT, self).__init__()
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon

        init_bound = 1 / 400
        embedding = torch.Tensor(n_embeddings, embedding_dim)
        
        embedding.uniform_(-init_bound, init_bound)
        self.register_buffer("embedding", embedding)
        self.register_buffer("ema_count", torch.zeros(n_embeddings))
        self.register_buffer("ema_weight", self.embedding.clone())
        self.register_buffer("unactivated_count", -torch.ones(n_embeddings))


    def Audio_vq_embedding(self, audio_semantic):

        B, T, D = audio_semantic.size()
        a_flat = audio_semantic.detach().reshape(-1, D)  # [B, T, D] -> [BxT, D]
        a_distances = torch.addmm(torch.sum(self.embedding ** 2, dim=1) +
                                  torch.sum(a_flat ** 2, dim=1, keepdim=True),
                                  a_flat, self.embedding.t(),
                                  alpha=-2.0, beta=1.0)  # [BxT, M]
        a_indices = torch.argmin(a_distances.double(), dim=-1)  # [BxT,1]
        a_quantized = F.embedding(a_indices, self.embedding)  
        a_quantized = a_quantized.view_as(audio_semantic)  # [BxT,D]->[B,T,D]
        a_quantized = audio_semantic + (a_quantized - audio_semantic).detach()
        return a_quantized

    def Text_vq_embedding(self, text_semantic):
        B, T, D = text_semantic.size()
        t_flat = text_semantic.detach().reshape(-1, D)
        t_distance = torch.addmm(torch.sum(self.embedding ** 2, dim=1) +
                                 torch.sum(t_flat**2, dim=1, keepdim=True),
                                 t_flat, self.embedding.t(),
                                 alpha=-2.0, beta=1.0)
        t_indices = torch.argmin(t_distance.double(), dim=-1)
        t_quantized = F.embedding(t_indices, self.embedding)
        t_quantized = t_quantized.view_as(text_semantic)
        t_quantized = text_semantic + (t_quantized - text_semantic).detach()
        return t_quantized


    def forward(self, audio_semantic, video_semantic, epoch):
        M, D = self.embedding.size()
        B, T, D = audio_semantic.size()
        # x_flat = x.detach().reshape(-1, D) #x:[B,T,D]->x_flat:[BxT,D]
        a_flat = audio_semantic.detach().reshape(-1, D)  # [B, T, D] -> [BxT, D]
        v_flat = video_semantic.detach().reshape(-1, D)  # [B, T, D] -> [BxT, D]

        a_distances = torch.addmm(torch.sum(self.embedding ** 2, dim=1) +
                                  torch.sum(a_flat ** 2, dim=1, keepdim=True),
                                  a_flat, self.embedding.t(),
                                  alpha=-2.0, beta=1.0)  # [BxT, M]

        v_distances = torch.addmm(torch.sum(self.embedding ** 2, dim=1) +
                                  torch.sum(v_flat ** 2, dim=1, keepdim=True),
                                  v_flat, self.embedding.t(),
                                  alpha=-2.0, beta=1.0)  # [BxT, M]

        a_distances_gradient = torch.addmm(torch.sum(self.embedding ** 2, dim=1) +
                                  torch.sum(audio_semantic.reshape(-1, D) ** 2, dim=1, keepdim=True),
                                  audio_semantic.reshape(-1, D), self.embedding.t(),
                                  alpha=-2.0, beta=1.0)  # [BxT, M]

        v_distances_gradient = torch.addmm(torch.sum(self.embedding ** 2, dim=1) +
                                           torch.sum(video_semantic.reshape(-1, D) ** 2, dim=1, keepdim=True),
                                           video_semantic.reshape(-1, D), self.embedding.t(),
                                           alpha=-2.0, beta=1.0)  # [BxT, M]

        a_ph = F.softmax(-torch.sqrt(a_distances_gradient), dim=1)  # [BxT, M] torch.Size([160, 512])
        v_ph = F.softmax(-torch.sqrt(v_distances_gradient), dim=1)  # [BxT, M] torch.Size([160, 512])
        a_ph = torch.reshape(a_ph, ((B, T, M)))  # [BxT, M] -> [B, T, M]
        a_pH = torch.mean(a_ph, dim=1)  # [B, T, M] -> [B, M]
        v_ph = torch.reshape(v_ph, ((B, T, M)))  # [BxT, M] -> [B, T, M]
        v_pH = torch.mean(v_ph, dim=1)  # [B, T, M] -> [B, M]

        Scode = a_pH @ torch.log(v_pH.t() + 1e-10) + v_pH @ torch.log(a_pH.t() + 1e-10)

        MaxScode = torch.max(-Scode)
        EScode = torch.exp(Scode + MaxScode)

        EScode_sumdim1 = torch.sum(EScode, dim=1)
        Lcmcm = 0
        for i in range(B):
            Lcmcm -= torch.log(EScode[i, i] / (EScode_sumdim1[i] + self.epsilon))
        Lcmcm /= B


        a_indices = torch.argmin(a_distances.double(), dim=-1)  # [BxT,1]
        a_encodings = F.one_hot(a_indices, M).double()  # [BxT, M]
        a_quantized = F.embedding(a_indices, self.embedding)  
        a_quantized = a_quantized.view_as(audio_semantic)  # [BxT,D]->[B,T,D]

        v_indices = torch.argmin(v_distances.double(), dim=-1)  # [BxT,1]
        v_encodings = F.one_hot(v_indices, M).double()  # [BxT, M]
        v_quantized = F.embedding(v_indices, self.embedding)  
        v_quantized = v_quantized.view_as(video_semantic)  # [BxT,D]->[B,T,D]


        if True:
            a_indices_reshape = a_indices.reshape(B, T)
            v_indices_reshape = v_indices.reshape(B, T)
            a_indices_mode = torch.mode(a_indices_reshape, dim=-1, keepdim=False)
            v_indices_mode = torch.mode(v_indices_reshape, dim=-1, keepdim=False)

            equal_item = (a_indices_mode.values == v_indices_mode.values)
            equal_num = equal_item.sum()
            

        if self.training:
            # audio
            self.ema_count = self.decay * self.ema_count + (1 - self.decay) * torch.sum(a_encodings, dim=0)
            n = torch.sum(self.ema_count)
            self.ema_count = (self.ema_count + self.epsilon) / (n + M * self.epsilon) * n
            a_dw = torch.matmul(a_encodings.t(), a_flat)
            # ********************************************************
            av_dw = torch.matmul(a_encodings.t(), v_flat)
            # ********************************************************

            self.ema_weight = self.decay * self.ema_weight + 0.5*(1 - self.decay) * a_dw + 0.5*(1 - self.decay) * av_dw
            self.embedding = self.ema_weight / self.ema_count.unsqueeze(-1)

            # video
            self.ema_count = self.decay * self.ema_count + (1 - self.decay) * torch.sum(v_encodings, dim=0)
            n = torch.sum(self.ema_count)
            self.ema_count = (self.ema_count + self.epsilon) / (n + M * self.epsilon) * n
            v_dw = torch.matmul(v_encodings.t(), v_flat)
            # ********************************************************
            va_dw = torch.matmul(v_encodings.t(), a_flat)
            # ********************************************************
            self.ema_weight = self.decay * self.ema_weight + 0.5*(1 - self.decay) * v_dw + 0.5*(1 - self.decay) * va_dw
            self.embedding = self.ema_weight / self.ema_count.unsqueeze(-1)

        
        self.unactivated_count += 1
        for indice in a_indices:
            self.unactivated_count[indice.item()] = 0
        for indice in v_indices:
            self.unactivated_count[indice.item()] = 0
        activated_indices = []
        unactivated_indices = []
        for i, x in enumerate(self.unactivated_count):
            if x > 300:
                unactivated_indices.append(i)
                self.unactivated_count[i] = 0
            elif x >= 0 and x < 100:
                activated_indices.append(i)
        activated_quantized = F.embedding(torch.tensor(activated_indices,dtype=torch.int32).cuda(), self.embedding)
        for i in unactivated_indices:
            self.embedding[i] = activated_quantized[random.randint(0,len(activated_indices)-1)] + torch.Tensor(256).uniform_(-1/1024, 1/1024).cuda()

        cmcm_loss = 0.5 * Lcmcm  

        a_e_latent_loss = F.mse_loss(audio_semantic, a_quantized.detach())
        av_e_latent_loss = F.mse_loss(audio_semantic, v_quantized.detach())
        #a_loss = self.commitment_cost * 1.0 * a_e_latent_loss
        a_loss = self.commitment_cost * 2.0 * a_e_latent_loss + self.commitment_cost * av_e_latent_loss
        v_e_latent_loss = F.mse_loss(video_semantic, v_quantized.detach())
        va_e_latent_loss = F.mse_loss(video_semantic, a_quantized.detach())
        #v_loss = self.commitment_cost * 1.0 * v_e_latent_loss
        v_loss = self.commitment_cost * 2.0 * v_e_latent_loss + self.commitment_cost * va_e_latent_loss

        a_quantized = audio_semantic + (a_quantized - audio_semantic).detach()
        v_quantized = video_semantic + (v_quantized - video_semantic).detach()

        a_avg_probs = torch.mean(a_encodings, dim=0)
        a_perplexity = torch.exp(-torch.sum(a_avg_probs * torch.log(a_avg_probs + 1e-10)))
        v_avg_probs = torch.mean(v_encodings, dim=0)
        v_perplexity = torch.exp(-torch.sum(v_avg_probs * torch.log(v_avg_probs + 1e-10)))
        return a_quantized, v_quantized, a_loss, v_loss, a_perplexity, v_perplexity, cmcm_loss, equal_num


class Cross_VQEmbeddingEMA(nn.Module):
    def __init__(self, n_embeddings, embedding_dim, commitment_cost=0.25, decay=0.99, epsilon=1e-5):
        super(Cross_VQEmbeddingEMA, self).__init__()
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon

        init_bound = 1 / 400
        embedding = torch.Tensor(n_embeddings, embedding_dim)
        
        embedding.uniform_(-init_bound, init_bound)
        self.register_buffer("embedding", embedding)
        self.register_buffer("ema_count", torch.zeros(n_embeddings))
        self.register_buffer("ema_weight", self.embedding.clone())
        self.register_buffer("unactivated_count", -torch.ones(n_embeddings))


    def Audio_vq_embedding(self, audio_semantic):

        B, T, D = audio_semantic.size()
        a_flat = audio_semantic.detach().reshape(-1, D)  # [B, T, D] -> [BxT, D]
        a_distances = torch.addmm(torch.sum(self.embedding ** 2, dim=1) +
                                  torch.sum(a_flat ** 2, dim=1, keepdim=True),
                                  a_flat, self.embedding.t(),
                                  alpha=-2.0, beta=1.0)  # [BxT, M]
        a_indices = torch.argmin(a_distances.double(), dim=-1)  # [BxT,1]
        a_quantized = F.embedding(a_indices, self.embedding) 
        a_quantized = a_quantized.view_as(audio_semantic)  # [BxT,D]->[B,T,D]
        a_quantized = audio_semantic + (a_quantized - audio_semantic).detach()
        return a_quantized

    def Video_vq_embedding(self, video_semantic):

        B, T, D = video_semantic.size()
        v_flat = video_semantic.detach().reshape(-1, D)
        v_distance = torch.addmm(torch.sum(self.embedding ** 2, dim=1) +
                                 torch.sum(v_flat**2, dim=1, keepdim=True),
                                 v_flat, self.embedding.t(),
                                 alpha=-2.0, beta=1.0)
        v_indices = torch.argmin(v_distance.double(), dim=-1)
        v_quantized = F.embedding(v_indices, self.embedding)
        v_quantized = v_quantized.view_as(video_semantic)
        v_quantized = video_semantic + (v_quantized - video_semantic).detach()
        return v_quantized


    def forward(self, audio_semantic, video_semantic, epoch):
        M, D = self.embedding.size()
        B, T, D = audio_semantic.size()
        # x_flat = x.detach().reshape(-1, D) #x:[B,T,D]->x_flat:[BxT,D]
        a_flat = audio_semantic.detach().reshape(-1, D)  # [B, T, D] -> [BxT, D]
        v_flat = video_semantic.detach().reshape(-1, D)  # [B, T, D] -> [BxT, D]


        a_distances = torch.addmm(torch.sum(self.embedding ** 2, dim=1) +
                                  torch.sum(a_flat ** 2, dim=1, keepdim=True),
                                  a_flat, self.embedding.t(),
                                  alpha=-2.0, beta=1.0)  # [BxT, M]

        v_distances = torch.addmm(torch.sum(self.embedding ** 2, dim=1) +
                                  torch.sum(v_flat ** 2, dim=1, keepdim=True),
                                  v_flat, self.embedding.t(),
                                  alpha=-2.0, beta=1.0)  # [BxT, M]

        a_distances_gradient = torch.addmm(torch.sum(self.embedding ** 2, dim=1) +
                                  torch.sum(audio_semantic.reshape(-1, D) ** 2, dim=1, keepdim=True),
                                  audio_semantic.reshape(-1, D), self.embedding.t(),
                                  alpha=-2.0, beta=1.0)  # [BxT, M]

        v_distances_gradient = torch.addmm(torch.sum(self.embedding ** 2, dim=1) +
                                           torch.sum(video_semantic.reshape(-1, D) ** 2, dim=1, keepdim=True),
                                           video_semantic.reshape(-1, D), self.embedding.t(),
                                           alpha=-2.0, beta=1.0)  # [BxT, M]

        a_ph = F.softmax(-torch.sqrt(a_distances_gradient), dim=1)  # [BxT, M] torch.Size([160, 512])
        v_ph = F.softmax(-torch.sqrt(v_distances_gradient), dim=1)  # [BxT, M] torch.Size([160, 512])
        a_ph = torch.reshape(a_ph, ((B, T, M)))  # [BxT, M] -> [B, T, M]
        a_pH = torch.mean(a_ph, dim=1)  # [B, T, M] -> [B, M]
        v_ph = torch.reshape(v_ph, ((B, T, M)))  # [BxT, M] -> [B, T, M]
        v_pH = torch.mean(v_ph, dim=1)  # [B, T, M] -> [B, M]

        Scode = a_pH @ torch.log(v_pH.t() + 1e-10) + v_pH @ torch.log(a_pH.t() + 1e-10)

        MaxScode = torch.max(-Scode)
        EScode = torch.exp(Scode + MaxScode)

        EScode_sumdim1 = torch.sum(EScode, dim=1)
        Lcmcm = 0
        for i in range(B):
            Lcmcm -= torch.log(EScode[i, i] / (EScode_sumdim1[i] + self.epsilon))
        Lcmcm /= B

        a_indices = torch.argmin(a_distances.double(), dim=-1)  # [BxT,1]
        a_encodings = F.one_hot(a_indices, M).double()  # [BxT, M]
        a_quantized = F.embedding(a_indices, self.embedding)  
        a_quantized = a_quantized.view_as(audio_semantic)  # [BxT,D]->[B,T,D]

        v_indices = torch.argmin(v_distances.double(), dim=-1)  # [BxT,1]
        v_encodings = F.one_hot(v_indices, M).double()  # [BxT, M]
        v_quantized = F.embedding(v_indices, self.embedding)  
        v_quantized = v_quantized.view_as(video_semantic)  # [BxT,D]->[B,T,D]


        if True:
            a_indices_reshape = a_indices.reshape(B, T)
            v_indices_reshape = v_indices.reshape(B, T)
            a_indices_mode = torch.mode(a_indices_reshape, dim=-1, keepdim=False)
            v_indices_mode = torch.mode(v_indices_reshape, dim=-1, keepdim=False)

            equal_item = (a_indices_mode.values == v_indices_mode.values)
            equal_num = equal_item.sum()

        if self.training:
            # audio
            self.ema_count = self.decay * self.ema_count + (1 - self.decay) * torch.sum(a_encodings, dim=0)
            n = torch.sum(self.ema_count)
            self.ema_count = (self.ema_count + self.epsilon) / (n + M * self.epsilon) * n
            a_dw = torch.matmul(a_encodings.t(), a_flat)
            # ********************************************************
            av_dw = torch.matmul(a_encodings.t(), v_flat)
            # ********************************************************

            self.ema_weight = self.decay * self.ema_weight + 0.5*(1 - self.decay) * a_dw + 0.5*(1 - self.decay) * av_dw
            self.embedding = self.ema_weight / self.ema_count.unsqueeze(-1)

            # video
            self.ema_count = self.decay * self.ema_count + (1 - self.decay) * torch.sum(v_encodings, dim=0)
            n = torch.sum(self.ema_count)
            self.ema_count = (self.ema_count + self.epsilon) / (n + M * self.epsilon) * n
            v_dw = torch.matmul(v_encodings.t(), v_flat)
            # ********************************************************
            va_dw = torch.matmul(v_encodings.t(), a_flat)
            # ********************************************************
            self.ema_weight = self.decay * self.ema_weight + 0.5*(1 - self.decay) * v_dw + 0.5*(1 - self.decay) * va_dw
            self.embedding = self.ema_weight / self.ema_count.unsqueeze(-1)

        
        self.unactivated_count += 1
        for indice in a_indices:
            self.unactivated_count[indice.item()] = 0
        for indice in v_indices:
            self.unactivated_count[indice.item()] = 0
        activated_indices = []
        unactivated_indices = []
        for i, x in enumerate(self.unactivated_count):
            if x > 300:
                unactivated_indices.append(i)
                self.unactivated_count[i] = 0
            elif x >= 0 and x < 100:
                activated_indices.append(i)
        activated_quantized = F.embedding(torch.tensor(activated_indices,dtype=torch.int32).cuda(), self.embedding)
        for i in unactivated_indices:
            self.embedding[i] = activated_quantized[random.randint(0,len(activated_indices)-1)] + torch.Tensor(256).uniform_(-1/1024, 1/1024).cuda()


        cmcm_loss = 0.5 * Lcmcm  

        a_e_latent_loss = F.mse_loss(audio_semantic, a_quantized.detach())
        av_e_latent_loss = F.mse_loss(audio_semantic, v_quantized.detach())
        #a_loss = self.commitment_cost * 1.0 * a_e_latent_loss
        a_loss = self.commitment_cost * 2.0 * a_e_latent_loss + self.commitment_cost * av_e_latent_loss
        v_e_latent_loss = F.mse_loss(video_semantic, v_quantized.detach())
        va_e_latent_loss = F.mse_loss(video_semantic, a_quantized.detach())
        #v_loss = self.commitment_cost * 1.0 * v_e_latent_loss
        v_loss = self.commitment_cost * 2.0 * v_e_latent_loss + self.commitment_cost * va_e_latent_loss

        a_quantized = audio_semantic + (a_quantized - audio_semantic).detach()
        v_quantized = video_semantic + (v_quantized - video_semantic).detach()

        a_avg_probs = torch.mean(a_encodings, dim=0)
        a_perplexity = torch.exp(-torch.sum(a_avg_probs * torch.log(a_avg_probs + 1e-10)))
        v_avg_probs = torch.mean(v_encodings, dim=0)
        v_perplexity = torch.exp(-torch.sum(v_avg_probs * torch.log(v_avg_probs + 1e-10)))
        return a_quantized, v_quantized, a_loss, v_loss, a_perplexity, v_perplexity, cmcm_loss, equal_num
    
