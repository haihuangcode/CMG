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

class Simple_Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Simple_Encoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.input_linear = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()

    def forward(self, input_feat):
        return self.relu(self.input_linear(input_feat))

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

class Simple_Decoder(nn.Module):
    def __init__(self, input_dim, output_dim, vq_dim):
        super(Simple_Decoder, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.relu = nn.ReLU()
        self.simple_rec = nn.Linear(input_dim * 2, output_dim)
        self.simple_linear = nn.Linear(vq_dim, input_dim)

    def forward(self, encoder_result, vq):
        vq_result = self.simple_linear(vq)
        encoder_result = torch.cat([vq_result, encoder_result], dim=2)
        decoder_result = self.simple_rec(encoder_result)
        return decoder_result

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
        # audio_vq = self.Cross_quantizer.Audio_vq_embedding(audio_semantic_result)
        audio_vq = self.Cross_quantizer.rq_embedding(audio_semantic_result)
        return audio_vq
    
    def Audio_VQ_Encoder_indices(self, audio_feat):
        audio_feat = audio_feat.cuda()
        audio_semantic_result = audio_feat.transpose(0, 1).contiguous()
        audio_semantic_result = self.audio_self_att(audio_semantic_result)
        audio_semantic_result = audio_semantic_result.transpose(0, 1).contiguous()
        # audio_vq = self.Cross_quantizer.Audio_vq_embedding(audio_semantic_result)
        indices = self.Cross_quantizer.vq_embedding_indices(audio_semantic_result)
        return indices

    def Video_VQ_Encoder(self, video_feat):
        video_feat = video_feat.cuda()
        video_semantic_result = self.video_semantic_encoder(video_feat).transpose(0, 1).contiguous()
        video_semantic_result = self.video_self_att(video_semantic_result)
        video_semantic_result = video_semantic_result.transpose(0, 1).contiguous()  # [batch, length, hidden_dim]
        # video_vq = self.Cross_quantizer.Video_vq_embedding(video_semantic_result)
        video_vq = self.Cross_quantizer.rq_embedding(video_semantic_result)
        return video_vq
    
    def Video_VQ_Encoder_indices(self, video_feat):
        video_feat = video_feat.cuda()
        video_semantic_result = self.video_semantic_encoder(video_feat).transpose(0, 1).contiguous()
        video_semantic_result = self.video_self_att(video_semantic_result)
        video_semantic_result = video_semantic_result.transpose(0, 1).contiguous()  # [batch, length, hidden_dim]
        # video_vq = self.Cross_quantizer.Video_vq_embedding(video_semantic_result)
        indices = self.Cross_quantizer.vq_embedding_indices(video_semantic_result)
        return indices

    def Text_VQ_Encoder(self, text_feat):
        text_feat = text_feat.cuda()
        text_semantic_result = text_feat.transpose(0, 1).contiguous()
        text_semantic_result = self.text_self_att(text_semantic_result)
        text_semantic_result = text_semantic_result.transpose(0, 1).contiguous()
        # text_vq = self.Cross_quantizer.Text_vq_embedding(text_semantic_result)
        text_vq = self.Cross_quantizer.rq_embedding(text_semantic_result)
        return text_vq
    
    def Text_VQ_Encoder_indices(self, text_feat):
        text_feat = text_feat.cuda()
        text_semantic_result = text_feat.transpose(0, 1).contiguous()
        text_semantic_result = self.text_self_att(text_semantic_result)
        text_semantic_result = text_semantic_result.transpose(0, 1).contiguous()
        # text_vq = self.Cross_quantizer.Text_vq_embedding(text_semantic_result)
        indices = self.Cross_quantizer.vq_embedding_indices(text_semantic_result)
        return indices

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

    def vq_embedding_indices(self, audio_semantic):

        B, T, D = audio_semantic.size()
        a_flat = audio_semantic.detach().mean(dim=1)
        # a_flat = audio_semantic.detach().reshape(-1, D)  # [B, T, D] -> [BxT, D]
        a_distances = torch.addmm(torch.sum(self.embedding ** 2, dim=1) +
                                  torch.sum(a_flat ** 2, dim=1, keepdim=True),
                                  a_flat, self.embedding.t(),
                                  alpha=-2.0, beta=1.0)  # [BxT, M]
        a_indices = torch.argmin(a_distances.double(), dim=-1)  # [BxT,1]
        return a_indices

    def rq_embedding(self, semantic, beam_size=1):
        assert beam_size >= 1
        self.embedding[0] = 0
        if beam_size == 1:
            """residual_quantize"""
            M, D = self.embedding.size()
            B, T, D = semantic.size()
            residual_feature = semantic.detach().clone()
            residual_list = []
            quant_list = []
            indice_list = []
            encoding_list = []
            aggregated_quants = torch.zeros_like(residual_feature)
            for i in range(1):
                j = i*0
                distances = torch.addmm(torch.sum(self.embedding  ** 2, dim=1) +
                                    torch.sum(residual_feature.reshape(-1, D) ** 2, dim=1, keepdim=True),
                                    residual_feature.reshape(-1, D), self.embedding.t(),
                                    alpha=-2.0, beta=1.0)# [BxT, M]
                indices = torch.argmin(distances.double(), dim=-1)# [BxT,1]
                
                """autodeep"""
                
                # min_distances = torch.min(distances, dim=-1)[0]  
                # indices = torch.argmin(distances.double(), dim=-1)
                # # 如果最近邻距离比第一个 Anchor 的距离大超过一半,则使用第一个 Anchor
                # mask = min_distances > 0.95 * distances[:,0]  
                # indices[mask] = 0
                
                # min_distances = torch.min(distances, dim=-1)[0]  
                # indices = torch.argmin(distances.double(), dim=-1)
                # # 如果最近邻距离比第一个 Anchor 的距离大超过一半,则使用第一个 Anchor
                # mask = min_distances > (1-0.01*i) * distances[:,0]  
                # indices[mask] = 0
                
                encodings = F.one_hot(indices, M).double()  # [BxT, M]
                quantized = F.embedding(indices, self.embedding).view_as(semantic)# [BxT,D]->[B,T,D]
                residual_list.append(residual_feature.clone().reshape(-1, D))  # [B, T, D] -> [BxT, D] -> [codebook_size,BxT, D]
                residual_feature = residual_feature - quantized
                aggregated_quants = aggregated_quants + quantized
                # residual_feature.sub_(quantized)#[B,T,D]
                # aggregated_quants.add_(quantized)#[B,T,D]
                quant_list.append(aggregated_quants.clone())#[codebook_size,B,T,D]
                indice_list.append(indices)#[codebook_size,BxT,1]
                encoding_list.append(encodings)#[codebook_size,BxT,M]
            
            aggregated_quants = semantic + (aggregated_quants - semantic).detach()#[B,T,D]
            # for i in range(self.codebook_size):
            #     quant_list[i] = semantic + (quant_list[i] - semantic).detach()
            
            return aggregated_quants
        
        else:
            return None

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
            self.embedding[i] = activated_quantized[random.randint(0,len(activated_indices)-1)] + torch.Tensor(256).uniform_(-1/1024, -1/1024).cuda()

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
            self.embedding[i] = activated_quantized[random.randint(0,len(activated_indices)-1)] + torch.Tensor(256).uniform_(-1/1024, -1/1024).cuda()

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
            self.embedding[i] = activated_quantized[random.randint(0,len(activated_indices)-1)] + torch.Tensor(256).uniform_(-1/1024, -1/1024).cuda()


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
    
    
class AVT_RQVAE_Encoder(nn.Module):
    def __init__(self, audio_dim, video_dim, text_dim, audio_output_dim, video_output_dim, text_output_dim, n_embeddings, embedding_dim):
        super(AVT_RQVAE_Encoder, self).__init__()
        self.video_dim = video_dim
        self.text_dim = text_dim
        self.audio_dim = audio_dim
        self.hidden_dim = embedding_dim
        self.Video_encoder = Video_Encoder(video_dim, video_output_dim)
        self.Audio_encoder = Audio_Encoder(audio_dim, audio_output_dim)
        self.Text_encoder = Text_Encoder(text_dim, text_output_dim)
        
        # 统一VQ
        self.Cross_quantizer = Cross_RQEmbeddingEMA_AVT(n_embeddings, self.hidden_dim)
        self.video_semantic_encoder = Video_Semantic_Encoder(video_dim)
        self.video_self_att = InternalTemporalRelationModule(input_dim=video_dim, d_model=self.hidden_dim)
        self.text_self_att = InternalTemporalRelationModule(input_dim=text_dim, d_model=self.hidden_dim)
        self.audio_self_att = InternalTemporalRelationModule(input_dim=audio_dim, d_model=self.hidden_dim)

    def Audio_VQ_Encoder(self, audio_feat):
        audio_feat = audio_feat.cuda()
        audio_semantic_result = audio_feat.transpose(0, 1).contiguous()
        audio_semantic_result = self.audio_self_att(audio_semantic_result)
        audio_semantic_result = audio_semantic_result.transpose(0, 1).contiguous()
        #audio_semantic_result = self.common_encoder(audio_semantic_result)
        audio_vq,_,_,_,_ = self.Cross_quantizer.rq_embedding(audio_semantic_result)
        return audio_vq

    def Video_VQ_Encoder(self, video_feat):
        video_feat = video_feat.cuda()
        video_semantic_result = self.video_semantic_encoder(video_feat).transpose(0, 1).contiguous()
        video_semantic_result = self.video_self_att(video_semantic_result)
        video_semantic_result = video_semantic_result.transpose(0, 1).contiguous()  # [batch, length, hidden_dim]
        #video_semantic_result = self.common_encoder(video_semantic_result)
        video_vq,_,_,_,_ = self.Cross_quantizer.rq_embedding(video_semantic_result)
        return video_vq

    def Text_VQ_Encoder(self, text_feat):
        text_feat = text_feat.cuda()
        text_semantic_result = text_feat.transpose(0, 1).contiguous()
        text_semantic_result = self.text_self_att(text_semantic_result)
        text_semantic_result = text_semantic_result.transpose(0, 1).contiguous()
        #audio_semantic_result = self.common_encoder(audio_semantic_result)
        text_vq,_,_,_,_ = self.Cross_quantizer.rq_embedding(text_semantic_result)
        return text_vq

    def forward(self, audio_feat, video_feat, text_feat, epoch):
        video_feat = video_feat.cuda()
        text_feat = text_feat.cuda()
        audio_feat = audio_feat.cuda()
        video_encoder_result, video_club_feature = self.Video_encoder(video_feat)

        # print('video_feat.shape:',video_feat.shape)# [80, 10, 7, 7, 512]
        # print('video_encoder_result.shape:',video_encoder_result.shape)# [80, 10, 3, 3, 2048]
        # print('video_club_feature.shape:',video_club_feature.shape)# [80, 10, 512]
        
        text_encoder_result = self.Text_encoder(text_feat)  # [batch, length, audio_output_dim]
        audio_encoder_result = self.Audio_encoder(audio_feat)  # [batch, length, audio_output_dim]
        
        video_semantic_result = self.video_semantic_encoder(video_feat).transpose(0, 1).contiguous()
        video_semantic_result = self.video_self_att(video_semantic_result)
        video_semantic_result = video_semantic_result.transpose(0, 1).contiguous()  # [batch, length, hidden_dim]
        
        text_semantic_result = text_feat.transpose(0, 1).contiguous()
        text_semantic_result = self.text_self_att(text_semantic_result)# [length, batch, hidden_dim]
        text_semantic_result = text_semantic_result.transpose(0, 1).contiguous()  # [batch, length, hidden_dim]
        
        audio_semantic_result = audio_feat.transpose(0, 1).contiguous()
        audio_semantic_result = self.audio_self_att(audio_semantic_result)# [length, batch, hidden_dim]
        audio_semantic_result = audio_semantic_result.transpose(0, 1).contiguous()  # [batch, length, hidden_dim]

        audio_vq, video_vq, text_vq, a_quant_list, v_quant_list,t_quant_list, audio_embedding_loss, video_embedding_loss, text_embedding_loss, cmcm_loss, equal_num, zero_num, same_layer_fusion_loss,adjacent_layer_separation_loss \
            = self.Cross_quantizer(audio_semantic_result, video_semantic_result, text_semantic_result, epoch)

        return audio_semantic_result, video_semantic_result, text_semantic_result, \
               audio_encoder_result, video_encoder_result, video_club_feature, text_encoder_result, \
               audio_vq, video_vq, text_vq, a_quant_list, v_quant_list,t_quant_list, audio_embedding_loss, video_embedding_loss, text_embedding_loss, cmcm_loss, equal_num, zero_num, same_layer_fusion_loss,adjacent_layer_separation_loss
              
infonce = InfoNCE().double().cuda()
               
class Cross_RQEmbeddingEMA_AVT(nn.Module):
    def __init__(self, n_embeddings, embedding_dim, codebook_size = 4, shared_codebook=False, commitment_cost=0.25, decay=0.99, epsilon=1e-5):
        super(Cross_RQEmbeddingEMA_AVT, self).__init__()
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon
        self.codebook_size = codebook_size
        self.shared_codebook = shared_codebook
        
        from typing import Iterable
        n_embeddings = n_embeddings if isinstance(n_embeddings, Iterable) else [n_embeddings for _ in range(self.codebook_size)]
            
        init_bound = 1.0 / 400.0
        embedding = [torch.Tensor(n_embeddings[i], embedding_dim).uniform_(-init_bound, init_bound) for i in range(self.codebook_size)]
        # address 0 is always 0
        for i in range(self.codebook_size):
            embedding[i][0] = 0
        self.register_buffer("embedding",torch.stack(embedding))
        
        ema_count = [torch.zeros(n_embeddings[i]) for i in range(self.codebook_size)]
        self.register_buffer("ema_count", torch.stack(ema_count))
        
        self.register_buffer("ema_weight", self.embedding.clone())
        
        unactivated_count = [-torch.ones(n_embeddings[i]) for i in range(self.codebook_size)]
        self.register_buffer("unactivated_count", torch.stack(unactivated_count))
        # embedding[i*int(not self.shared_codebook)]    

    
    def rq_embedding(self, semantic, beam_size=1):
        assert beam_size >= 1
        
        if beam_size == 1:
            """residual_quantize"""
            M, D = self.embedding[0].size()
            B, T, D = semantic.size()
            residual_feature = semantic.detach().clone()
            residual_list = []
            quant_list = []
            indice_list = []
            encoding_list = []
            aggregated_quants = torch.zeros_like(residual_feature)
            for i in range(self.codebook_size):
                j = i*int(not self.shared_codebook)
                distances = torch.addmm(torch.sum(self.embedding[j]  ** 2, dim=1) +
                                    torch.sum(residual_feature.reshape(-1, D) ** 2, dim=1, keepdim=True),
                                    residual_feature.reshape(-1, D), self.embedding[j].t(),
                                    alpha=-2.0, beta=1.0)# [BxT, M]
                indices = torch.argmin(distances.double(), dim=-1)# [BxT,1]
                
                """autodeep"""
                
                # min_distances = torch.min(distances, dim=-1)[0]  
                # indices = torch.argmin(distances.double(), dim=-1)
                # # 如果最近邻距离比第一个 Anchor 的距离大超过一半,则使用第一个 Anchor
                # mask = min_distances > 0.95 * distances[:,0]  
                # indices[mask] = 0
                
                # min_distances = torch.min(distances, dim=-1)[0]  
                # indices = torch.argmin(distances.double(), dim=-1)
                # # 如果最近邻距离比第一个 Anchor 的距离大超过一半,则使用第一个 Anchor
                # mask = min_distances > (1-0.01*i) * distances[:,0]  
                # indices[mask] = 0
                
                encodings = F.one_hot(indices, M).double()  # [BxT, M]
                quantized = F.embedding(indices, self.embedding[j]).view_as(semantic)# [BxT,D]->[B,T,D]
                residual_list.append(residual_feature.clone().reshape(-1, D))  # [B, T, D] -> [BxT, D] -> [codebook_size,BxT, D]
                residual_feature = residual_feature - quantized
                aggregated_quants = aggregated_quants + quantized
                # residual_feature.sub_(quantized)#[B,T,D]
                # aggregated_quants.add_(quantized)#[B,T,D]
                quant_list.append(aggregated_quants.clone())#[codebook_size,B,T,D]
                indice_list.append(indices)#[codebook_size,BxT,1]
                encoding_list.append(encodings)#[codebook_size,BxT,M]
            
            aggregated_quants = semantic + (aggregated_quants - semantic).detach()#[B,T,D]
            # for i in range(self.codebook_size):
            #     quant_list[i] = semantic + (quant_list[i] - semantic).detach()
            
            return aggregated_quants, quant_list, indice_list, encoding_list, residual_list
        
        else:
            """beam_quantize"""
            # bsz = semantic.shape[0]
        
            # # 初始化beam
            # beam = [ [ [semantic[b].detach().clone(), torch.zeros_like(semantic[b]), [], 0, []] ] for b in range(bsz) ]

            # aggregated_quants = []
            # quant_list = []
            # indice_list = []
            
            # for i in range(self.codebook_size):
            #     j = i*int(not self.shared_codebook)
            #     beam_candidates = [[] for _ in range(bsz)]
            #     residual = torch.stack([b2[0] for b1 in beam for b2 in b1])

            #     M, D = self.embedding[j].size()
            #     distances = torch.addmm(torch.sum(self.embedding[j] ** 2, dim=1) + 
            #                             torch.sum(residual.reshape(-1, D) ** 2, dim=1, keepdim=True),
            #                             residual.reshape(-1, D), self.embedding[j].t(),
            #                             alpha=-2.0, beta=1.0)
                                        
            #     dist, indices = distances.topk(beam_size, dim=-1, largest=False)
            #     quantized = F.embedding(indices, self.embedding[j]).view_as(semantic)# [BxT,D]->[B,T,D]
                
            #     for b in range(bsz):
            #         for seq in beam[b]:
            #             for j in range(beam_size):
            #                 new_seq = copy.deepcopy(seq)
            #                 new_seq[0].sub_(self.embedding[i][indices[b][j]].view_as(semantic[b]))
            #                 new_seq[1].add_(self.embedding[i][indices[b][j]].view_as(semantic[b])) 
            #                 new_seq[2].append(new_seq[1])
            #                 new_seq[3] += dist[b][j]
            #                 new_seq[4].append(indices[b][j])
                        
            #                 beam_candidates[b].append(new_seq)

            #         beam_candidates[b].sort(key=lambda x: -x[3])
            #         beam[b] = beam_candidates[b][:beam_size]
                
            #     # 记录当前topk路径的量化结果
            #     quant_list.append(torch.stack([b[0][1] for b in beam]))  
            #     indice_list.append(torch.stack([b[0][4] for b in beam]))

            # 返回累积量化、量化码字索引
            # return aggregated_quants, quant_list, indice_list, None ,None
        
            # """beam_quantize"""
            return None,None,None,None,None

    def forward(self, audio_semantic, video_semantic, text_semantic, epoch):
        
        M, D = self.embedding[0].size()
        B, T, D = audio_semantic.size()

        # a_aggregated_quants, a_quant_list, a_indice_list, a_encoding_list, a_residual_list = self.rq_embedding(audio_semantic.detach().clone())
        # v_aggregated_quants, v_quant_list, v_indice_list, v_encoding_list, v_residual_list = self.rq_embedding(video_semantic.detach().clone())
        # t_aggregated_quants, t_quant_list, t_indice_list, t_encoding_list, t_residual_list = self.rq_embedding(text_semantic.detach().clone())
        
        
        # if beam_size == 1:
        """residual_quantize"""
        a_residual_feature = audio_semantic.detach().clone()
        a_residual_list = []
        a_quant_list = []
        a_indice_list = []
        a_encoding_list = []
        a_aggregated_quants = torch.zeros_like(a_residual_feature)
        
        v_residual_feature = video_semantic.detach().clone()
        v_residual_list = []
        v_quant_list = []
        v_indice_list = []
        v_encoding_list = []
        v_aggregated_quants = torch.zeros_like(v_residual_feature)
        
        t_residual_feature = text_semantic.detach().clone()
        t_residual_list = []
        t_quant_list = []
        t_indice_list = []
        t_encoding_list = []
        t_aggregated_quants = torch.zeros_like(t_residual_feature)
        
        
        """differ quantize"""
        for i in range(self.codebook_size):
            j = i*int(not self.shared_codebook)
            """audio"""
            distances = torch.addmm(torch.sum(self.embedding[j]  ** 2, dim=1) +
                                torch.sum(a_residual_feature.reshape(-1, D) ** 2, dim=1, keepdim=True),
                                a_residual_feature.reshape(-1, D), self.embedding[j].t(),
                                alpha=-2.0, beta=1.0)# [BxT, M]
            
            indices = torch.argmin(distances.double(), dim=-1)# [BxT,1]
            
            """autodeep"""
            # min_distances = torch.min(distances, dim=-1)[0]  
            # indices = torch.argmin(distances.double(), dim=-1)
            # # 如果最近邻距离比第一个 Anchor 的距离大超过一半,则使用第一个 Anchor
            # mask = min_distances > 0.95 * distances[:,0]  
            # indices[mask] = 0
            
            encodings = F.one_hot(indices, M).double()  # [BxT, M]
            quantized = F.embedding(indices, self.embedding[j]).view_as(audio_semantic)# [BxT,D]->[B,T,D]
            a_residual_list.append(a_residual_feature.clone().reshape(-1, D))  # [B, T, D] -> [BxT, D] -> [codebook_size,BxT, D]
            a_residual_feature = a_residual_feature - quantized
            a_aggregated_quants = a_aggregated_quants + quantized
            # residual_feature.sub_(quantized)#[B,T,D]
            # aggregated_quants.add_(quantized)#[B,T,D]
            a_quant_list.append(a_aggregated_quants.clone())#[codebook_size,B,T,D]
            a_indice_list.append(indices.clone())#[codebook_size,BxT,1]
            a_encoding_list.append(encodings.clone())#[codebook_size,BxT,M]
            """video"""
            distances = torch.addmm(torch.sum(self.embedding[j]  ** 2, dim=1) +
                                torch.sum(v_residual_feature.reshape(-1, D) ** 2, dim=1, keepdim=True),
                                v_residual_feature.reshape(-1, D), self.embedding[j].t(),
                                alpha=-2.0, beta=1.0)# [BxT, M]
            
            indices = torch.argmin(distances.double(), dim=-1)# [BxT,1]
            
            """autodeep"""
            # min_distances = torch.min(distances, dim=-1)[0]  
            # indices = torch.argmin(distances.double(), dim=-1)
            # # 如果最近邻距离比第一个 Anchor 的距离大超过一半,则使用第一个 Anchor
            # mask = min_distances > 0.95 * distances[:,0]  
            # indices[mask] = 0
            
            encodings = F.one_hot(indices, M).double()  # [BxT, M]
            quantized = F.embedding(indices, self.embedding[j]).view_as(video_semantic)# [BxT,D]->[B,T,D]
            v_residual_list.append(v_residual_feature.clone().reshape(-1, D))  # [B, T, D] -> [BxT, D] -> [codebook_size,BxT, D]
            v_residual_feature = v_residual_feature - quantized
            v_aggregated_quants = v_aggregated_quants + quantized
            # residual_feature.sub_(quantized)#[B,T,D]
            # aggregated_quants.add_(quantized)#[B,T,D]
            v_quant_list.append(v_aggregated_quants.clone())#[codebook_size,B,T,D]
            v_indice_list.append(indices.clone())#[codebook_size,BxT,1]
            v_encoding_list.append(encodings.clone())#[codebook_size,BxT,M]
            """text"""
            distances = torch.addmm(torch.sum(self.embedding[j]  ** 2, dim=1) +
                                torch.sum(t_residual_feature.reshape(-1, D) ** 2, dim=1, keepdim=True),
                                t_residual_feature.reshape(-1, D), self.embedding[j].t(),
                                alpha=-2.0, beta=1.0)# [BxT, M]
            
            indices = torch.argmin(distances.double(), dim=-1)# [BxT,1]
            
            """autodeep"""
            # min_distances = torch.min(distances, dim=-1)[0]  
            # indices = torch.argmin(distances.double(), dim=-1)
            # # 如果最近邻距离比第一个 Anchor 的距离大超过一半,则使用第一个 Anchor
            # mask = min_distances > 0.95 * distances[:,0]  
            # indices[mask] = 0
            
            encodings = F.one_hot(indices, M).double()  # [BxT, M]
            quantized = F.embedding(indices, self.embedding[j]).view_as(text_semantic)# [BxT,D]->[B,T,D]
            t_residual_list.append(t_residual_feature.clone().reshape(-1, D))  # [B, T, D] -> [BxT, D] -> [codebook_size,BxT, D]
            t_residual_feature = t_residual_feature - quantized
            t_aggregated_quants = t_aggregated_quants + quantized
            # residual_feature.sub_(quantized)#[B,T,D]
            # aggregated_quants.add_(quantized)#[B,T,D]
            t_quant_list.append(t_aggregated_quants.clone())#[codebook_size,B,T,D]
            t_indice_list.append(indices.clone())#[codebook_size,BxT,1]
            t_encoding_list.append(encodings.clone())#[codebook_size,BxT,M]
        
        """same residual_quantize"""
        # a_residual_feature = audio_semantic.detach().clone()
        # a_residual_list = []
        # a_quant_list = []
        # a_aggregated_quants = torch.zeros_like(a_residual_feature)
        
        # v_residual_feature = video_semantic.detach().clone()
        # v_residual_list = []
        # v_quant_list = []
        # v_aggregated_quants = torch.zeros_like(v_residual_feature)
        
        # t_residual_feature = text_semantic.detach().clone()
        # t_residual_list = []
        # t_quant_list = []
        # t_aggregated_quants = torch.zeros_like(t_residual_feature)
        
        # indice_list = []
        # encoding_list = []
        
        """same quantize"""
        # for i in range(self.codebook_size):
        #     j = i*int(not self.shared_codebook)
        #     """audio"""
        #     a_distances = torch.addmm(torch.sum(self.embedding[j]  ** 2, dim=1) +
        #                         torch.sum(a_residual_feature.reshape(-1, D) ** 2, dim=1, keepdim=True),
        #                         a_residual_feature.reshape(-1, D), self.embedding[j].t(),
        #                         alpha=-2.0, beta=1.0)# [BxT, M]
        #     v_distances = torch.addmm(torch.sum(self.embedding[j]  ** 2, dim=1) +
        #                         torch.sum(v_residual_feature.reshape(-1, D) ** 2, dim=1, keepdim=True),
        #                         v_residual_feature.reshape(-1, D), self.embedding[j].t(),
        #                         alpha=-2.0, beta=1.0)# [BxT, M]
        #     t_distances = torch.addmm(torch.sum(self.embedding[j]  ** 2, dim=1) +
        #                         torch.sum(t_residual_feature.reshape(-1, D) ** 2, dim=1, keepdim=True),
        #                         t_residual_feature.reshape(-1, D), self.embedding[j].t(),
        #                         alpha=-2.0, beta=1.0)# [BxT, M]
            
        #     distances = a_distances + v_distances + t_distances
            
        #     indices = torch.argmin(distances.double(), dim=-1)# [BxT,1]
        #     encodings = F.one_hot(indices, M).double()  # [BxT, M]
        #     quantized = F.embedding(indices, self.embedding[j]).view_as(audio_semantic)# [BxT,D]->[B,T,D]
            
        #     a_residual_list.append(a_residual_feature.clone().reshape(-1, D))  # [B, T, D] -> [BxT, D] -> [codebook_size,BxT, D]
        #     a_residual_feature = a_residual_feature - quantized
        #     a_aggregated_quants = a_aggregated_quants + quantized
        #     a_quant_list.append(a_aggregated_quants.clone())#[codebook_size,B,T,D]
            
        #     v_residual_list.append(v_residual_feature.clone().reshape(-1, D))  # [B, T, D] -> [BxT, D] -> [codebook_size,BxT, D]
        #     v_residual_feature = v_residual_feature - quantized
        #     v_aggregated_quants = v_aggregated_quants + quantized
        #     v_quant_list.append(v_aggregated_quants.clone())#[codebook_size,B,T,D]
            
        #     t_residual_list.append(t_residual_feature.clone().reshape(-1, D))  # [B, T, D] -> [BxT, D] -> [codebook_size,BxT, D]
        #     t_residual_feature = t_residual_feature - quantized
        #     t_aggregated_quants = t_aggregated_quants + quantized
        #     t_quant_list.append(t_aggregated_quants.clone())#[codebook_size,B,T,D]
            
        #     indice_list.append(indices.clone())#[codebook_size,BxT,1]
        #     encoding_list.append(encodings.clone())#[codebook_size,BxT,M]

        """compute equal_num"""
        equal_num_list = []
        zero_num_list = []
        for i in range(self.codebook_size):
            a_indices_reshape = a_indice_list[i].reshape(B, T)
            v_indices_reshape = v_indice_list[i].reshape(B, T)
            t_indices_reshape = t_indice_list[i].reshape(B, T)
            a_indices_mode = torch.mode(a_indices_reshape, dim=-1, keepdim=False)#返回众数
            v_indices_mode = torch.mode(v_indices_reshape, dim=-1, keepdim=False)
            t_indices_mode = torch.mode(t_indices_reshape, dim=-1, keepdim=False)

            equal_item = (a_indices_mode.values == v_indices_mode.values) & (a_indices_mode.values == t_indices_mode.values)
            equal_num = int(equal_item.sum())
            equal_num_list.append(equal_num)
            
            a_zero_num = int((a_indices_mode.values == 0).sum())
            v_zero_num = int((v_indices_mode.values == 0).sum())
            t_zero_num = int((t_indices_mode.values == 0).sum())
            zero_num_list.append([a_zero_num,v_zero_num,t_zero_num])
            
        """differ MM-EMA update"""
        if self.training:
            for i in range(self.codebook_size):
                j = i*int(not self.shared_codebook) #
                # audio
                self.ema_count[j] = self.decay * self.ema_count[j] + (1 - self.decay) * torch.sum(a_encoding_list[i], dim=0)
                n = torch.sum(self.ema_count[j])
                self.ema_count[j] = (self.ema_count[j] + self.epsilon) / (n + M * self.epsilon) * n
                a_dw = torch.matmul(a_encoding_list[i].t(), a_residual_list[i])
                av_dw = torch.matmul(a_encoding_list[i].t(), v_residual_list[i])
                at_dw = torch.matmul(a_encoding_list[i].t(), t_residual_list[i])
                self.ema_weight[j] = self.decay * self.ema_weight[j] + 0.5*(1 - self.decay) * a_dw + 0.25*(1 - self.decay) * av_dw + 0.25*(1 - self.decay) * at_dw
                
                # self.ema_weight[j] = self.decay * self.ema_weight[j] 
                self.embedding[j] = self.ema_weight[j] / self.ema_count[j].unsqueeze(-1)

                # video
                self.ema_count[j] = self.decay * self.ema_count[j] + (1 - self.decay) * torch.sum(v_encoding_list[i], dim=0)
                n = torch.sum(self.ema_count[j])
                self.ema_count[j] = (self.ema_count[j] + self.epsilon) / (n + M * self.epsilon) * n
                v_dw = torch.matmul(v_encoding_list[i].t(), v_residual_list[i])
                va_dw = torch.matmul(v_encoding_list[i].t(), a_residual_list[i])
                vt_dw = torch.matmul(v_encoding_list[i].t(), t_residual_list[i])
                self.ema_weight[j] = self.decay * self.ema_weight[j] + 0.5*(1 - self.decay) * v_dw + 0.25*(1 - self.decay) * va_dw + 0.25*(1 - self.decay) * vt_dw
                
                # self.ema_weight[j] = self.decay * self.ema_weight[j]
                self.embedding[j] = self.ema_weight[j] / self.ema_count[j].unsqueeze(-1)
                
                # text
                self.ema_count[j] = self.decay * self.ema_count[j] + (1 - self.decay) * torch.sum(t_encoding_list[i], dim=0)
                n = torch.sum(self.ema_count[j])
                self.ema_count[j] = (self.ema_count[j] + self.epsilon) / (n + M * self.epsilon) * n
                t_dw = torch.matmul(t_encoding_list[i].t(), t_residual_list[i])
                ta_dw = torch.matmul(t_encoding_list[i].t(), a_residual_list[i])
                tv_dw = torch.matmul(t_encoding_list[i].t(), v_residual_list[i])
                self.ema_weight[j] = self.decay * self.ema_weight[j] + 0.5*(1 - self.decay) * t_dw + 0.25*(1 - self.decay) * ta_dw + 0.25*(1 - self.decay) * tv_dw
                
                # self.ema_weight[j] = self.decay * self.ema_weight[j]
                self.embedding[j] = self.ema_weight[j] / self.ema_count[j].unsqueeze(-1)
                
                
        """same MM-EMA update"""
        # if self.training:
        #     for i in range(self.codebook_size):
        #         j = i*int(not self.shared_codebook) # buffer的用这个索引，因为涉及到shared_codebook
        #         # audio
        #         self.ema_count[j] = self.decay * self.ema_count[j] + (1 - self.decay) * torch.sum(encoding_list[i], dim=0)
        #         n = torch.sum(self.ema_count[j])
        #         self.ema_count[j] = (self.ema_count[j] + self.epsilon) / (n + M * self.epsilon) * n
        #         a_dw = torch.matmul(encoding_list[i].t(), a_residual_list[i])
        #         v_dw = torch.matmul(encoding_list[i].t(), v_residual_list[i])
        #         t_dw = torch.matmul(encoding_list[i].t(), t_residual_list[i])
        #         self.ema_weight[j] = self.decay * self.ema_weight[j] + 1.0/3.0*(1 - self.decay) * a_dw + 1.0/3.0*(1 - self.decay) * v_dw + 1.0/3.0*(1 - self.decay) * t_dw
        #         self.embedding[j] = self.ema_weight[j] / self.ema_count[j].unsqueeze(-1)


        """reset code"""
        for i in range(self.codebook_size):
            j = i*int(not self.shared_codebook)
            self.unactivated_count[j] = self.unactivated_count[j] + 1
            """differ quantize"""
            for indice in a_indice_list[i]:
                self.unactivated_count[j][indice.item()] = 0
            for indice in v_indice_list[i]:
                self.unactivated_count[j][indice.item()] = 0
            for indice in t_indice_list[i]:
                self.unactivated_count[j][indice.item()] = 0
            """same quantize"""
            # for indice in indice_list[i]:
            #     self.unactivated_count[j][indice.item()] = 0  
            
            activated_indices = []
            unactivated_indices = []
            for k, x in enumerate(self.unactivated_count[j]):
                if x > 300:
                    unactivated_indices.append(k)
                    self.unactivated_count[j][k] = 0
                elif x >= 0 and x < 100:
                    activated_indices.append(k)
            activated_quantized = F.embedding(torch.tensor(activated_indices,dtype=torch.int32).cuda(), self.embedding[j])
            for k in unactivated_indices:
                self.embedding[j][k] = activated_quantized[random.randint(0,len(activated_indices)-1)] + torch.Tensor(int(D)).uniform_(-1.0/400.0, -1.0/400.0).cuda()
        
        # address 0 is always 0
        for i in range(self.codebook_size):
            self.embedding[i][0] = 0

        """cmcm_loss"""
        # w = [0.5,0.25,0.125,0.125]# weight
        # cmcm_loss = torch.zeros(1).cuda()
        # for i in range(self.codebook_size):
        #     j = i*int(not self.shared_codebook)
        #     a_residual_semantic = w[i]*audio_semantic.reshape(-1, D) + (a_residual_list[j] - w[i]*audio_semantic.reshape(-1, D)).detach()
        #     v_residual_semantic = w[i]*video_semantic.reshape(-1, D) + (v_residual_list[j] - w[i]*video_semantic.reshape(-1, D)).detach()
        #     t_residual_semantic = w[i]*text_semantic.reshape(-1, D) + (t_residual_list[j] - w[i]*text_semantic.reshape(-1, D)).detach()
        #     a_distances_gradient = torch.addmm(torch.sum(self.embedding[j] ** 2, dim=1) +
        #                                     torch.sum(a_residual_semantic ** 2, dim=1, keepdim=True),
        #                                     a_residual_semantic, self.embedding[j].t(),
        #                                     alpha=-2.0, beta=1.0)  # [BxT, M]

        #     v_distances_gradient = torch.addmm(torch.sum(self.embedding[j] ** 2, dim=1) +
        #                                     torch.sum(v_residual_semantic ** 2, dim=1, keepdim=True),
        #                                     v_residual_semantic, self.embedding[j].t(),
        #                                     alpha=-2.0, beta=1.0)  # [BxT, M]
            
        #     t_distances_gradient = torch.addmm(torch.sum(self.embedding[j] ** 2, dim=1) +
        #                                     torch.sum(t_residual_semantic ** 2, dim=1, keepdim=True),
        #                                     t_residual_semantic, self.embedding[j].t(),
        #                                     alpha=-2.0, beta=1.0)  # [BxT, M]

        #     a_ph = F.softmax(-torch.sqrt(a_distances_gradient), dim=1)  # [BxT, M] torch.Size([160, 512])
        #     v_ph = F.softmax(-torch.sqrt(v_distances_gradient), dim=1)  # [BxT, M] torch.Size([160, 512])
        #     t_ph = F.softmax(-torch.sqrt(t_distances_gradient), dim=1)  # [BxT, M] torch.Size([160, 512])

        #     a_ph = torch.reshape(a_ph, ((B, T, M)))  # [BxT, M] -> [B, T, M]
        #     a_pH = torch.mean(a_ph, dim=1)  # [B, T, M] -> [B, M]
        #     v_ph = torch.reshape(v_ph, ((B, T, M)))  # [BxT, M] -> [B, T, M]
        #     v_pH = torch.mean(v_ph, dim=1)  # [B, T, M] -> [B, M]
        #     t_ph = torch.reshape(t_ph, ((B, T, M)))  # [BxT, M] -> [B, T, M]
        #     t_pH = torch.mean(t_ph, dim=1)  # [B, T, M] -> [B, M]

        #     Scode_av = a_pH @ torch.log(v_pH.t() + 1e-10) + v_pH @ torch.log(a_pH.t() + 1e-10)
        #     Scode_at = a_pH @ torch.log(t_pH.t() + 1e-10) + t_pH @ torch.log(a_pH.t() + 1e-10)
        #     Scode_tv = t_pH @ torch.log(v_pH.t() + 1e-10) + v_pH @ torch.log(t_pH.t() + 1e-10)

        #     # caculate Lcmcm
        #     # If the numerical values in the calculation process of exp are too large, 
        #     # you can add a logC to each item in the matrix, where logC = -Scode.
        #     MaxScode_av = torch.max(-Scode_av)
        #     EScode_av = torch.exp(Scode_av + MaxScode_av)
            
        #     MaxScode_at = torch.max(-Scode_at)
        #     EScode_at = torch.exp(Scode_at + MaxScode_at)
            
        #     MaxScode_tv = torch.max(-Scode_tv)
        #     EScode_tv = torch.exp(Scode_tv + MaxScode_tv)

        #     EScode_sumdim1_av = torch.sum(EScode_av, dim=1)
        #     Lcmcm_av = 0
            
        #     EScode_sumdim1_at = torch.sum(EScode_at, dim=1)
        #     Lcmcm_at = 0
            
        #     EScode_sumdim1_tv = torch.sum(EScode_tv, dim=1)
        #     Lcmcm_tv = 0
            
        #     for i in range(B):
        #         Lcmcm_av -= torch.log(EScode_av[i, i] / (EScode_sumdim1_av[i] + self.epsilon))
        #         Lcmcm_at -= torch.log(EScode_at[i, i] / (EScode_sumdim1_at[i] + self.epsilon))
        #         Lcmcm_tv -= torch.log(EScode_tv[i, i] / (EScode_sumdim1_tv[i] + self.epsilon))
                
        #     Lcmcm_av /= B
        #     Lcmcm_at /= B
        #     Lcmcm_tv /= B
        #     cmcm_loss = cmcm_loss + Lcmcm_av + Lcmcm_at + Lcmcm_tv

        # cmcm_loss = cmcm_loss * 0.5
        cmcm_loss = torch.zeros(1).cuda()

        """differ quantize"""
        a_e_latent_loss = F.mse_loss(audio_semantic, a_aggregated_quants.detach())
        av_e_latent_loss = F.mse_loss(audio_semantic, v_aggregated_quants.detach())
        at_e_latent_loss = F.mse_loss(audio_semantic, t_aggregated_quants.detach())

        a_loss = self.commitment_cost * 2.0 * a_e_latent_loss + 0.5*self.commitment_cost * av_e_latent_loss + 0.5*self.commitment_cost * at_e_latent_loss
        
        v_e_latent_loss = F.mse_loss(video_semantic, v_aggregated_quants.detach())
        va_e_latent_loss = F.mse_loss(video_semantic, a_aggregated_quants.detach())
        vt_e_latent_loss = F.mse_loss(video_semantic, t_aggregated_quants.detach())

        v_loss = self.commitment_cost * 2.0 * v_e_latent_loss + 0.5*self.commitment_cost * va_e_latent_loss + 0.5*self.commitment_cost * vt_e_latent_loss
        
        t_e_latent_loss = F.mse_loss(text_semantic, t_aggregated_quants.detach())
        ta_e_latent_loss = F.mse_loss(text_semantic, a_aggregated_quants.detach())
        tv_e_latent_loss = F.mse_loss(text_semantic, v_aggregated_quants.detach())

        t_loss = self.commitment_cost * 2.0 * t_e_latent_loss + 0.5*self.commitment_cost * ta_e_latent_loss + 0.5*self.commitment_cost * tv_e_latent_loss
        
        
        """same quantize"""
        # a_e_latent_loss = F.mse_loss(audio_semantic, a_aggregated_quants.detach())

        # a_loss = self.commitment_cost * 2.0 * a_e_latent_loss
        
        # v_e_latent_loss = F.mse_loss(video_semantic, v_aggregated_quants.detach())

        # v_loss = self.commitment_cost * 2.0 * v_e_latent_loss
        
        # t_e_latent_loss = F.mse_loss(text_semantic, t_aggregated_quants.detach())

        # t_loss = self.commitment_cost * 2.0 * t_e_latent_loss
        
        """new-embedding-loss"""
        # a_e_latent_loss,av_e_latent_loss,at_e_latent_loss,v_e_latent_loss,va_e_latent_loss,vt_e_latent_loss,t_e_latent_loss,ta_e_latent_loss,tv_e_latent_loss = \
        #     torch.zeros(1).cuda(),torch.zeros(1).cuda(),torch.zeros(1).cuda(),torch.zeros(1).cuda(),torch.zeros(1).cuda(),torch.zeros(1).cuda(),torch.zeros(1).cuda(),torch.zeros(1).cuda(),torch.zeros(1).cuda()
            
        # for i in range(self.codebook_size):
            
        #     a_e_latent_loss += F.mse_loss(audio_semantic, a_quant_list[i].detach())
        #     av_e_latent_loss += F.mse_loss(audio_semantic, v_quant_list[i].detach())
        #     at_e_latent_loss += F.mse_loss(audio_semantic, t_quant_list[i].detach())
        
        #     v_e_latent_loss += F.mse_loss(video_semantic, v_quant_list[i].detach())
        #     va_e_latent_loss += F.mse_loss(video_semantic, a_quant_list[i].detach())
        #     vt_e_latent_loss += F.mse_loss(video_semantic, t_quant_list[i].detach())

        #     t_e_latent_loss += F.mse_loss(text_semantic, t_quant_list[i].detach())
        #     ta_e_latent_loss += F.mse_loss(text_semantic, a_quant_list[i].detach())
        #     tv_e_latent_loss += F.mse_loss(text_semantic, v_quant_list[i].detach())

            
            
        # a_loss = self.commitment_cost * 2.0 * a_e_latent_loss + 0.5*self.commitment_cost * av_e_latent_loss + 0.5*self.commitment_cost * at_e_latent_loss
        # v_loss = self.commitment_cost * 2.0 * v_e_latent_loss + 0.5*self.commitment_cost * va_e_latent_loss + 0.5*self.commitment_cost * vt_e_latent_loss
        # t_loss = self.commitment_cost * 2.0 * t_e_latent_loss + 0.5*self.commitment_cost * ta_e_latent_loss + 0.5*self.commitment_cost * tv_e_latent_loss
            
        a_aggregated_quants = audio_semantic + (a_aggregated_quants - audio_semantic).detach()#[B,T,D]
        v_aggregated_quants = video_semantic + (v_aggregated_quants - video_semantic).detach()#[B,T,D]
        t_aggregated_quants = text_semantic + (t_aggregated_quants - text_semantic).detach()#[B,T,D]
        
       
        for i in range(self.codebook_size):
            quant_w = 1.0
            a_quant_list[i] = quant_w*audio_semantic + (a_quant_list[i] - quant_w*audio_semantic).detach()#[B,T,D]
            v_quant_list[i] = quant_w*video_semantic + (v_quant_list[i] - quant_w*video_semantic).detach()
            t_quant_list[i] = quant_w*text_semantic + (t_quant_list[i] - quant_w*text_semantic).detach()
        
        """same_layer_fusion"""
        # same_layer_fusion_loss = torch.zeros(1).cuda()
        # for i in range(self.codebook_size):
        #     """mean"""
        #     # same_layer_fusion_loss = same_layer_fusion_loss + infonce(torch.mean(a_quant_list[i], dim=1), torch.mean(v_quant_list[i], dim=1)) + \
        #     #                          infonce(torch.mean(a_quant_list[i], dim=1), torch.mean(t_quant_list[i], dim=1)) + \
        #     #                          infonce(torch.mean(v_quant_list[i], dim=1), torch.mean(t_quant_list[i], dim=1))
        #     """reshape"""
        #     same_layer_fusion_loss = same_layer_fusion_loss + infonce(a_quant_list[i].reshape(-1,D), v_quant_list[i].reshape(-1,D)) + \
        #                              infonce(a_quant_list[i].reshape(-1,D), t_quant_list[i].reshape(-1,D)) + \
        #                              infonce(v_quant_list[i].reshape(-1,D), t_quant_list[i].reshape(-1,D))
                                     
        # same_layer_fusion_loss = same_layer_fusion_loss / T
                                     
        
        """adjacent_layer_separation"""
        # if(epoch=>4):
        #     adjacent_layer_separation_loss = torch.zeros(1).cuda()
        #     for i in range(self.codebook_size-1):
        #         adjacent_layer_separation_loss = adjacent_layer_separation_loss - \
        #             infonce(torch.mean(a_quant_list[i], dim=1), torch.mean(a_quant_list[i+1] - a_quant_list[i] , dim=1)) - \
        #             infonce(torch.mean(v_quant_list[i], dim=1), torch.mean(v_quant_list[i+1] - v_quant_list[i] , dim=1)) - \
        #             infonce(torch.mean(t_quant_list[i], dim=1), torch.mean(t_quant_list[i+1] - t_quant_list[i] , dim=1))
        # else:
        #     adjacent_layer_separation_loss = torch.zeros(1).cuda()
        
        same_layer_fusion_loss = torch.zeros(1).cuda()
        adjacent_layer_separation_loss = torch.zeros(1).cuda() 
        
        return a_aggregated_quants, v_aggregated_quants, t_aggregated_quants, a_quant_list, v_quant_list,t_quant_list,a_loss, v_loss, t_loss, cmcm_loss, equal_num_list,zero_num_list, same_layer_fusion_loss,adjacent_layer_separation_loss 
    
    
class AVT_MLVQVAE_Encoder(nn.Module):
    def __init__(self, audio_dim, video_dim, text_dim, audio_output_dim, video_output_dim, text_output_dim, n_embeddings, embedding_dim):
        super(AVT_MLVQVAE_Encoder, self).__init__()
        self.video_dim = video_dim
        self.text_dim = text_dim
        self.audio_dim = audio_dim
        self.hidden_dim = embedding_dim
        self.Video_encoder = Video_Encoder(video_dim, video_output_dim)
        self.Audio_encoder = Audio_Encoder(audio_dim, audio_output_dim)
        self.Text_encoder = Text_Encoder(text_dim, text_output_dim)
        
        self.Video_encoder_2 = Simple_Encoder(video_dim, video_dim)# input is club_feature
        self.Audio_encoder_2 = Simple_Encoder(audio_output_dim, audio_output_dim)
        self.Text_encoder_2 = Simple_Encoder(text_output_dim, text_output_dim)
        
        # 统一VQ
        self.Cross_quantizer = Cross_MLVQEmbeddingEMA_AVT(n_embeddings, self.hidden_dim)
        self.video_semantic_encoder = Video_Semantic_Encoder(video_dim)
        self.video_self_att = InternalTemporalRelationModule(input_dim=video_dim, d_model=self.hidden_dim)
        self.text_self_att = InternalTemporalRelationModule(input_dim=text_dim, d_model=self.hidden_dim)
        self.audio_self_att = InternalTemporalRelationModule(input_dim=audio_dim, d_model=self.hidden_dim)
        
        self.video_self_att_2 = InternalTemporalRelationModule(input_dim=video_dim, d_model=self.hidden_dim)
        self.text_self_att_2 = InternalTemporalRelationModule(input_dim=text_output_dim, d_model=self.hidden_dim)
        self.audio_self_att_2 = InternalTemporalRelationModule(input_dim=audio_output_dim, d_model=self.hidden_dim)

    def Audio_VQ_Encoder(self, audio_feat):
        audio_feat = audio_feat.cuda()
        audio_semantic_result = audio_feat.transpose(0, 1).contiguous()
        audio_semantic_result = self.audio_self_att(audio_semantic_result)
        audio_semantic_result = audio_semantic_result.transpose(0, 1).contiguous()
        
        audio_encoder_result = self.Audio_encoder(audio_feat)  # [batch, length, audio_output_dim]
        audio_semantic_result_2 = audio_encoder_result.transpose(0, 1).contiguous()
        audio_semantic_result_2 = self.audio_self_att_2(audio_semantic_result_2)# [length, batch, hidden_dim]
        audio_semantic_result_2 = audio_semantic_result_2.transpose(0, 1).contiguous()  # [batch, length, hidden_dim]
        
        
        audio_vq, indices = self.Cross_quantizer.mlvq_embedding_draw(audio_semantic_result, layer=0)
        audio_vq_2, indices_2 = self.Cross_quantizer.mlvq_embedding_draw(audio_semantic_result_2, layer=1)
        
        return torch.cat([audio_vq, audio_vq_2], dim=2), indices, indices_2
        # return audio_vq

    def Video_VQ_Encoder(self, video_feat):
        video_feat = video_feat.cuda()
        video_semantic_result = self.video_semantic_encoder(video_feat).transpose(0, 1).contiguous()
        video_semantic_result = self.video_self_att(video_semantic_result)
        video_semantic_result = video_semantic_result.transpose(0, 1).contiguous()  # [batch, length, hidden_dim]
        
        video_encoder_result, video_club_feature = self.Video_encoder(video_feat)
        video_semantic_result_2 = video_club_feature.transpose(0, 1).contiguous()
        video_semantic_result_2 = self.video_self_att_2(video_semantic_result_2)
        video_semantic_result_2 = video_semantic_result_2.transpose(0, 1).contiguous()  # [batch, length, hidden_dim]
        
        video_vq, indices = self.Cross_quantizer.mlvq_embedding_draw(video_semantic_result, layer=0)
        video_vq_2, indices_2 = self.Cross_quantizer.mlvq_embedding_draw(video_semantic_result_2, layer=1)
        
        return torch.cat([video_vq, video_vq_2], dim=2), indices, indices_2
        # return video_vq

    def Text_VQ_Encoder(self, text_feat):
        text_feat = text_feat.cuda()
        text_semantic_result = text_feat.transpose(0, 1).contiguous()
        text_semantic_result = self.text_self_att(text_semantic_result)
        text_semantic_result = text_semantic_result.transpose(0, 1).contiguous()
        
        text_encoder_result = self.Text_encoder(text_feat)  # [batch, length, text_output_dim]
        text_semantic_result_2 = text_encoder_result.transpose(0, 1).contiguous()
        text_semantic_result_2 = self.text_self_att_2(text_semantic_result_2)# [length, batch, hidden_dim]
        text_semantic_result_2 = text_semantic_result_2.transpose(0, 1).contiguous()  # [batch, length, hidden_dim]
        
        text_vq = self.Cross_quantizer.mlvq_embedding(text_semantic_result, layer=0)
        text_vq_2 = self.Cross_quantizer.mlvq_embedding(text_semantic_result_2, layer=1)
        
        return torch.cat([text_vq, text_vq_2], dim=2)
        # return text_vq

    def forward(self, audio_feat, video_feat, text_feat, epoch):
        video_feat = video_feat.cuda()
        text_feat = text_feat.cuda()
        audio_feat = audio_feat.cuda()
        
        # print('video_feat.shape:',video_feat.shape)# [80, 10, 7, 7, 512]
        # print('video_encoder_result.shape:',video_encoder_result.shape)# [80, 10, 3, 3, 2048]
        # print('video_club_feature.shape:',video_club_feature.shape)# [80, 10, 512]
        
        video_encoder_result, video_club_feature = self.Video_encoder(video_feat)
        text_encoder_result = self.Text_encoder(text_feat)  # [batch, length, text_output_dim]
        audio_encoder_result = self.Audio_encoder(audio_feat)  # [batch, length, audio_output_dim]
        
        video_semantic_result = self.video_semantic_encoder(video_feat).transpose(0, 1).contiguous()
        video_semantic_result = self.video_self_att(video_semantic_result)
        video_semantic_result = video_semantic_result.transpose(0, 1).contiguous()  # [batch, length, hidden_dim]
        
        text_semantic_result = text_feat.transpose(0, 1).contiguous()
        text_semantic_result = self.text_self_att(text_semantic_result)# [length, batch, hidden_dim]
        text_semantic_result = text_semantic_result.transpose(0, 1).contiguous()  # [batch, length, hidden_dim]
        
        audio_semantic_result = audio_feat.transpose(0, 1).contiguous()
        audio_semantic_result = self.audio_self_att(audio_semantic_result)# [length, batch, hidden_dim]
        audio_semantic_result = audio_semantic_result.transpose(0, 1).contiguous()  # [batch, length, hidden_dim]
        
        """layer 2"""
        video_encoder_result_2 = self.Video_encoder_2(video_club_feature)# [batch, length, video_dim] -> [batch, length, video_dim] 
        text_encoder_result_2 = self.Text_encoder_2(text_encoder_result)
        audio_encoder_result_2 = self.Audio_encoder_2(audio_encoder_result)
        
        video_semantic_result_2 = video_club_feature.transpose(0, 1).contiguous()
        video_semantic_result_2 = self.video_self_att_2(video_semantic_result_2)
        video_semantic_result_2 = video_semantic_result_2.transpose(0, 1).contiguous()  # [batch, length, hidden_dim]
        
        text_semantic_result_2 = text_encoder_result.transpose(0, 1).contiguous()
        text_semantic_result_2 = self.text_self_att_2(text_semantic_result_2)# [length, batch, hidden_dim]
        text_semantic_result_2 = text_semantic_result_2.transpose(0, 1).contiguous()  # [batch, length, hidden_dim]
        
        audio_semantic_result_2 = audio_encoder_result.transpose(0, 1).contiguous()
        audio_semantic_result_2 = self.audio_self_att_2(audio_semantic_result_2)# [length, batch, hidden_dim]
        audio_semantic_result_2 = audio_semantic_result_2.transpose(0, 1).contiguous()  # [batch, length, hidden_dim]

        audio_vq, video_vq, text_vq, audio_embedding_loss, video_embedding_loss, text_embedding_loss, cmcm_loss, equal_num \
            = self.Cross_quantizer(audio_semantic_result, video_semantic_result, text_semantic_result, epoch, layer=0)
        #
        audio_vq_2, video_vq_2, text_vq_2, audio_embedding_loss_2, video_embedding_loss_2, text_embedding_loss_2, cmcm_loss_2, equal_num_2 \
            = self.Cross_quantizer(audio_semantic_result_2, video_semantic_result_2, text_semantic_result_2, epoch, layer=1)
            
        # audio_embedding_loss_2, video_embedding_loss_2, text_embedding_loss_2, cmcm_loss_2, equal_num_2 \
        #     = torch.zeros(1).cuda(), torch.zeros(1).cuda(), torch.zeros(1).cuda(), torch.zeros(1).cuda(), torch.zeros(1).cuda()
        # cmcm_loss_2 = cmcm_loss

        return audio_semantic_result, video_semantic_result, text_semantic_result, audio_semantic_result_2, video_semantic_result_2, text_semantic_result_2,\
               audio_encoder_result, video_encoder_result, video_club_feature, text_encoder_result, audio_encoder_result_2, video_encoder_result_2, text_encoder_result_2, \
               audio_vq, video_vq, text_vq, audio_vq_2, video_vq_2, text_vq_2, \
               audio_embedding_loss, video_embedding_loss, text_embedding_loss, audio_embedding_loss_2, video_embedding_loss_2, text_embedding_loss_2,\
               cmcm_loss, equal_num, cmcm_loss_2, equal_num_2
           
class AVT_MLVQVAE_Decoder(nn.Module):
    def __init__(self, audio_dim, video_dim, text_dim, audio_output_dim, video_output_dim, text_output_dim):
        super(AVT_MLVQVAE_Decoder, self).__init__()
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
        # self.video_semantic_decoder = Semantic_Decoder(self.hidden_dim, class_num=141)
        # self.text_semantic_decoder = Semantic_Decoder(self.hidden_dim, class_num=141)
        # self.audio_semantic_decoder = Semantic_Decoder(self.hidden_dim, class_num=141)
        
        self.Video_decoder_2 = Simple_Decoder(video_dim, video_dim, self.hidden_dim)# club_feature
        self.Audio_decoder_2 = Simple_Decoder(audio_output_dim, audio_output_dim, self.hidden_dim)
        self.Text_decoder_2 = Simple_Decoder(text_output_dim, text_output_dim, self.hidden_dim)

    def forward(self, audio_feat, video_feat, text_feat, audio_encoder_result, video_encoder_result, text_encoder_result, audio_vq, video_vq, text_vq,
                video_club_feature, audio_encoder_result_2, video_encoder_result_2, text_encoder_result_2, audio_vq_2, video_vq_2, text_vq_2):
        video_feat = video_feat.cuda()
        text_feat = text_feat.cuda()
        audio_feat = audio_feat.cuda()
        video_recon_result = self.Video_decoder(video_encoder_result, video_vq)
        text_recon_result = self.Text_decoder(text_encoder_result, text_vq)
        audio_recon_result = self.Audio_decoder(audio_encoder_result, audio_vq)
        video_recon_loss = F.mse_loss(video_recon_result, video_feat)
        text_recon_loss = F.mse_loss(text_recon_result, text_feat)
        audio_recon_loss = F.mse_loss(audio_recon_result, audio_feat)
        # video_class = self.video_semantic_decoder(video_vq)
        # text_class = self.text_semantic_decoder(text_vq)
        # audio_class = self.audio_semantic_decoder(audio_vq)
        
        """layer 2"""
        video_recon_result_2 = self.Video_decoder_2(video_encoder_result_2, video_vq_2)
        text_recon_result_2 = self.Text_decoder_2(text_encoder_result_2, text_vq_2)
        audio_recon_result_2 = self.Audio_decoder_2(audio_encoder_result_2, audio_vq_2)
        video_recon_loss_2 = F.mse_loss(video_recon_result_2, video_club_feature)
        text_recon_loss_2 = F.mse_loss(text_recon_result_2, text_encoder_result)
        audio_recon_loss_2 = F.mse_loss(audio_recon_result_2, audio_encoder_result)
        

        return audio_recon_loss, video_recon_loss, text_recon_loss, audio_recon_loss_2, video_recon_loss_2, text_recon_loss_2
              
class Cross_MLVQEmbeddingEMA_AVT(nn.Module):
    def __init__(self, n_embeddings, embedding_dim, codebook_size = 2, shared_codebook=False, commitment_cost=0.25, decay=0.99, epsilon=1e-5):
        super(Cross_MLVQEmbeddingEMA_AVT, self).__init__()
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon
        self.ml_size = 2# multi_layer
        self.rl_size = 1# residual_layer
        self.codebook_size = codebook_size# 2*2
        self.shared_codebook = shared_codebook
        
        from typing import Iterable
        n_embeddings = n_embeddings if isinstance(n_embeddings, Iterable) else [n_embeddings for _ in range(self.codebook_size)]
            
        init_bound = 1.0 / 400.0
        embedding = [torch.Tensor(n_embeddings[i], embedding_dim).uniform_(-init_bound, init_bound) for i in range(self.codebook_size)]
        # address 0 is always 0
        # for i in range(self.codebook_size):
        #     embedding[i][0] = 0
        self.register_buffer("embedding",torch.stack(embedding))
        
        ema_count = [torch.zeros(n_embeddings[i]) for i in range(self.codebook_size)]
        self.register_buffer("ema_count", torch.stack(ema_count))
        
        self.register_buffer("ema_weight", self.embedding.clone())
        
        unactivated_count = [-torch.ones(n_embeddings[i]) for i in range(self.codebook_size)]
        self.register_buffer("unactivated_count", torch.stack(unactivated_count))
        # embedding[i*int(not self.shared_codebook)]    

    
    def mlvq_embedding(self, semantic, layer):
        B, T, D = semantic.size()
        flat = semantic.detach().reshape(-1, D)
        distance = torch.addmm(torch.sum(self.embedding[layer] ** 2, dim=1) +
                                 torch.sum(flat**2, dim=1, keepdim=True),
                                 flat, self.embedding[layer].t(),
                                 alpha=-2.0, beta=1.0)
        indices = torch.argmin(distance.double(), dim=-1)
        quantized = F.embedding(indices, self.embedding[layer])
        quantized = quantized.view_as(semantic)
        quantized = semantic + (quantized - semantic).detach()
        return quantized
    
    def mlvq_embedding_draw(self, semantic, layer):
        B, T, D = semantic.size()
        flat = semantic.detach().reshape(-1, D)
        distance = torch.addmm(torch.sum(self.embedding[layer] ** 2, dim=1) +
                                 torch.sum(flat**2, dim=1, keepdim=True),
                                 flat, self.embedding[layer].t(),
                                 alpha=-2.0, beta=1.0)
        indices = torch.argmin(distance.double(), dim=-1)
        quantized = F.embedding(indices, self.embedding[layer])
        quantized = quantized.view_as(semantic)
        quantized = semantic + (quantized - semantic).detach()
        return quantized, indices
    
    def mlrq_embedding(self, semantic, layer):
        B, T, D = semantic.size()
        residual_feature = semantic.detach().clone()
        aggregated_quants = torch.zeros_like(residual_feature)
        for i in range(self.rl_size):# 2层残差，先这么写着，之后要开源再换成codebook_size = multi_layer_size * residual_size
            distances = torch.addmm(torch.sum(self.embedding[layer*self.rl_size+i]  ** 2, dim=1) +
                                    torch.sum(residual_feature.reshape(-1, D) ** 2, dim=1, keepdim=True),
                                    residual_feature.reshape(-1, D), self.embedding[layer*self.rl_size+i].t(),
                                    alpha=-2.0, beta=1.0)# [BxT, M]
            indices = torch.argmin(distances.double(), dim=-1)# [BxT,1]
            quantized = F.embedding(indices, self.embedding[layer*self.rl_size+i]).view_as(semantic)# [BxT,D]->[B,T,D]   
            residual_feature = residual_feature - quantized
            aggregated_quants = aggregated_quants + quantized
        
        aggregated_quants = semantic + (aggregated_quants - semantic).detach()#[B,T,D]
        return aggregated_quants
        
    
    """MLVQ"""
    def forward(self, audio_semantic, video_semantic, text_semantic, epoch, layer):
        M, D = self.embedding[layer].size()
        B, T, D = audio_semantic.size()
        
        a_flat = audio_semantic.detach().reshape(-1, D)  # [B, T, D] -> [BxT, D]
        v_flat = video_semantic.detach().reshape(-1, D)  # [B, T, D] -> [BxT, D]
        t_flat = text_semantic.detach().reshape(-1, D)  # [B, T, D] -> [BxT, D]
        
        """cmcm loss"""
        cmcm_loss = torch.zeros(1).cuda()

        # M:512  B:batchsize  T:10
        # b * mat + a * (mat1@mat2) ([M,] + [BxT,1]) - 2*([BxT,D]@[D,M]) =
        a_distances = torch.addmm(torch.sum(self.embedding[layer] ** 2, dim=1) +
                                  torch.sum(a_flat ** 2, dim=1, keepdim=True),
                                  a_flat, self.embedding[layer].t(),
                                  alpha=-2.0, beta=1.0)  # [BxT, M]

        v_distances = torch.addmm(torch.sum(self.embedding[layer] ** 2, dim=1) +
                                  torch.sum(v_flat ** 2, dim=1, keepdim=True),
                                  v_flat, self.embedding[layer].t(),
                                  alpha=-2.0, beta=1.0)  # [BxT, M]
        
        t_distances = torch.addmm(torch.sum(self.embedding[layer] ** 2, dim=1) +
                                  torch.sum(t_flat ** 2, dim=1, keepdim=True),
                                  t_flat, self.embedding[layer].t(),
                                  alpha=-2.0, beta=1.0)  # [BxT, M]

        a_indices = torch.argmin(a_distances.double(), dim=-1)  # [BxT,1]
        # 
        # print('d:',a_distances[0,:10])
        # a_encodings = torch.ones(B*T,M).cuda() - torch.nn.functional.normalize(a_distances, dim=1)
        # print('e:',a_encodings[0,:10])
        a_encodings = F.one_hot(a_indices, M).double()  # [BxT, M]
        a_quantized = F.embedding(a_indices, self.embedding[layer])  
        a_quantized = a_quantized.view_as(audio_semantic)  # [BxT,D]->[B,T,D]

        v_indices = torch.argmin(v_distances.double(), dim=-1)  # [BxT,1]
        # v_encodings = torch.ones(B*T,M).cuda() - torch.nn.functional.normalize(v_distances, dim=1)
        v_encodings = F.one_hot(v_indices, M).double()  # [BxT, M]
        v_quantized = F.embedding(v_indices, self.embedding[layer])
        v_quantized = v_quantized.view_as(video_semantic)  # [BxT,D]->[B,T,D]
        
        t_indices = torch.argmin(t_distances.double(), dim=-1)  # [BxT,1]
        # t_encodings = torch.ones(B*T,M).cuda() - torch.nn.functional.normalize(t_distances, dim=1)
        t_encodings = F.one_hot(t_indices, M).double()  # [BxT, M]
        t_quantized = F.embedding(t_indices, self.embedding[layer])  
        t_quantized = t_quantized.view_as(text_semantic)  # [BxT,D]->[B,T,D]

        """equal_num"""
        a_indices_reshape = a_indices.reshape(B, T)
        v_indices_reshape = v_indices.reshape(B, T)
        t_indices_reshape = t_indices.reshape(B, T)
        a_indices_mode = torch.mode(a_indices_reshape, dim=-1, keepdim=False)
        v_indices_mode = torch.mode(v_indices_reshape, dim=-1, keepdim=False)
        t_indices_mode = torch.mode(t_indices_reshape, dim=-1, keepdim=False)

        equal_item = (a_indices_mode.values == v_indices_mode.values) & (a_indices_mode.values == t_indices_mode.values)
        equal_num = equal_item.sum()
            
        """MM-EMA"""
        if self.training:
            # audio
            self.ema_count[layer] = self.decay * self.ema_count[layer] + (1 - self.decay) * torch.sum(a_encodings, dim=0)
            n = torch.sum(self.ema_count[layer])
            self.ema_count[layer] = (self.ema_count[layer] + self.epsilon) / (n + M * self.epsilon) * n
            a_dw = torch.matmul(a_encodings.t(), a_flat)# [M, BxT] x [BxT, D] -> [M ,D]
            av_dw = torch.matmul(a_encodings.t(), v_flat)
            at_dw = torch.matmul(a_encodings.t(), t_flat)

            self.ema_weight[layer] = self.decay * self.ema_weight[layer] + 0.5*(1 - self.decay) * a_dw + 0.25*(1 - self.decay) * av_dw + 0.25*(1 - self.decay) * at_dw
            self.embedding[layer] = self.ema_weight[layer] / self.ema_count[layer].unsqueeze(-1)

            # video
            self.ema_count[layer] = self.decay * self.ema_count[layer] + (1 - self.decay) * torch.sum(v_encodings, dim=0)
            n = torch.sum(self.ema_count[layer])
            self.ema_count[layer] = (self.ema_count[layer] + self.epsilon) / (n + M * self.epsilon) * n
            v_dw = torch.matmul(v_encodings.t(), v_flat)
            va_dw = torch.matmul(v_encodings.t(), a_flat)
            vt_dw = torch.matmul(v_encodings.t(), t_flat)

            self.ema_weight[layer] = self.decay * self.ema_weight[layer] + 0.5*(1 - self.decay) * v_dw + 0.25*(1 - self.decay) * va_dw + 0.25*(1 - self.decay) * vt_dw
            self.embedding[layer] = self.ema_weight[layer] / self.ema_count[layer].unsqueeze(-1)
            
            # text
            self.ema_count[layer] = self.decay * self.ema_count[layer] + (1 - self.decay) * torch.sum(t_encodings, dim=0)
            n = torch.sum(self.ema_count[layer])
            self.ema_count[layer] = (self.ema_count[layer] + self.epsilon) / (n + M * self.epsilon) * n
            t_dw = torch.matmul(t_encodings.t(), t_flat)
            ta_dw = torch.matmul(t_encodings.t(), a_flat)
            tv_dw = torch.matmul(t_encodings.t(), v_flat)
            
            self.ema_weight[layer] = self.decay * self.ema_weight[layer] + 0.5*(1 - self.decay) * t_dw + 0.25*(1 - self.decay) * ta_dw + 0.25*(1 - self.decay) * tv_dw
            self.embedding[layer] = self.ema_weight[layer] / self.ema_count[layer].unsqueeze(-1)


        """code reset"""
        self.unactivated_count[layer] = self.unactivated_count[layer] + 1
        for indice in a_indices:
            self.unactivated_count[layer][indice.item()] = 0
        for indice in v_indices:
            self.unactivated_count[layer][indice.item()] = 0
        for indice in t_indices:
            self.unactivated_count[layer][indice.item()] = 0
        activated_indices = []
        unactivated_indices = []
        for i, x in enumerate(self.unactivated_count[layer]):
            if x > 300:
                unactivated_indices.append(i)
                self.unactivated_count[layer][i] = 0
            elif x >= 0 and x < 100:
                activated_indices.append(i)
        activated_quantized = F.embedding(torch.tensor(activated_indices,dtype=torch.int32).cuda(), self.embedding[layer])
        for i in unactivated_indices:
            self.embedding[layer][i] = activated_quantized[random.randint(0,len(activated_indices)-1)] + torch.Tensor(256).uniform_(-1/1024, -1/1024).cuda()

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
        
        return a_quantized, v_quantized, t_quantized, a_loss, v_loss, t_loss, cmcm_loss, equal_num

    """MLRQ"""
    # def forward(self, audio_semantic, video_semantic, text_semantic, epoch, layer):
        
    #     M, D = self.embedding[0].size()
    #     B, T, D = audio_semantic.size()

    #     # a_aggregated_quants, a_quant_list, a_indice_list, a_encoding_list, a_residual_list = self.rq_embedding(audio_semantic.detach().clone())
    #     # v_aggregated_quants, v_quant_list, v_indice_list, v_encoding_list, v_residual_list = self.rq_embedding(video_semantic.detach().clone())
    #     # t_aggregated_quants, t_quant_list, t_indice_list, t_encoding_list, t_residual_list = self.rq_embedding(text_semantic.detach().clone())
        
        
    #     # if beam_size == 1:
    #     """residual_quantize"""
    #     a_residual_feature = audio_semantic.detach().clone()
    #     a_residual_list = []
    #     a_quant_list = []
    #     a_indice_list = []
    #     a_encoding_list = []
    #     a_aggregated_quants = torch.zeros_like(a_residual_feature)
        
    #     v_residual_feature = video_semantic.detach().clone()
    #     v_residual_list = []
    #     v_quant_list = []
    #     v_indice_list = []
    #     v_encoding_list = []
    #     v_aggregated_quants = torch.zeros_like(v_residual_feature)
        
    #     t_residual_feature = text_semantic.detach().clone()
    #     t_residual_list = []
    #     t_quant_list = []
    #     t_indice_list = []
    #     t_encoding_list = []
    #     t_aggregated_quants = torch.zeros_like(t_residual_feature)
        
        
    #     """differ quantize"""
    #     for i in range(self.rl_size):
    #         """audio"""
    #         j = layer*self.rl_size+i
    #         distances = torch.addmm(torch.sum(self.embedding[j]  ** 2, dim=1) +
    #                             torch.sum(a_residual_feature.reshape(-1, D) ** 2, dim=1, keepdim=True),
    #                             a_residual_feature.reshape(-1, D), self.embedding[j].t(),
    #                             alpha=-2.0, beta=1.0)# [BxT, M]
            
    #         indices = torch.argmin(distances.double(), dim=-1)# [BxT,1]
            
    #         """autodeep"""
    #         # min_distances = torch.min(distances, dim=-1)[0]  
    #         # indices = torch.argmin(distances.double(), dim=-1)
    #         
    #         # mask = min_distances > 0.95 * distances[:,0]  
    #         # indices[mask] = 0
            
    #         encodings = F.one_hot(indices, M).double()  # [BxT, M]
    #         quantized = F.embedding(indices, self.embedding[j]).view_as(audio_semantic)# [BxT,D]->[B,T,D]
    #         a_residual_list.append(a_residual_feature.clone().reshape(-1, D))  # [B, T, D] -> [BxT, D] -> [codebook_size,BxT, D]
    #         a_residual_feature = a_residual_feature - quantized
    #         a_aggregated_quants = a_aggregated_quants + quantized
    #         # residual_feature.sub_(quantized)#[B,T,D]
    #         # aggregated_quants.add_(quantized)#[B,T,D]
    #         a_quant_list.append(a_aggregated_quants.clone())#[codebook_size,B,T,D]
    #         a_indice_list.append(indices.clone())#[codebook_size,BxT,1]
    #         a_encoding_list.append(encodings.clone())#[codebook_size,BxT,M]
    #         """video"""
    #         distances = torch.addmm(torch.sum(self.embedding[j]  ** 2, dim=1) +
    #                             torch.sum(v_residual_feature.reshape(-1, D) ** 2, dim=1, keepdim=True),
    #                             v_residual_feature.reshape(-1, D), self.embedding[j].t(),
    #                             alpha=-2.0, beta=1.0)# [BxT, M]
            
    #         indices = torch.argmin(distances.double(), dim=-1)# [BxT,1]
            
    #         """autodeep"""
    #         # min_distances = torch.min(distances, dim=-1)[0]  
    #         # indices = torch.argmin(distances.double(), dim=-1)
    #         
    #         # mask = min_distances > 0.95 * distances[:,0]  
    #         # indices[mask] = 0
            
    #         encodings = F.one_hot(indices, M).double()  # [BxT, M]
    #         quantized = F.embedding(indices, self.embedding[j]).view_as(video_semantic)# [BxT,D]->[B,T,D]
    #         v_residual_list.append(v_residual_feature.clone().reshape(-1, D))  # [B, T, D] -> [BxT, D] -> [codebook_size,BxT, D]
    #         v_residual_feature = v_residual_feature - quantized
    #         v_aggregated_quants = v_aggregated_quants + quantized
    #         # residual_feature.sub_(quantized)#[B,T,D]
    #         # aggregated_quants.add_(quantized)#[B,T,D]
    #         v_quant_list.append(v_aggregated_quants.clone())#[codebook_size,B,T,D]
    #         v_indice_list.append(indices.clone())#[codebook_size,BxT,1]
    #         v_encoding_list.append(encodings.clone())#[codebook_size,BxT,M]
    #         """text"""
    #         distances = torch.addmm(torch.sum(self.embedding[j]  ** 2, dim=1) +
    #                             torch.sum(t_residual_feature.reshape(-1, D) ** 2, dim=1, keepdim=True),
    #                             t_residual_feature.reshape(-1, D), self.embedding[j].t(),
    #                             alpha=-2.0, beta=1.0)# [BxT, M]
            
    #         indices = torch.argmin(distances.double(), dim=-1)# [BxT,1]
            
    #         """autodeep"""
    #         # min_distances = torch.min(distances, dim=-1)[0]  
    #         # indices = torch.argmin(distances.double(), dim=-1)
    #         
    #         # mask = min_distances > 0.95 * distances[:,0]  
    #         # indices[mask] = 0
            
    #         encodings = F.one_hot(indices, M).double()  # [BxT, M]
    #         quantized = F.embedding(indices, self.embedding[j]).view_as(text_semantic)# [BxT,D]->[B,T,D]
    #         t_residual_list.append(t_residual_feature.clone().reshape(-1, D))  # [B, T, D] -> [BxT, D] -> [codebook_size,BxT, D]
    #         t_residual_feature = t_residual_feature - quantized
    #         t_aggregated_quants = t_aggregated_quants + quantized
    #         # residual_feature.sub_(quantized)#[B,T,D]
    #         # aggregated_quants.add_(quantized)#[B,T,D]
    #         t_quant_list.append(t_aggregated_quants.clone())#[codebook_size,B,T,D]
    #         t_indice_list.append(indices.clone())#[codebook_size,BxT,1]
    #         t_encoding_list.append(encodings.clone())#[codebook_size,BxT,M]


    #     """compute equal_num"""
    #     equal_num_list = []
    #     for i in range(self.rl_size):
    #         a_indices_reshape = a_indice_list[i].reshape(B, T)
    #         v_indices_reshape = v_indice_list[i].reshape(B, T)
    #         t_indices_reshape = t_indice_list[i].reshape(B, T)
    #         a_indices_mode = torch.mode(a_indices_reshape, dim=-1, keepdim=False)#返回众数
    #         v_indices_mode = torch.mode(v_indices_reshape, dim=-1, keepdim=False)
    #         t_indices_mode = torch.mode(t_indices_reshape, dim=-1, keepdim=False)

    #         equal_item = (a_indices_mode.values == v_indices_mode.values) & (a_indices_mode.values == t_indices_mode.values)
    #         equal_num = int(equal_item.sum())
    #         equal_num_list.append(equal_num)
            
    #     """differ MM-EMA update"""
    #     if self.training:
    #         for i in range(self.rl_size):
    #             j = layer*self.rl_size+i 
    #             # audio
    #             self.ema_count[j] = self.decay * self.ema_count[j] + (1 - self.decay) * torch.sum(a_encoding_list[i], dim=0)
    #             n = torch.sum(self.ema_count[j])
    #             self.ema_count[j] = (self.ema_count[j] + self.epsilon) / (n + M * self.epsilon) * n
    #             a_dw = torch.matmul(a_encoding_list[i].t(), a_residual_list[i])
    #             av_dw = torch.matmul(a_encoding_list[i].t(), v_residual_list[i])
    #             at_dw = torch.matmul(a_encoding_list[i].t(), t_residual_list[i])
    #             self.ema_weight[j] = self.decay * self.ema_weight[j] + 0.5*(1 - self.decay) * a_dw + 0.25*(1 - self.decay) * av_dw + 0.25*(1 - self.decay) * at_dw
                
    #             # self.ema_weight[j] = self.decay * self.ema_weight[j] 
    #             self.embedding[j] = self.ema_weight[j] / self.ema_count[j].unsqueeze(-1)

    #             # video
    #             self.ema_count[j] = self.decay * self.ema_count[j] + (1 - self.decay) * torch.sum(v_encoding_list[i], dim=0)
    #             n = torch.sum(self.ema_count[j])
    #             self.ema_count[j] = (self.ema_count[j] + self.epsilon) / (n + M * self.epsilon) * n
    #             v_dw = torch.matmul(v_encoding_list[i].t(), v_residual_list[i])
    #             va_dw = torch.matmul(v_encoding_list[i].t(), a_residual_list[i])
    #             vt_dw = torch.matmul(v_encoding_list[i].t(), t_residual_list[i])
    #             self.ema_weight[j] = self.decay * self.ema_weight[j] + 0.5*(1 - self.decay) * v_dw + 0.25*(1 - self.decay) * va_dw + 0.25*(1 - self.decay) * vt_dw
                
    #             # self.ema_weight[j] = self.decay * self.ema_weight[j]
    #             self.embedding[j] = self.ema_weight[j] / self.ema_count[j].unsqueeze(-1)
                
    #             # text
    #             self.ema_count[j] = self.decay * self.ema_count[j] + (1 - self.decay) * torch.sum(t_encoding_list[i], dim=0)
    #             n = torch.sum(self.ema_count[j])
    #             self.ema_count[j] = (self.ema_count[j] + self.epsilon) / (n + M * self.epsilon) * n
    #             t_dw = torch.matmul(t_encoding_list[i].t(), t_residual_list[i])
    #             ta_dw = torch.matmul(t_encoding_list[i].t(), a_residual_list[i])
    #             tv_dw = torch.matmul(t_encoding_list[i].t(), v_residual_list[i])
    #             self.ema_weight[j] = self.decay * self.ema_weight[j] + 0.5*(1 - self.decay) * t_dw + 0.25*(1 - self.decay) * ta_dw + 0.25*(1 - self.decay) * tv_dw
                
    #             # self.ema_weight[j] = self.decay * self.ema_weight[j]
    #             self.embedding[j] = self.ema_weight[j] / self.ema_count[j].unsqueeze(-1)
                

    #     """reset code"""
    #     for i in range(self.rl_size):
    #         j = layer*self.rl_size+i
    #         self.unactivated_count[j] = self.unactivated_count[j] + 1
    #         """differ quantize"""
    #         for indice in a_indice_list[i]:
    #             self.unactivated_count[j][indice.item()] = 0
    #         for indice in v_indice_list[i]:
    #             self.unactivated_count[j][indice.item()] = 0
    #         for indice in t_indice_list[i]:
    #             self.unactivated_count[j][indice.item()] = 0
    #         """same quantize"""
    #         # for indice in indice_list[i]:
    #         #     self.unactivated_count[j][indice.item()] = 0  
            
    #         activated_indices = []
    #         unactivated_indices = []
    #         for k, x in enumerate(self.unactivated_count[j]):
    #             if x > 300:
    #                 unactivated_indices.append(k)
    #                 self.unactivated_count[j][k] = 0
    #             elif x >= 0 and x < 100:
    #                 activated_indices.append(k)
    #         activated_quantized = F.embedding(torch.tensor(activated_indices,dtype=torch.int32).cuda(), self.embedding[j])
    #         for k in unactivated_indices:
    #             self.embedding[j][k] = activated_quantized[random.randint(0,len(activated_indices)-1)] + torch.Tensor(int(D)).uniform_(-1.0/400.0, -1.0/400.0).cuda()
        
    #     # address 0 is always 0
    #     # for i in range(self.codebook_size):
    #     #     self.embedding[i][0] = 0

    #     """cmcm_loss"""
    #     cmcm_loss = torch.zeros(1).cuda()

    #     """differ quantize"""
    #     a_e_latent_loss = F.mse_loss(audio_semantic, a_aggregated_quants.detach())
    #     av_e_latent_loss = F.mse_loss(audio_semantic, v_aggregated_quants.detach())
    #     at_e_latent_loss = F.mse_loss(audio_semantic, t_aggregated_quants.detach())

    #     a_loss = self.commitment_cost * 2.0 * a_e_latent_loss + 0.5*self.commitment_cost * av_e_latent_loss + 0.5*self.commitment_cost * at_e_latent_loss
        
    #     v_e_latent_loss = F.mse_loss(video_semantic, v_aggregated_quants.detach())
    #     va_e_latent_loss = F.mse_loss(video_semantic, a_aggregated_quants.detach())
    #     vt_e_latent_loss = F.mse_loss(video_semantic, t_aggregated_quants.detach())

    #     v_loss = self.commitment_cost * 2.0 * v_e_latent_loss + 0.5*self.commitment_cost * va_e_latent_loss + 0.5*self.commitment_cost * vt_e_latent_loss
        
    #     t_e_latent_loss = F.mse_loss(text_semantic, t_aggregated_quants.detach())
    #     ta_e_latent_loss = F.mse_loss(text_semantic, a_aggregated_quants.detach())
    #     tv_e_latent_loss = F.mse_loss(text_semantic, v_aggregated_quants.detach())

    #     t_loss = self.commitment_cost * 2.0 * t_e_latent_loss + 0.5*self.commitment_cost * ta_e_latent_loss + 0.5*self.commitment_cost * tv_e_latent_loss
        
        
    #     """same quantize"""
    #     # a_e_latent_loss = F.mse_loss(audio_semantic, a_aggregated_quants.detach())

    #     # a_loss = self.commitment_cost * 2.0 * a_e_latent_loss
        
    #     # v_e_latent_loss = F.mse_loss(video_semantic, v_aggregated_quants.detach())

    #     # v_loss = self.commitment_cost * 2.0 * v_e_latent_loss
        
    #     # t_e_latent_loss = F.mse_loss(text_semantic, t_aggregated_quants.detach())

    #     # t_loss = self.commitment_cost * 2.0 * t_e_latent_loss
        
    #     """new-embedding-loss"""
    #     # a_e_latent_loss,av_e_latent_loss,at_e_latent_loss,v_e_latent_loss,va_e_latent_loss,vt_e_latent_loss,t_e_latent_loss,ta_e_latent_loss,tv_e_latent_loss = \
    #     #     torch.zeros(1).cuda(),torch.zeros(1).cuda(),torch.zeros(1).cuda(),torch.zeros(1).cuda(),torch.zeros(1).cuda(),torch.zeros(1).cuda(),torch.zeros(1).cuda(),torch.zeros(1).cuda(),torch.zeros(1).cuda()
            
    #     # for i in range(self.codebook_size):
            
    #     #     a_e_latent_loss += F.mse_loss(audio_semantic, a_quant_list[i].detach())
    #     #     av_e_latent_loss += F.mse_loss(audio_semantic, v_quant_list[i].detach())
    #     #     at_e_latent_loss += F.mse_loss(audio_semantic, t_quant_list[i].detach())
        
    #     #     v_e_latent_loss += F.mse_loss(video_semantic, v_quant_list[i].detach())
    #     #     va_e_latent_loss += F.mse_loss(video_semantic, a_quant_list[i].detach())
    #     #     vt_e_latent_loss += F.mse_loss(video_semantic, t_quant_list[i].detach())

    #     #     t_e_latent_loss += F.mse_loss(text_semantic, t_quant_list[i].detach())
    #     #     ta_e_latent_loss += F.mse_loss(text_semantic, a_quant_list[i].detach())
    #     #     tv_e_latent_loss += F.mse_loss(text_semantic, v_quant_list[i].detach())

            
            
    #     # a_loss = self.commitment_cost * 2.0 * a_e_latent_loss + 0.5*self.commitment_cost * av_e_latent_loss + 0.5*self.commitment_cost * at_e_latent_loss
    #     # v_loss = self.commitment_cost * 2.0 * v_e_latent_loss + 0.5*self.commitment_cost * va_e_latent_loss + 0.5*self.commitment_cost * vt_e_latent_loss
    #     # t_loss = self.commitment_cost * 2.0 * t_e_latent_loss + 0.5*self.commitment_cost * ta_e_latent_loss + 0.5*self.commitment_cost * tv_e_latent_loss
            
    #     a_aggregated_quants = audio_semantic + (a_aggregated_quants - audio_semantic).detach()#[B,T,D]
    #     v_aggregated_quants = video_semantic + (v_aggregated_quants - video_semantic).detach()#[B,T,D]
    #     t_aggregated_quants = text_semantic + (t_aggregated_quants - text_semantic).detach()#[B,T,D]
        
    #     return a_aggregated_quants, v_aggregated_quants, t_aggregated_quants, a_loss, v_loss, t_loss, cmcm_loss, equal_num_list
