import torch
import torch.nn as nn

import torch
import copy
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import Module
from torch.nn import MultiheadAttention
from torch.nn import ModuleList
from torch.nn.init import xavier_uniform_
from torch.nn import Dropout
from torch.nn import Linear
from torch.nn import LayerNorm

class Encoder(nn.Module):
    r"""Encoder is a stack of N encoder layers

    Args:
        encoder_layer: an instance of the EncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).

    """

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(Encoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src):
        r"""Pass the input through the endocder layers in turn.

        """
        output = src

        for i in range(self.num_layers):
            output = self.layers[i](output)

        if self.norm:
            output = self.norm(output)

        return output


class Decoder(Module):
    r"""Decoder is a stack of N decoder layers

    Args:
        decoder_layer: an instance of the DecoderLayer() class (required).
        num_layers: the number of sub-decoder-layers in the decoder (required).
        norm: the layer normalization component (optional).
    """

    def __init__(self, decoder_layer, num_layers, norm=None):
        super(Decoder, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, tgt, memory):
        r"""Pass the inputs (and mask) through the decoder layer in turn.
        """
        output = tgt

        for i in range(self.num_layers):
            output = self.layers[i](output, memory)

        if self.norm:
            output = self.norm(output)

        return output



class EncoderLayer(Module):
    r"""EncoderLayer is mainly made up of self-attention.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).

    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def forward(self, src):
        r"""Pass the input through the endocder layer.
        """
        src2 = self.self_attn(src, src, src)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        if hasattr(self, "activation"):
            src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        else:  # for backward compatibility
            src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class DecoderLayer(Module):
    r"""DecoderLayer is mainly made up of the proposed cross-modal relation attention (CMRA).

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).

    """

    def __init__(self, d_model, nhead, dim_feedforward=1024, dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def forward(self, tgt, memory):
        r"""Pass the inputs (and mask) through the decoder layer.
        """
        memory = torch.cat([memory, tgt], dim=0)
        tgt2 = self.multihead_attn(tgt, memory, memory)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        if hasattr(self, "activation"):
            tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        else:  # for backward compatibility
            tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt))))
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        return tgt



def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    else:
        raise RuntimeError("activation should be relu/gelu, not %s." % activation)


class New_Audio_Guided_Attention(nn.Module):
    def __init__(self):
        super(New_Audio_Guided_Attention, self).__init__()
        self.hidden_size = 512
        self.relu = nn.ReLU()
        # channel attention
        self.affine_video_1 = nn.Linear(512, 512)
        self.affine_audio_1 = nn.Linear(128, 512)
        self.affine_bottleneck = nn.Linear(512, 256)
        self.affine_v_c_att = nn.Linear(256, 512)
        # spatial attention
        self.affine_video_2 = nn.Linear(512, 256)
        self.affine_audio_2 = nn.Linear(128, 256)
        self.affine_v_s_att = nn.Linear(256, 1)

        # video-guided audio attention
        self.affine_video_guided_1 = nn.Linear(512, 64)
        self.affine_video_guided_2 = nn.Linear(64, 128)

        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, video, audio):
        '''
        :param visual_feature: [batch, 10, 7, 7, 512]
        :param audio_feature:  [batch, 10, 128]
        :return: [batch, 10, 512]
        '''
        audio = audio.transpose(1, 0)
        batch, t_size, h, w, v_dim = video.size()
        a_dim = audio.size(-1)
        audio_feature = audio.reshape(batch * t_size, a_dim)
        visual_feature = video.reshape(batch, t_size, -1, v_dim)
        raw_visual_feature = visual_feature
        # ============================== Channel Attention ====================================
        audio_query_1 = self.relu(self.affine_audio_1(audio_feature)).unsqueeze(-2)
        video_query_1 = self.relu(self.affine_video_1(visual_feature)).reshape(batch * t_size, h * w, -1)
        audio_video_query_raw = (audio_query_1 * video_query_1).mean(-2)
        audio_video_query = self.relu(self.affine_bottleneck(audio_video_query_raw))
        channel_att_maps = self.affine_v_c_att(audio_video_query).sigmoid().reshape(batch, t_size, -1, v_dim)
        c_att_visual_feat = (raw_visual_feature * (channel_att_maps + 1))  # 加一是残差连接

        # ============================== Spatial Attention =====================================
        # channel attended visual feature: [batch * 10, 49, v_dim]
        c_att_visual_feat = c_att_visual_feat.reshape(batch * t_size, -1, v_dim)
        c_att_visual_query = self.relu(self.affine_video_2(c_att_visual_feat))
        audio_query_2 = self.relu(self.affine_audio_2(audio_feature)).unsqueeze(-2)
        audio_video_query_2 = c_att_visual_query * audio_query_2
        spatial_att_maps = self.softmax(self.tanh(self.affine_v_s_att(audio_video_query_2)).transpose(2, 1))
        c_s_att_visual_feat = torch.bmm(spatial_att_maps, c_att_visual_feat).squeeze().reshape(batch, t_size, v_dim)

        return c_s_att_visual_feat
class RNNEncoder(nn.Module):
    def __init__(self, audio_dim,video_dim, d_model):
        super(RNNEncoder, self).__init__()

        self.d_model = d_model
        self.audio_rnn = nn.LSTM(audio_dim, int(d_model/2), 1, batch_first=True, bidirectional=True, dropout=0.2)
        self.visual_rnn = nn.LSTM(video_dim, int(d_model/2), 1, batch_first=True, bidirectional=True, dropout=0.2)


    def forward(self, audio_feature, visual_feature):
        audio_output,_ = self.audio_rnn(audio_feature)
        video_output,_ = self.visual_rnn(visual_feature)
        return audio_output, video_output


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

class CrossModalRelationAttModule(nn.Module):
    def __init__(self, input_dim, d_model):
        super(CrossModalRelationAttModule, self).__init__()

        self.decoder_layer = DecoderLayer(d_model=d_model, nhead=4)
        self.decoder = Decoder(self.decoder_layer, num_layers=1)

        self.affine_matrix = nn.Linear(input_dim, d_model)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, query_feature, memory_feature):
        query_feature = self.affine_matrix(query_feature)
        output = self.decoder(query_feature, memory_feature)

        return output



class SupvLocalizeModule(nn.Module):
    def __init__(self, d_model):
        super(SupvLocalizeModule, self).__init__()
        # self.affine_concat = nn.Linear(2*256, 256)
        self.relu = nn.ReLU(inplace=True)
        self.classifier = nn.Linear(d_model, 1)  # start and end
        self.event_classifier = nn.Linear(d_model, 28)

    # self.softmax = nn.Softmax(dim=-1)

    def forward(self, fused_content):
        max_fused_content, _ = fused_content.transpose(1, 0).max(1)
        logits = self.classifier(fused_content)
        # scores = self.softmax(logits)
        class_logits = self.event_classifier(max_fused_content)
        class_scores = class_logits

        return logits, class_scores


class AudioVideoInter(nn.Module):
    def __init__(self, d_model, n_head, head_dropout=0.1):
        super(AudioVideoInter, self).__init__()
        self.dropout = nn.Dropout(0.1)
        self.video_multihead = MultiheadAttention(d_model, num_heads=n_head, dropout=head_dropout)
        self.norm1 = nn.LayerNorm(d_model)

    def forward(self, video_feat, audio_feat):
        # video_feat, audio_feat: [10, batch, 256]
        global_feat = video_feat * audio_feat
        memory = torch.cat([audio_feat, video_feat], dim=0)
        mid_out = self.video_multihead(global_feat, memory, memory)[0]
        output = self.norm1(global_feat + self.dropout(mid_out))

        return output

class CAS_Module(nn.Module):
    def __init__(self, d_model, num_class=28):
        super(CAS_Module, self).__init__()
        self.d_model = d_model
        self.num_class = num_class
        self.dropout = nn.Dropout(0.2)

        self.classifier = nn.Sequential(
            nn.Conv1d(in_channels=d_model, out_channels=self.num_class+1, kernel_size=1, stride=1, padding=0, bias=False)
        )

    def forward(self, content):

        content = content.permute(0, 2, 1)  # [32, 256, 10]

        out = self.classifier(content)  # [32, 29, 10]
        out = out.permute(0, 2, 1)
        return out


class weak_main_model(nn.Module):
    def __init__(self):
        super(weak_main_model, self).__init__()
        self.spatial_channel_att = New_Audio_Guided_Attention()
        self.video_input_dim = 512
        self.video_fc_dim = 512
        self.d_model = 256
        self.v_fc = nn.Linear(self.video_input_dim, self.video_fc_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

        self.video_encoder = InternalTemporalRelationModule(input_dim=self.video_fc_dim, d_model=self.d_model)
        self.video_decoder = CrossModalRelationAttModule(input_dim=self.video_fc_dim, d_model=self.d_model)
        self.audio_encoder = InternalTemporalRelationModule(input_dim=128, d_model=self.d_model)
        self.audio_decoder = CrossModalRelationAttModule(input_dim=128, d_model=self.d_model)
        #self.audio_visual_rnn_layer = RNNEncoder(audio_dim=128, video_dim=512, d_model=256, num_layers=1)
        self.AVInter = AudioVideoInter(self.d_model, n_head=2, head_dropout=0.2)
        self.VAInter = AudioVideoInter(self.d_model, n_head=2, head_dropout=0.2)
        #self.localize_module = WeaklyLocalizationModule(self.d_model)
        self.audio_gated = nn.Sequential(
            nn.Linear(256, 1),

            nn.Sigmoid()
        )
        self.video_gated = nn.Sequential(
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        self.CAS_model = CAS_Module(d_model=self.d_model, num_class=28)
        self.classifier = nn.Linear(self.d_model, 1)
        self.softmax = nn.Softmax(dim=-1)
        self.audio_cas = nn.Linear(self.d_model, 29)
        self.video_cas = nn.Linear(self.d_model, 29)

    def forward(self, visual_feature, audio_feature):
        # [batch, 10, 512]
        # this fc is optinal, that is used for adaption of different visual features (e.g., vgg, resnet).
        #audio_rnn_input = audio_feature
        audio_feature = audio_feature.transpose(1, 0).contiguous()
        visual_feature = self.v_fc(visual_feature)
        visual_feature = self.dropout(self.relu(visual_feature))

        # spatial-channel attention
        visual_feature = self.spatial_channel_att(visual_feature, audio_feature)
        visual_feature = visual_feature.transpose(1, 0).contiguous()
        #visual_rnn_input = visual_feature


        # audio_rnn_output1, visual_rnn_output1 = self.audio_visual_rnn_layer(audio_rnn_input, visual_rnn_input)
        # audio_encoder_input1 = audio_rnn_output1.transpose(1, 0).contiguous()  # [10, 32, 256]
        # visual_encoder_input1 = visual_rnn_output1.transpose(1, 0).contiguous()  # [10, 32, 512]

        # audio query
        video_key_value_feature = self.video_encoder(visual_feature)
        audio_query_output = self.audio_decoder(audio_feature, video_key_value_feature)

        # video query
        audio_key_value_feature = self.audio_encoder(audio_feature)
        video_query_output = self.video_decoder(visual_feature, audio_key_value_feature)

        audio_gate = self.audio_gated(video_key_value_feature)
        video_gate = self.video_gated(audio_key_value_feature)

        av_gate = (audio_gate + video_gate) / 2
        av_gate = av_gate.permute(1, 0, 2)

        video_query_output = video_query_output + audio_gate * video_query_output * 0.1
        audio_query_output = audio_query_output + video_gate * audio_query_output * 0.1

        video_cas = self.video_cas(video_query_output)
        audio_cas = self.audio_cas(audio_query_output)
        video_cas = video_cas.permute(1, 0, 2)
        audio_cas = audio_cas.permute(1, 0, 2)

        video_cas_gate = video_cas.sigmoid()
        audio_cas_gate = audio_cas.sigmoid()
        #
        # video_cas_gate = (video_cas_gate > 0.01).float()*video_cas_gate
        # audio_cas_gate = (audio_cas_gate > 0.01).float()*audio_cas_gate

        # video_cas = audio_cas_gate.unsqueeze(1) * video_cas
        # audio_cas = video_cas_gate.unsqueeze(1) * audio_cas
        #
        # sorted_scores_video, _ = video_cas.sort(descending=True, dim=1)
        # topk_scores_video = sorted_scores_video[:, :4, :]
        # score_video = torch.mean(topk_scores_video, dim=1)
        # sorted_scores_audio, _ = audio_cas.sort(descending=True, dim=1)
        # topk_scores_audio = sorted_scores_audio[:, :4, :]
        # score_audio = torch.mean(topk_scores_audio, dim=1)  # [32, 29]
        #
        # video_cas_gate = score_video.sigmoid()
        # audio_cas_gate = score_audio.sigmoid()
        # video_cas_gate = (video_cas_gate > 0.5).float()*video_cas_gate
        # audio_cas_gate = (audio_cas_gate > 0.5).float()*audio_cas_gate

        #
        # av_score = (score_video + score_audio) / 2


        video_query_output = self.AVInter(video_query_output, audio_query_output)
        audio_query_output = self.VAInter(audio_query_output, video_query_output)
        #scores = self.localize_module((video_query_output+audio_query_output)/2)



        fused_content = (video_query_output+audio_query_output)/2
       # fused_content = video_query_output
        fused_content = fused_content.transpose(0, 1)
        #is_event_scores = self.classifier(fused_content)

        cas_score = self.CAS_model(fused_content)
        #cas_score = cas_score + 0.2*video_cas_gate.unsqueeze(1)*cas_score + 0.2*audio_cas_gate.unsqueeze(1)*cas_score
        cas_score = 0.5*video_cas_gate*cas_score + 0.5*audio_cas_gate*cas_score
        #cas_score = cas_score*2
        sorted_scores, _ = cas_score.sort(descending=True, dim=1)
        topk_scores = sorted_scores[:, :4, :]
        raw_logits = torch.mean(topk_scores, dim=1)[:, None, :]       #[32, 29]

        #fused_logits = is_event_scores.sigmoid() * raw_logits
        fused_logits = av_gate * raw_logits
        print("av_gate:",av_gate.size())
        print("raw_logits:",raw_logits.size())
        # fused_scores, _ = fused_logits.sort(descending=True, dim=1)
        # topk_scores = fused_scores[:, :3, :]
        # logits = torch.mean(topk_scores, dim=1)
        # Training: max pooling for adapting labels
        logits, _ = torch.max(fused_logits, dim=1)
        event_scores = self.softmax(logits)

        event_scores = event_scores

        return None




class Audio_Guided_Attention(nn.Module):
    def __init__(self):
        super(Audio_Guided_Attention, self).__init__()
        self.hidden_size = 512
        self.relu = nn.ReLU()
        # channel attention
        self.affine_video_1 = nn.Linear(512, 512)
        self.affine_audio_1 = nn.Linear(128, 512)
        self.affine_bottleneck = nn.Linear(512, 256)
        self.affine_v_c_att = nn.Linear(256, 512)
        # spatial attention
        self.affine_video_2 = nn.Linear(512, 256)
        self.affine_audio_2 = nn.Linear(128, 256)
        self.affine_v_s_att = nn.Linear(256, 1)

        # video-guided audio attention
        self.affine_video_guided_1 = nn.Linear(512, 64)
        self.affine_video_guided_2 = nn.Linear(64, 128)

        self.latent_dim = 2
        self.video_query = nn.Linear(512, 512//self.latent_dim)
        self.video_key = nn.Linear(512, 512//self.latent_dim)
        self.video_value = nn.Linear(512, 512)


        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(0.1)
        self.norm = nn.LayerNorm(512)


    def forward(self, video, audio):
        '''
        :param visual_feature: [batch, 10, 7, 7, 512]
        :param audio_feature:  [batch, 10, 128]
        :return: [batch, 10, 512]
        '''
        audio = audio.transpose(1, 0)
        batch, t_size, h, w, v_dim = video.size()
        a_dim = audio.size(-1)
        audio_feature = audio.reshape(batch * t_size, a_dim)
        visual_feature = video.reshape(batch, t_size, -1, v_dim)
        raw_visual_feature = visual_feature

        # ============================== Self Attention =======================================
        video_query_feature = self.video_query(visual_feature).reshape(batch*t_size, h*w, -1)  #[B, h*w, C]
        video_key_feature = self.video_key(visual_feature).reshape(batch*t_size, h*w, -1).permute(0, 2, 1)  #[B, C, h*w]
        energy = torch.bmm(video_query_feature, video_key_feature)
        attention = self.softmax(energy)
        video_value_feature = self.video_value(visual_feature).reshape(batch*t_size, h*w, -1)
        output = torch.matmul(attention, video_value_feature)
        output = self.norm(visual_feature.reshape(batch*t_size, h*w, -1) + self.dropout(output))
        visual_feature = output

        # ============================== Channel Attention ====================================
        audio_query_1 = self.relu(self.affine_audio_1(audio_feature)).unsqueeze(-2)
        video_query_1 = self.relu(self.affine_video_1(visual_feature)).reshape(batch*t_size, h*w, -1)
        audio_video_query_raw = (audio_query_1 * video_query_1).mean(-2)
        audio_video_query = self.relu(self.affine_bottleneck(audio_video_query_raw))
        channel_att_maps = self.affine_v_c_att(audio_video_query).sigmoid().reshape(batch, t_size, -1, v_dim)
        c_att_visual_feat = (raw_visual_feature * (channel_att_maps + 1))      

        # ============================== Spatial Attention =====================================
        # channel attended visual feature: [batch * 10, 49, v_dim]
        c_att_visual_feat = c_att_visual_feat.reshape(batch*t_size, -1, v_dim)
        c_att_visual_query = self.relu(self.affine_video_2(c_att_visual_feat))
        audio_query_2 = self.relu(self.affine_audio_2(audio_feature)).unsqueeze(-2)
        audio_video_query_2 = c_att_visual_query * audio_query_2
        spatial_att_maps = self.softmax(self.tanh(self.affine_v_s_att(audio_video_query_2)).transpose(2, 1))
        c_s_att_visual_feat = torch.bmm(spatial_att_maps, c_att_visual_feat).squeeze().reshape(batch, t_size, v_dim)

        return c_s_att_visual_feat

model = weak_main_model()
audio = torch.randn(32, 10, 128)
video = torch.randn(32, 10, 7, 7, 512)
model(video, audio)