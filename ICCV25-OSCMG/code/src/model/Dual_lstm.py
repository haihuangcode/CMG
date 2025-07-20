
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
import math
from torch.autograd import Variable



class Dual_lstm_cell(nn.Module):
    def __init__(self, visual_input_dim, audio_input_dim, hidden_dim, alph=0.5, bias=True):
        super(Dual_lstm_cell, self).__init__()

        self.visual_input_dim = visual_input_dim
        self.audio_input_dim  = audio_input_dim
        self.hidden_dim = hidden_dim
        self.alph = alph
        self.vs_linear = nn.Linear(self.visual_input_dim, 4 * self.hidden_dim, bias=bias)
        self.vh_linear = nn.Linear(self.hidden_dim, 4* self.hidden_dim, bias=bias)
        self.as_linear = nn.Linear(self.audio_input_dim, 4 * self.hidden_dim, bias=bias)
        self.ah_linear = nn.Linear(self.hidden_dim, 4 * self.hidden_dim, bias=bias)

        self.as_linear2 = nn.Linear(self.audio_input_dim, 4*self.hidden_dim, bias=bias)
        self.ah_linear2 = nn.Linear(self.hidden_dim, 4*self.hidden_dim, bias=bias)
        self.vs_linear2 = nn.Linear(self.visual_input_dim, 4*self.hidden_dim, bias=bias)
        self.vh_linear2 = nn.Linear(self.hidden_dim, 4*self.hidden_dim, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_dim)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, visual_state, visual_hidden, visual_cell, audio_state, audio_hidden, audio_cell):
        visual_gates = self.vs_linear(visual_state) + self.vh_linear(visual_hidden)
            #self.alph*self.as_linear(audio_state) + self.alph*self.ah_linear(audio_hidden)


        audio_gates = self.as_linear2(audio_state) + self.ah_linear2(audio_hidden)
            #self.alph*self.vs_linear2(visual_state) + self.alph*self.vh_linear2(visual_hidden)

        visual_i_gate, visual_f_gate, visual_c_gate, visual_o_gate = visual_gates.chunk(4,1)
        audio_i_gate, audio_f_gate, audio_c_gate, audio_o_gate = audio_gates.chunk(4,1)

        visual_i_gate = F.sigmoid(visual_i_gate)
        visual_f_gate = F.sigmoid(visual_f_gate)
        visual_c_gate = F.tanh(visual_c_gate)
        visual_o_gate = F.sigmoid(visual_o_gate)

        visual_cell = visual_f_gate * visual_cell + visual_i_gate * visual_c_gate
        visual_output = visual_o_gate * torch.tanh(visual_cell)

        audio_i_gate = F.sigmoid(audio_i_gate)
        audio_f_gate = F.sigmoid(audio_f_gate)
        audio_c_gate = F.tanh(audio_c_gate)
        audio_o_gate = F.sigmoid(audio_o_gate)

        audio_cell = audio_f_gate * audio_cell + audio_i_gate * audio_c_gate
        audio_output = audio_o_gate * torch.tanh(audio_cell)

        return visual_output, visual_cell, audio_output, audio_cell

class Dual_lstm(nn.Module):
    def __init__(self):

        super(Dual_lstm, self).__init__()

        self.video_input_dim = 512
        self.video_fc_dim = 512
        self.d_model = 256
        self.v_fc = nn.Linear(self.video_input_dim, self.video_fc_dim)
        self.LSTM_cell = Dual_lstm_cell(visual_input_dim=512, audio_input_dim=128, hidden_dim=256)
        #self.LSTM_cell_r = Dual_lstm_cell(visual_input_dim=512, audio_input_dim=128, hidden_dim=256)


        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)


    def forward(self, audio_feature, visual_feature):
        audio_rnn_input = audio_feature

        visual_rnn_input = visual_feature

        if torch.cuda.is_available():
            visual_hidden = Variable(torch.zeros(visual_rnn_input.size(0), self.d_model).cuda())
            visual_hidden_r = Variable(torch.zeros(visual_rnn_input.size(0), self.d_model).cuda())
        else:
            visual_hidden = Variable(torch.zeros(visual_rnn_input.size(0), self.d_model))
            visual_hidden_r = Variable(torch.zeros(visual_rnn_input.size(0), self.d_model))

        if torch.cuda.is_available():
            visual_cell = Variable(torch.zeros(visual_rnn_input.size(0), self.d_model).cuda())
            visual_cell_r = Variable(torch.zeros(visual_rnn_input.size(0), self.d_model).cuda())
        else:
            visual_cell = Variable(torch.zeros(visual_rnn_input.size(0), self.d_model))
            visual_cell_r = Variable(torch.zeros(visual_rnn_input.size(0), self.d_model))

        if torch.cuda.is_available():
            audio_hidden = Variable(torch.zeros(audio_rnn_input.size(0), self.d_model).cuda())
            audio_hidden_r = Variable(torch.zeros(audio_rnn_input.size(0), self.d_model).cuda())
        else:
            audio_hidden = Variable(torch.zeros(audio_rnn_input.size(0), self.d_model))
            audio_hidden_r = Variable(torch.zeros(audio_rnn_input.size(0), self.d_model))

        if torch.cuda.is_available():
            audio_cell = Variable(torch.zeros(audio_rnn_input.size(0), self.d_model).cuda())
            audio_cell_r = Variable(torch.zeros(audio_rnn_input.size(0), self.d_model).cuda())
        else:
            audio_cell = Variable(torch.zeros(audio_rnn_input.size(0), self.d_model))
            audio_cell_r = Variable(torch.zeros(audio_rnn_input.size(0), self.d_model))

        visual_output = []
        audio_output = []
        visual_output_r = []
        audio_output_r = []
        length = visual_rnn_input.size(1)

        visual_hidden = visual_hidden.double()
        visual_cell = visual_cell.double()
        audio_hidden = audio_hidden.double()
        audio_cell = audio_cell.double()
        visual_hidden_r = visual_hidden_r.double()
        visual_cell_r = visual_cell_r.double()
        audio_hidden_r = audio_hidden_r.double()
        audio_cell_r = audio_cell_r.double()


        for i in range(length):
            visual_hidden, visual_cell, audio_hidden, audio_cell = self.LSTM_cell(visual_rnn_input[:,i,:], visual_hidden, visual_cell,
                                                                                  audio_rnn_input[:,i,:], audio_hidden, audio_cell)
            visual_output.append(visual_hidden)
            audio_output.append(audio_hidden)

        visual_output = torch.stack(visual_output,dim=1)
        audio_output = torch.stack(audio_output, dim=1)


        # for i in range(length):
        #     visual_hidden_r, visual_cell_r, audio_hidden_r, audio_cell_r = self.LSTM_cell_r(visual_rnn_input[:,length-1-i,:], visual_hidden_r,
        #                                                                                     visual_cell_r, audio_rnn_input[:,length-1-i,:],
        #                                                                                     audio_hidden_r, audio_cell_r)
        #     visual_output_r.append(visual_hidden_r)
        #     audio_output_r.append(audio_hidden_r)

        # visual_output_r = torch.stack(visual_output_r, dim=1)
        # visual_output_r = torch.flip(visual_output_r, dims=[1])
        # audio_output_r = torch.stack(audio_output_r, dim=1)
        # audio_output_r = torch.flip(audio_output_r, dims=[1])
        # visual_output = torch.cat((visual_output, visual_output_r), dim=2)
        # audio_output = torch.cat((audio_output, audio_output_r), dim=2)
        return audio_output, visual_output


# model = Dual_lstm()
# visual_feature = torch.randn(32, 10,512)
# audio_feature = torch.randn(32, 10, 128)
# model(audio_feature, visual_feature)
#
