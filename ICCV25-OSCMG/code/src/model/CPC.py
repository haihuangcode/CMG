import torch
import torch.nn as nn
import torch.nn.functional as F
# class Attention(nn.Module):
#     def __init__(self, hidden_dim):
#         super(Attention, self).__init__()
#         self.linear = nn.Linear(hidden_dim, hidden_dim) 
#         self.softmax = nn.Softmax(dim=1)
    
#     def forward(self, context, input_seq):# 80*512  80*t_samples*256
#         # print('context.shape:',context.shape)
#         # print('input_seq.shape:',input_seq.shape)
#         scores = self.linear(context).unsqueeze(1) @ input_seq.transpose(1,2) 
#         attn = self.softmax(scores)
#         focused = (attn @ input_seq).squeeze(1)
#         # print('focused.shape:',focused.shape)
#         return focused

class Cross_CPC(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, context_dim, num_layers, n_prediction_steps=1, min_start_steps=1):
        super(Cross_CPC, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.context_dim = context_dim
        self.num_layers = num_layers
        self.n_prediction_steps = n_prediction_steps
        self.min_start_steps = min_start_steps
        self.softmax  = nn.Softmax()
        self.lsoftmax = nn.LogSoftmax()
        
        # Autoregressive LSTM network for video
        self.video_ar_lstm = nn.LSTM(embedding_dim, context_dim, num_layers, batch_first=True)
        
        # Autoregressive LSTM network for audio
        self.audio_ar_lstm = nn.LSTM(embedding_dim, context_dim, num_layers, batch_first=True)
        
        # Predictor network for video
        self.video_predictors = nn.ModuleList([
            nn.Linear(context_dim, embedding_dim) for _ in range(n_prediction_steps)
        ])
        
        # Predictor network for audio
        self.audio_predictors = nn.ModuleList([
            nn.Linear(context_dim, embedding_dim) for _ in range(n_prediction_steps)
        ])
    
    """
    video_forward_seq took the first t_samples+1 samples [0:t_samples].
    video_encode_samples took the samples [t_samples+1:t_samples+self.n_prediction_steps] as true future results.
    The LSTM utilized video_forward_seq to obtain video_context,
    then used video_predictors to predict video_pred.
    Calculate NCE between pred and encode_samples.
    """
    def forward(self, video_vq, audio_vq):
        batch_dim, time_length, _ = video_vq.shape# [batch_dim, time_length, embedding_dim] e.g.[80, 10, 256]
        # closedOpen
        # Choose any number from [3, 8) as a starting point, then predict the next two digits. Therefore, forward_seq has a minimum length of 4 (starting from 0).
        t_samples = (torch.randint(time_length - self.n_prediction_steps - self.min_start_steps, size=(1,)) + self.min_start_steps).long() # randomly pick time stamps
        # losses = list()
        nce = 0 # average over timestep and batch
        video_encode_samples = torch.empty((self.n_prediction_steps,batch_dim,self.embedding_dim), device = video_vq.device).double() # e.g. size 5*80*256
        audio_encode_samples = torch.empty((self.n_prediction_steps,batch_dim,self.embedding_dim), device = audio_vq.device).double() # e.g. size 5*80*256
        for i in range(1, self.n_prediction_steps+1):# closedOpen
            video_encode_samples[i-1] = video_vq[:,t_samples+i,:].reshape(batch_dim,self.embedding_dim) # z_tk e.g. size 80*256
            audio_encode_samples[i-1] = audio_vq[:,t_samples+i,:].reshape(batch_dim,self.embedding_dim) # z_tk e.g. size 80*256
        video_forward_seq = video_vq[:,:t_samples+1,:] # e.g. size 80*t_samples*256
        audio_forward_seq = audio_vq[:,:t_samples+1,:] # e.g. size 80*t_samples*256
        # Autoregressive LSTM for video
        video_hidden = (torch.zeros(self.num_layers, batch_dim, self.hidden_dim, device = video_vq.device).float(),
                  torch.zeros(self.num_layers, batch_dim, self.hidden_dim, device = video_vq.device).float())
        video_context, video_hidden = self.video_ar_lstm(video_forward_seq, video_hidden)
        
        # Autoregressive LSTM for audio
        audio_hidden = (torch.zeros(self.num_layers, batch_dim, self.hidden_dim, device = audio_vq.device).float(),
                  torch.zeros(self.num_layers, batch_dim, self.hidden_dim, device = audio_vq.device).float())
        audio_context, audio_hidden = self.audio_ar_lstm(audio_forward_seq, audio_hidden)
        
        video_context = video_context[:,t_samples,:].reshape(batch_dim,self.context_dim) # c_t e.g. size 80*512
        audio_context = audio_context[:,t_samples,:].reshape(batch_dim,self.context_dim) # c_t e.g. size 80*512
        
        video_pred = torch.empty((self.n_prediction_steps,batch_dim,self.embedding_dim), device = video_vq.device).double() # e.g. size 5*80*256
        audio_pred = torch.empty((self.n_prediction_steps,batch_dim,self.embedding_dim), device = audio_vq.device).double() # e.g. size 5*80*256
        for i in range(0, self.n_prediction_steps):
            video_linear = self.video_predictors[i]
            video_pred[i] = video_linear(video_context) #e.g. size 80*512 -> 80*256
            audio_linear = self.audio_predictors[i]
            audio_pred[i] = audio_linear(audio_context) #e.g. size 80*512 -> 80*256
        for i in range(0, self.n_prediction_steps):
            total1 = torch.mm(video_encode_samples[i], torch.transpose(audio_pred[i],0,1)) # e.g. size 80*80
            total2 = torch.mm(audio_encode_samples[i], torch.transpose(video_pred[i],0,1)) # e.g. size 80*80
            total3 = torch.mm(video_encode_samples[i], torch.transpose(video_pred[i],0,1)) # e.g. size 80*80
            total4 = torch.mm(audio_encode_samples[i], torch.transpose(audio_pred[i],0,1)) # e.g. size 80*80
            correct1 = torch.sum(torch.eq(torch.argmax(self.softmax(total1), dim=1), torch.arange(0, batch_dim, device = video_vq.device))) # correct is a tensor
            correct2 = torch.sum(torch.eq(torch.argmax(self.softmax(total2), dim=1), torch.arange(0, batch_dim, device = video_vq.device))) # correct is a tensor
            correct3 = torch.sum(torch.eq(torch.argmax(self.softmax(total3), dim=1), torch.arange(0, batch_dim, device = video_vq.device))) # correct is a tensor
            correct4 = torch.sum(torch.eq(torch.argmax(self.softmax(total4), dim=1), torch.arange(0, batch_dim, device = video_vq.device))) # correct is a tensor
            w1 = 1.0
            w2 = 1.0
            # Slightly computing self nce for each modality can provide a direction to align different modalities.
            w3 = 0.1
            w4 = 0.1
            nce += w1 * torch.sum(torch.diag(self.lsoftmax(total1))) # nce is a tensor
            nce += w2 * torch.sum(torch.diag(self.lsoftmax(total2))) # nce is a tensor
            nce += w3 * torch.sum(torch.diag(self.lsoftmax(total3))) # nce is a tensor
            nce += w4 * torch.sum(torch.diag(self.lsoftmax(total4))) # nce is a tensor
            
        nce /= -1.*batch_dim*self.n_prediction_steps
        accuracy1 = 1.*correct1/batch_dim
        accuracy2 = 1.*correct2/batch_dim
        accuracy3 = 1.*correct3/batch_dim
        accuracy4 = 1.*correct4/batch_dim
        return accuracy1, accuracy2, accuracy3, accuracy4, nce
    

class Cross_CPC_AVT(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, context_dim, num_layers, n_prediction_steps=1, min_start_steps=1):
        super(Cross_CPC_AVT, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.context_dim = context_dim
        self.num_layers = num_layers
        self.n_prediction_steps = n_prediction_steps
        self.min_start_steps = min_start_steps
        self.softmax  = nn.Softmax()
        self.lsoftmax = nn.LogSoftmax()
        
        # self.attention = Attention(hidden_dim) # 添加注意力模块
        
        # Autoregressive LSTM network for video
        self.video_ar_lstm = nn.LSTM(embedding_dim, context_dim, num_layers, batch_first=True)
        
        # Autoregressive LSTM network for audio
        self.audio_ar_lstm = nn.LSTM(embedding_dim, context_dim, num_layers, batch_first=True)
        
        # Autoregressive LSTM network for audio
        self.text_ar_lstm = nn.LSTM(embedding_dim, context_dim, num_layers, batch_first=True)
        
        # Predictor network for video
        self.video_predictors = nn.ModuleList([
            nn.Linear(context_dim, embedding_dim) for _ in range(n_prediction_steps)
        ])
        
        # Predictor network for audio
        self.audio_predictors = nn.ModuleList([
            nn.Linear(context_dim, embedding_dim) for _ in range(n_prediction_steps)
        ])
        
        # Predictor network for text
        self.text_predictors = nn.ModuleList([
            nn.Linear(context_dim, embedding_dim) for _ in range(n_prediction_steps)
        ])
    
    """
    video_forward_seq took the first t_samples+1 samples [0:t_samples].
    video_encode_samples took the samples [t_samples+1:t_samples+self.n_prediction_steps] as true future results.
    The LSTM utilized video_forward_seq to obtain video_context,
    then used video_predictors to predict video_pred.
    Calculate NCE between pred and encode_samples.
    """
    def forward(self, audio_vq, video_vq, text_vq):
        batch_dim, time_length, _ = video_vq.shape# [batch_dim, time_length, embedding_dim] e.g.[80, 10, 256]

        t_samples = (torch.randint(time_length - self.n_prediction_steps - self.min_start_steps, size=(1,)) + self.min_start_steps).long() # randomly pick time stamps
        # losses = list()
        nce = 0 # average over timestep and batch
        video_encode_samples = torch.empty((self.n_prediction_steps,batch_dim,self.embedding_dim), device = video_vq.device).double() # e.g. size 5*80*256
        audio_encode_samples = torch.empty((self.n_prediction_steps,batch_dim,self.embedding_dim), device = audio_vq.device).double() # e.g. size 5*80*256
        text_encode_samples = torch.empty((self.n_prediction_steps,batch_dim,self.embedding_dim), device = text_vq.device).double() # e.g. size 5*80*256
        for i in range(1, self.n_prediction_steps+1):
            video_encode_samples[i-1] = video_vq[:,t_samples+i,:].reshape(batch_dim,self.embedding_dim) # z_tk e.g. size 80*256
            audio_encode_samples[i-1] = audio_vq[:,t_samples+i,:].reshape(batch_dim,self.embedding_dim) # z_tk e.g. size 80*256
            text_encode_samples[i-1] = text_vq[:,t_samples+i,:].reshape(batch_dim,self.embedding_dim) # z_tk e.g. size 80*256
        video_forward_seq = video_vq[:,:t_samples+1,:] # e.g. size 80*t_samples*256
        audio_forward_seq = audio_vq[:,:t_samples+1,:] # e.g. size 80*t_samples*256
        text_forward_seq = text_vq[:,:t_samples+1,:] # e.g. size 80*t_samples*256
        # Autoregressive LSTM for video
        video_hidden = (torch.zeros(self.num_layers, batch_dim, self.hidden_dim, device = video_vq.device).double(),
                  torch.zeros(self.num_layers, batch_dim, self.hidden_dim, device = video_vq.device).double())
        video_context, video_hidden = self.video_ar_lstm(video_forward_seq, video_hidden)
        
        # Autoregressive LSTM for audio
        audio_hidden = (torch.zeros(self.num_layers, batch_dim, self.hidden_dim, device = audio_vq.device).double(),
                  torch.zeros(self.num_layers, batch_dim, self.hidden_dim, device = audio_vq.device).double())
        audio_context, audio_hidden = self.audio_ar_lstm(audio_forward_seq, audio_hidden)
        
        # Autoregressive LSTM for text
        text_hidden = (torch.zeros(self.num_layers, batch_dim, self.hidden_dim, device = text_vq.device).double(),
                  torch.zeros(self.num_layers, batch_dim, self.hidden_dim, device = text_vq.device).double())
        text_context, text_hidden = self.text_ar_lstm(text_forward_seq, text_hidden)
        
        video_context = video_context[:,t_samples,:].reshape(batch_dim,self.context_dim) # c_t e.g. size 80*512
        audio_context = audio_context[:,t_samples,:].reshape(batch_dim,self.context_dim) # c_t e.g. size 80*512
        text_context = text_context[:,t_samples,:].reshape(batch_dim,self.context_dim) # c_t e.g. size 80*512
        
        
        # 添加注意力模块
        # video_context = self.attention(video_context, video_forward_seq)  # 80*512  80*t_samples*256
        # audio_context = self.attention(audio_context, audio_forward_seq)
        # text_context = self.attention(text_context, text_forward_seq)
        
        video_pred = torch.empty((self.n_prediction_steps,batch_dim,self.embedding_dim), device = video_vq.device).double() # e.g. size 5*80*256
        audio_pred = torch.empty((self.n_prediction_steps,batch_dim,self.embedding_dim), device = audio_vq.device).double() # e.g. size 5*80*256
        text_pred = torch.empty((self.n_prediction_steps,batch_dim,self.embedding_dim), device = audio_vq.device).double() # e.g. size 5*80*256
        
        for i in range(0, self.n_prediction_steps):
            video_linear = self.video_predictors[i]
            video_pred[i] = video_linear(video_context) #e.g. size 80*512 -> 80*256
            audio_linear = self.audio_predictors[i]
            audio_pred[i] = audio_linear(audio_context) #e.g. size 80*512 -> 80*256
            text_linear = self.text_predictors[i]
            text_pred[i] = text_linear(text_context) #e.g. size 80*512 -> 80*256
        for i in range(0, self.n_prediction_steps):
            total1 = torch.mm(audio_encode_samples[i], torch.transpose(video_pred[i],0,1)) # e.g. size 80*80
            total2 = torch.mm(audio_encode_samples[i], torch.transpose(text_pred[i],0,1)) # e.g. size 80*80
            total3 = torch.mm(video_encode_samples[i], torch.transpose(text_pred[i],0,1)) # e.g. size 80*80
            total4 = torch.mm(video_encode_samples[i], torch.transpose(audio_pred[i],0,1)) # e.g. size 80*80
            total5 = torch.mm(text_encode_samples[i], torch.transpose(audio_pred[i],0,1)) # e.g. size 80*80
            total6 = torch.mm(text_encode_samples[i], torch.transpose(video_pred[i],0,1)) # e.g. size 80*80
            total7 = torch.mm(audio_encode_samples[i], torch.transpose(audio_pred[i],0,1)) # e.g. size 80*80
            total8 = torch.mm(video_encode_samples[i], torch.transpose(video_pred[i],0,1)) # e.g. size 80*80
            total9 = torch.mm(text_encode_samples[i], torch.transpose(text_pred[i],0,1)) # e.g. size 80*80
            correct1 = torch.sum(torch.eq(torch.argmax(self.softmax(total1), dim=1), torch.arange(0, batch_dim, device = video_vq.device))) # correct is a tensor
            correct2 = torch.sum(torch.eq(torch.argmax(self.softmax(total2), dim=1), torch.arange(0, batch_dim, device = video_vq.device))) # correct is a tensor
            correct3 = torch.sum(torch.eq(torch.argmax(self.softmax(total3), dim=1), torch.arange(0, batch_dim, device = video_vq.device))) # correct is a tensor
            correct4 = torch.sum(torch.eq(torch.argmax(self.softmax(total4), dim=1), torch.arange(0, batch_dim, device = video_vq.device))) # correct is a tensor
            correct5 = torch.sum(torch.eq(torch.argmax(self.softmax(total5), dim=1), torch.arange(0, batch_dim, device = video_vq.device))) # correct is a tensor
            correct6 = torch.sum(torch.eq(torch.argmax(self.softmax(total6), dim=1), torch.arange(0, batch_dim, device = video_vq.device))) # correct is a tensor
            correct7 = torch.sum(torch.eq(torch.argmax(self.softmax(total7), dim=1), torch.arange(0, batch_dim, device = video_vq.device))) # correct is a tensor
            correct8 = torch.sum(torch.eq(torch.argmax(self.softmax(total8), dim=1), torch.arange(0, batch_dim, device = video_vq.device))) # correct is a tensor
            correct9 = torch.sum(torch.eq(torch.argmax(self.softmax(total9), dim=1), torch.arange(0, batch_dim, device = video_vq.device))) # correct is a tensor
            w1 = 1.0
            w2 = 1.0
            w3 = 1.0
            w4 = 1.0
            w5 = 1.0
            w6 = 1.0
            # Slightly computing self nce for each modality can provide a direction to align different modalities.
            w7 = 0.1
            w8 = 0.1
            w9 = 0.1
            nce += w1 * torch.sum(torch.diag(self.lsoftmax(total1))) # nce is a tensor
            nce += w2 * torch.sum(torch.diag(self.lsoftmax(total2))) # nce is a tensor
            nce += w3 * torch.sum(torch.diag(self.lsoftmax(total3))) # nce is a tensor
            nce += w4 * torch.sum(torch.diag(self.lsoftmax(total4))) # nce is a tensor
            nce += w5 * torch.sum(torch.diag(self.lsoftmax(total5))) # nce is a tensor
            nce += w6 * torch.sum(torch.diag(self.lsoftmax(total6))) # nce is a tensor
            nce += w7 * torch.sum(torch.diag(self.lsoftmax(total7))) # nce is a tensor
            nce += w8 * torch.sum(torch.diag(self.lsoftmax(total8))) # nce is a tensor
            nce += w9 * torch.sum(torch.diag(self.lsoftmax(total9))) # nce is a tensor
            
        nce /= -1.*batch_dim*self.n_prediction_steps
        accuracy1 = 1.*correct1/batch_dim
        accuracy2 = 1.*correct2/batch_dim
        accuracy3 = 1.*correct3/batch_dim
        accuracy4 = 1.*correct4/batch_dim
        accuracy5 = 1.*correct5/batch_dim
        accuracy6 = 1.*correct6/batch_dim
        accuracy7 = 1.*correct7/batch_dim
        accuracy8 = 1.*correct8/batch_dim
        accuracy9 = 1.*correct9/batch_dim
        return accuracy1, accuracy2, accuracy3, accuracy4, accuracy5, accuracy6, accuracy7, accuracy8, accuracy9, nce
    

"""对输入进行mask"""
class Cross_Mask_CPC_AVT(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, context_dim, num_layers, n_prediction_steps=1, min_start_steps=1, mask_prob=0.1):
        super(Cross_Mask_CPC_AVT, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.context_dim = context_dim
        self.num_layers = num_layers
        self.n_prediction_steps = n_prediction_steps
        self.min_start_steps = min_start_steps
        self.softmax  = nn.Softmax()
        self.lsoftmax = nn.LogSoftmax()
        self.mask_prob = mask_prob
        
        # self.attention = Attention(hidden_dim) # 添加注意力模块
        
        # Autoregressive LSTM network for video
        self.video_ar_lstm = nn.LSTM(embedding_dim, context_dim, num_layers, batch_first=True)
        
        # Autoregressive LSTM network for audio
        self.audio_ar_lstm = nn.LSTM(embedding_dim, context_dim, num_layers, batch_first=True)
        
        # Autoregressive LSTM network for audio
        self.text_ar_lstm = nn.LSTM(embedding_dim, context_dim, num_layers, batch_first=True)
        
        # Predictor network for video
        self.video_predictors = nn.ModuleList([
            nn.Linear(context_dim, embedding_dim) for _ in range(n_prediction_steps)
        ])
        
        # Predictor network for audio
        self.audio_predictors = nn.ModuleList([
            nn.Linear(context_dim, embedding_dim) for _ in range(n_prediction_steps)
        ])
        
        # Predictor network for text
        self.text_predictors = nn.ModuleList([
            nn.Linear(context_dim, embedding_dim) for _ in range(n_prediction_steps)
        ])
        
    def _apply_mask(self, inputs1, inputs2, inputs3):
        mask = (torch.rand(inputs1.shape[:-1], device=inputs1.device) > self.mask_prob).float()
        mask = mask.unsqueeze(-1).expand_as(inputs1)
        return inputs1 * mask, inputs2 * mask, inputs3 * mask, mask, mask, mask
    
    
    
    """
    video_forward_seq took the first t_samples+1 samples [0:t_samples].
    video_encode_samples took the samples [t_samples+1:t_samples+self.n_prediction_steps] as true future results.
    The LSTM utilized video_forward_seq to obtain video_context,
    then used video_predictors to predict video_pred.
    Calculate NCE between pred and encode_samples.
    """
    def forward(self, audio_vq, video_vq, text_vq):
        batch_dim, time_length, _ = video_vq.shape# [batch_dim, time_length, embedding_dim] e.g.[80, 10, 256]
        
        # video_vq_mask, audio_vq_mask, text_vq_mask, video_mask, audio_mask, text_mask = self._apply_mask(video_vq, audio_vq, text_vq)
        video_vq, audio_vq, text_vq, video_mask, audio_mask, text_mask = self._apply_mask(video_vq, audio_vq, text_vq)

        t_samples = (torch.randint(time_length - self.n_prediction_steps - self.min_start_steps, size=(1,)) + self.min_start_steps).long() # randomly pick time stamps
        # losses = list()
        nce = 0 # average over timestep and batch
        video_encode_samples = torch.empty((self.n_prediction_steps,batch_dim,self.embedding_dim), device = video_vq.device).double() # e.g. size 5*80*256
        audio_encode_samples = torch.empty((self.n_prediction_steps,batch_dim,self.embedding_dim), device = audio_vq.device).double() # e.g. size 5*80*256
        text_encode_samples = torch.empty((self.n_prediction_steps,batch_dim,self.embedding_dim), device = text_vq.device).double() # e.g. size 5*80*256
        for i in range(1, self.n_prediction_steps+1):
            video_encode_samples[i-1] = video_vq[:,t_samples+i,:].reshape(batch_dim,self.embedding_dim) # z_tk e.g. size 80*256
            audio_encode_samples[i-1] = audio_vq[:,t_samples+i,:].reshape(batch_dim,self.embedding_dim) # z_tk e.g. size 80*256
            text_encode_samples[i-1] = text_vq[:,t_samples+i,:].reshape(batch_dim,self.embedding_dim) # z_tk e.g. size 80*256
        video_forward_seq = video_vq[:,:t_samples+1,:] # e.g. size 80*t_samples*256
        audio_forward_seq = audio_vq[:,:t_samples+1,:] # e.g. size 80*t_samples*256
        text_forward_seq = text_vq[:,:t_samples+1,:] # e.g. size 80*t_samples*256
        
        
        # video_forward_seq = video_vq_mask[:,:t_samples+1,:] # e.g. size 80*t_samples*256
        # audio_forward_seq = audio_vq_mask[:,:t_samples+1,:] # e.g. size 80*t_samples*256
        # text_forward_seq = text_vq_mask[:,:t_samples+1,:] # e.g. size 80*t_samples*256
        
        # Autoregressive LSTM for video
        video_hidden = (torch.zeros(self.num_layers, batch_dim, self.hidden_dim, device = video_vq.device).double(),
                  torch.zeros(self.num_layers, batch_dim, self.hidden_dim, device = video_vq.device).double())
        video_context, video_hidden = self.video_ar_lstm(video_forward_seq, video_hidden)
        
        # Autoregressive LSTM for audio
        audio_hidden = (torch.zeros(self.num_layers, batch_dim, self.hidden_dim, device = audio_vq.device).double(),
                  torch.zeros(self.num_layers, batch_dim, self.hidden_dim, device = audio_vq.device).double())
        audio_context, audio_hidden = self.audio_ar_lstm(audio_forward_seq, audio_hidden)
        
        # Autoregressive LSTM for text
        text_hidden = (torch.zeros(self.num_layers, batch_dim, self.hidden_dim, device = text_vq.device).double(),
                  torch.zeros(self.num_layers, batch_dim, self.hidden_dim, device = text_vq.device).double())
        text_context, text_hidden = self.text_ar_lstm(text_forward_seq, text_hidden)
        
        video_context = video_context[:,t_samples,:].reshape(batch_dim,self.context_dim) # c_t e.g. size 80*512
        audio_context = audio_context[:,t_samples,:].reshape(batch_dim,self.context_dim) # c_t e.g. size 80*512
        text_context = text_context[:,t_samples,:].reshape(batch_dim,self.context_dim) # c_t e.g. size 80*512
        
        
        video_pred = torch.empty((self.n_prediction_steps,batch_dim,self.embedding_dim), device = video_vq.device).double() # e.g. size 5*80*256
        audio_pred = torch.empty((self.n_prediction_steps,batch_dim,self.embedding_dim), device = audio_vq.device).double() # e.g. size 5*80*256
        text_pred = torch.empty((self.n_prediction_steps,batch_dim,self.embedding_dim), device = audio_vq.device).double() # e.g. size 5*80*256
        
        for i in range(0, self.n_prediction_steps):
            video_linear = self.video_predictors[i]
            video_pred[i] = video_linear(video_context) #e.g. size 80*512 -> 80*256
            audio_linear = self.audio_predictors[i]
            audio_pred[i] = audio_linear(audio_context) #e.g. size 80*512 -> 80*256
            text_linear = self.text_predictors[i]
            text_pred[i] = text_linear(text_context) #e.g. size 80*512 -> 80*256
        for i in range(0, self.n_prediction_steps):
            total1 = torch.mm(audio_encode_samples[i], torch.transpose(video_pred[i],0,1)) # e.g. size 80*80
            total2 = torch.mm(audio_encode_samples[i], torch.transpose(text_pred[i],0,1)) # e.g. size 80*80
            total3 = torch.mm(video_encode_samples[i], torch.transpose(text_pred[i],0,1)) # e.g. size 80*80
            total4 = torch.mm(video_encode_samples[i], torch.transpose(audio_pred[i],0,1)) # e.g. size 80*80
            total5 = torch.mm(text_encode_samples[i], torch.transpose(audio_pred[i],0,1)) # e.g. size 80*80
            total6 = torch.mm(text_encode_samples[i], torch.transpose(video_pred[i],0,1)) # e.g. size 80*80
            total7 = torch.mm(audio_encode_samples[i], torch.transpose(audio_pred[i],0,1)) # e.g. size 80*80
            total8 = torch.mm(video_encode_samples[i], torch.transpose(video_pred[i],0,1)) # e.g. size 80*80
            total9 = torch.mm(text_encode_samples[i], torch.transpose(text_pred[i],0,1)) # e.g. size 80*80
            correct1 = torch.sum(torch.eq(torch.argmax(self.softmax(total1), dim=1), torch.arange(0, batch_dim, device = video_vq.device))) # correct is a tensor
            correct2 = torch.sum(torch.eq(torch.argmax(self.softmax(total2), dim=1), torch.arange(0, batch_dim, device = video_vq.device))) # correct is a tensor
            correct3 = torch.sum(torch.eq(torch.argmax(self.softmax(total3), dim=1), torch.arange(0, batch_dim, device = video_vq.device))) # correct is a tensor
            correct4 = torch.sum(torch.eq(torch.argmax(self.softmax(total4), dim=1), torch.arange(0, batch_dim, device = video_vq.device))) # correct is a tensor
            correct5 = torch.sum(torch.eq(torch.argmax(self.softmax(total5), dim=1), torch.arange(0, batch_dim, device = video_vq.device))) # correct is a tensor
            correct6 = torch.sum(torch.eq(torch.argmax(self.softmax(total6), dim=1), torch.arange(0, batch_dim, device = video_vq.device))) # correct is a tensor
            correct7 = torch.sum(torch.eq(torch.argmax(self.softmax(total7), dim=1), torch.arange(0, batch_dim, device = video_vq.device))) # correct is a tensor
            correct8 = torch.sum(torch.eq(torch.argmax(self.softmax(total8), dim=1), torch.arange(0, batch_dim, device = video_vq.device))) # correct is a tensor
            correct9 = torch.sum(torch.eq(torch.argmax(self.softmax(total9), dim=1), torch.arange(0, batch_dim, device = video_vq.device))) # correct is a tensor
            w1 = 1.0
            w2 = 1.0
            w3 = 1.0
            w4 = 1.0
            w5 = 1.0
            w6 = 1.0
            # Slightly computing self nce for each modality can provide a direction to align different modalities.
            w7 = 0.1
            w8 = 0.1
            w9 = 0.1
            nce += w1 * torch.sum(torch.diag(self.lsoftmax(total1))) # nce is a tensor
            nce += w2 * torch.sum(torch.diag(self.lsoftmax(total2))) # nce is a tensor
            nce += w3 * torch.sum(torch.diag(self.lsoftmax(total3))) # nce is a tensor
            nce += w4 * torch.sum(torch.diag(self.lsoftmax(total4))) # nce is a tensor
            nce += w5 * torch.sum(torch.diag(self.lsoftmax(total5))) # nce is a tensor
            nce += w6 * torch.sum(torch.diag(self.lsoftmax(total6))) # nce is a tensor
            nce += w7 * torch.sum(torch.diag(self.lsoftmax(total7))) # nce is a tensor
            nce += w8 * torch.sum(torch.diag(self.lsoftmax(total8))) # nce is a tensor
            nce += w9 * torch.sum(torch.diag(self.lsoftmax(total9))) # nce is a tensor
            
        nce /= -1.*batch_dim*self.n_prediction_steps
        accuracy1 = 1.*correct1/batch_dim
        accuracy2 = 1.*correct2/batch_dim
        accuracy3 = 1.*correct3/batch_dim
        accuracy4 = 1.*correct4/batch_dim
        accuracy5 = 1.*correct5/batch_dim
        accuracy6 = 1.*correct6/batch_dim
        accuracy7 = 1.*correct7/batch_dim
        accuracy8 = 1.*correct8/batch_dim
        accuracy9 = 1.*correct9/batch_dim
        return accuracy1, accuracy2, accuracy3, accuracy4, accuracy5, accuracy6, accuracy7, accuracy8, accuracy9, nce
    
    

"""
mask2
动态掩码机制：根据输入数据的特征，动态生成掩码。可以使用注意力机制或不确定性估计来生成掩码矩阵，使模型更关注难以预测的区域。
掩码感知的对比损失：在计算对比学习损失时，引入掩码权重，使得被掩盖的部分对损失的贡献更大，强化模型对这些部分的学习。
信息瓶颈原理：通过掩码机制，模型被迫在有限的信息下进行预测，从而学习更具泛化性的特征。
"""
class Cross_Mask2_CPC_AVT(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, context_dim, num_layers, n_prediction_steps=1, min_start_steps=1, mask_prob=0.1):
        super(Cross_Mask2_CPC_AVT, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.context_dim = context_dim
        self.num_layers = num_layers
        self.n_prediction_steps = n_prediction_steps
        self.min_start_steps = min_start_steps
        self.mask_prob = mask_prob
        
        self.softmax  = nn.Softmax(dim=-1)
        self.lsoftmax = nn.LogSoftmax(dim=-1)
        
        # 定义注意力机制，用于动态生成掩码
        self.video_attention = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=8, batch_first=True)
        self.audio_attention = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=8, batch_first=True)
        self.text_attention  = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=8, batch_first=True)
        
        # 自回归 LSTM 网络
        self.video_ar_lstm = nn.LSTM(embedding_dim, context_dim, num_layers, batch_first=True)
        self.audio_ar_lstm = nn.LSTM(embedding_dim, context_dim, num_layers, batch_first=True)
        self.text_ar_lstm  = nn.LSTM(embedding_dim, context_dim, num_layers, batch_first=True)
        
        # 预测器网络
        self.video_predictors = nn.ModuleList([nn.Linear(context_dim, embedding_dim) for _ in range(n_prediction_steps)])
        self.audio_predictors = nn.ModuleList([nn.Linear(context_dim, embedding_dim) for _ in range(n_prediction_steps)])
        self.text_predictors  = nn.ModuleList([nn.Linear(context_dim, embedding_dim) for _ in range(n_prediction_steps)])
        
    def _apply_mask(self, inputs1, inputs2, inputs3):
        mask = (torch.rand(inputs1.shape[:-1], device=inputs1.device) > self.mask_prob).float()
        mask = mask.unsqueeze(-1).expand_as(inputs1)
        return inputs1 * mask, inputs2 * mask, inputs3 * mask, mask, mask, mask
    
    def forward(self, audio_vq, video_vq, text_vq):
        batch_dim, time_length, _ = video_vq.shape
        
        # 动态生成掩码(学得不稳定)
        # video_vq_masked, video_mask = self.apply_dynamic_mask(video_vq, self.video_attention)
        # audio_vq_masked, audio_mask = self.apply_dynamic_mask(audio_vq, self.audio_attention)
        # text_vq_masked,  text_mask  = self.apply_dynamic_mask(text_vq,  self.text_attention)
        video_vq_masked, audio_vq_masked, text_vq_masked, video_mask, audio_mask, text_mask = self._apply_mask(video_vq, audio_vq, text_vq)
        
        # audio_vq_masked, audio_mask = self._apply_mask(audio_vq)
        # text_vq_masked,  text_mask  = self._apply_mask(text_vq)

        
        
        t_samples = (torch.randint(time_length - self.n_prediction_steps - self.min_start_steps, size=(1,)).item() + self.min_start_steps)
        nce = 0.0
        
        # 准备编码的样本
        video_encode_samples = []
        audio_encode_samples = []
        text_encode_samples  = []
        for i in range(1, self.n_prediction_steps + 1):
            # video_encode_samples.append(video_vq_masked[:, t_samples + i, :].reshape(batch_dim, self.embedding_dim))
            # audio_encode_samples.append(audio_vq_masked[:, t_samples + i, :].reshape(batch_dim, self.embedding_dim))
            # text_encode_samples.append( text_vq_masked[:, t_samples + i, :].reshape(batch_dim, self.embedding_dim))
            video_encode_samples.append(video_vq[:, t_samples + i, :].reshape(batch_dim, self.embedding_dim))
            audio_encode_samples.append(audio_vq[:, t_samples + i, :].reshape(batch_dim, self.embedding_dim))
            text_encode_samples.append( text_vq[:, t_samples + i, :].reshape(batch_dim, self.embedding_dim))
        
        
        # 准备前向序列
        video_forward_seq = video_vq_masked[:, :t_samples + 1, :]
        audio_forward_seq = audio_vq_masked[:, :t_samples + 1, :]
        text_forward_seq  =  text_vq_masked[:, :t_samples + 1, :]
        
        # 通过自回归 LSTM 获取上下文
        video_context, _ = self.video_ar_lstm(video_forward_seq)
        audio_context, _ = self.audio_ar_lstm(audio_forward_seq)
        text_context,  _ = self.text_ar_lstm( text_forward_seq)
        
        # 获取时间步上的上下文向量
        video_context = video_context[:, -1, :]  # [batch_dim, context_dim]
        audio_context = audio_context[:, -1, :]
        text_context  = text_context[:, -1, :]
        
        # 通过预测器生成预测
        video_pred = []
        audio_pred = []
        text_pred  = []
        for i in range(self.n_prediction_steps):
            video_pred.append(self.video_predictors[i](video_context))
            audio_pred.append(self.audio_predictors[i](audio_context))
            text_pred.append( self.text_predictors[i]( text_context))
            
        
        # 计算掩码感知的对比损失
        for i in range(self.n_prediction_steps):
            # 计算相似度矩阵
            sim_matrix_av = torch.mm(audio_pred[i], video_encode_samples[i].t())
            sim_matrix_at = torch.mm(audio_pred[i], text_encode_samples[i].t())
            sim_matrix_va = torch.mm(video_pred[i], audio_encode_samples[i].t())
            sim_matrix_vt = torch.mm(video_pred[i], text_encode_samples[i].t())
            sim_matrix_ta = torch.mm(text_pred[i], audio_encode_samples[i].t())
            sim_matrix_tv = torch.mm(text_pred[i], video_encode_samples[i].t())
            sim_matrix_aa = torch.mm(audio_pred[i], audio_encode_samples[i].t())
            sim_matrix_vv = torch.mm(video_pred[i], video_encode_samples[i].t())
            sim_matrix_tt = torch.mm(text_pred[i], text_encode_samples[i].t())
            
            correct1 = torch.sum(torch.eq(torch.argmax(self.softmax(sim_matrix_av), dim=1), torch.arange(0, batch_dim, device = video_vq.device))) # correct is a tensor
            correct2 = torch.sum(torch.eq(torch.argmax(self.softmax(sim_matrix_at), dim=1), torch.arange(0, batch_dim, device = video_vq.device))) # correct is a tensor
            correct3 = torch.sum(torch.eq(torch.argmax(self.softmax(sim_matrix_va), dim=1), torch.arange(0, batch_dim, device = video_vq.device))) # correct is a tensor
            correct4 = torch.sum(torch.eq(torch.argmax(self.softmax(sim_matrix_vt), dim=1), torch.arange(0, batch_dim, device = video_vq.device))) # correct is a tensor
            correct5 = torch.sum(torch.eq(torch.argmax(self.softmax(sim_matrix_ta), dim=1), torch.arange(0, batch_dim, device = video_vq.device))) # correct is a tensor
            correct6 = torch.sum(torch.eq(torch.argmax(self.softmax(sim_matrix_tv), dim=1), torch.arange(0, batch_dim, device = video_vq.device))) # correct is a tensor
            correct7 = torch.sum(torch.eq(torch.argmax(self.softmax(sim_matrix_aa), dim=1), torch.arange(0, batch_dim, device = video_vq.device))) # correct is a tensor
            correct8 = torch.sum(torch.eq(torch.argmax(self.softmax(sim_matrix_vv), dim=1), torch.arange(0, batch_dim, device = video_vq.device))) # correct is a tensor
            correct9 = torch.sum(torch.eq(torch.argmax(self.softmax(sim_matrix_tt), dim=1), torch.arange(0, batch_dim, device = video_vq.device))) # correct is a tensor
            
            # 获取掩码权重（对比学习中，被掩盖的样本赋予更大权重）
            mask_weight_a = audio_mask[:, t_samples + i + 1].mean(dim=1)
            mask_weight_v = video_mask[:, t_samples + i + 1].mean(dim=1)
            mask_weight_t = text_mask[:, t_samples + i + 1].mean(dim=1)
            
            # 计算掩码感知的对比损失
            loss_av = self.masked_contrastive_loss(sim_matrix_av, mask_weight_a)
            loss_at = self.masked_contrastive_loss(sim_matrix_at, mask_weight_a)
            loss_va = self.masked_contrastive_loss(sim_matrix_va, mask_weight_v)
            loss_vt = self.masked_contrastive_loss(sim_matrix_vt, mask_weight_v)
            loss_ta = self.masked_contrastive_loss(sim_matrix_ta, mask_weight_t)
            loss_tv = self.masked_contrastive_loss(sim_matrix_tv, mask_weight_t)
            loss_aa = self.masked_contrastive_loss(sim_matrix_aa, mask_weight_a)
            loss_vv = self.masked_contrastive_loss(sim_matrix_vv, mask_weight_v)
            loss_tt = self.masked_contrastive_loss(sim_matrix_tt, mask_weight_t)
            
            
            nce += loss_av + loss_at + loss_va + loss_vt + loss_ta + loss_tv + 0.1*(loss_aa + loss_vv + loss_tt)
        
        nce /= self.n_prediction_steps
        accuracy1 = 1.*correct1/batch_dim
        accuracy2 = 1.*correct2/batch_dim
        accuracy3 = 1.*correct3/batch_dim
        accuracy4 = 1.*correct4/batch_dim
        accuracy5 = 1.*correct5/batch_dim
        accuracy6 = 1.*correct6/batch_dim
        accuracy7 = 1.*correct7/batch_dim
        accuracy8 = 1.*correct8/batch_dim
        accuracy9 = 1.*correct9/batch_dim
 
        return accuracy1, accuracy2, accuracy3, accuracy4, accuracy5, accuracy6, accuracy7, accuracy8, accuracy9, nce
    
    def apply_dynamic_mask(self, inputs, attention_layer):
        # inputs: [batch_size, seq_len, embedding_dim]
        attn_output, attn_weights = attention_layer(inputs, inputs, inputs)
        # 根据注意力权重生成掩码
        # 这里采用权重低于平均值的位置进行掩盖
        attn_scores = attn_weights.mean(dim=1)  # [batch_size, seq_len]
        threshold = attn_scores.mean(dim=1, keepdim=True)  # [batch_size, 1]
        mask = (attn_scores > threshold).float()  # [batch_size, seq_len]
        mask = mask.unsqueeze(-1)  # [batch_size, seq_len, 1]
        inputs_masked = inputs * mask  # [batch_size, seq_len, embedding_dim]
        return inputs_masked, mask  # 返回掩码后的输入和掩码矩阵
    
    def masked_contrastive_loss(self, sim_matrix, mask_weight):
        # sim_matrix: [batch_size, batch_size]
        logits = self.lsoftmax(sim_matrix)
        labels = torch.arange(sim_matrix.size(0)).to(sim_matrix.device)
        loss = F.nll_loss(logits, labels, reduction='none')#使用了 F.nll_loss 函数，它是 PyTorch 中的标准负对数似然损失函数，已经在内部对损失取了负号，并且默认情况下（reduction='mean'）会对批次中的样本求平均。
        # 对损失进行加权
        loss = loss * mask_weight
        return loss.mean()#在 F.nll_loss 中，使用 reduction='none'，然后对损失乘以掩码权重后，再通过 loss.mean() 对损失求平均。这已经对批次中的样本进行了平均。
