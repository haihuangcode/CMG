import torch
import torch.nn as nn
from model.transformer import TransformerEncoder

class UNIEncoder(nn.Module):
    def __init__(self):
        super(UNIEncoder, self).__init__()
        # define transformer head  
        self.tx = TransformerEncoder(d_model=256,
                                     d_kv=64,
                                     d_ff=4096,
                                     num_layers=24,
                                     num_heads=16,
                                     pre_norm=True,
                                     use_bias=True,
                                     activation="gelu",
                                     dropout_rate=0.1,
                                     layer_norm_epsilon=1e-6)
        
        # define post-tx projection head - it could be logits or embd space
        self.post_proj = nn.ModuleDict({# ReLU or GELU
            "video": nn.Sequential(nn.Linear(256, 256),nn.GELU()),#d_model=256 d_post_proj=256
            "audio": nn.Sequential(nn.Linear(256, 256),nn.GELU())
        })
        
    def _flatten_inputs(self,
                        inputs):
        input_shape = inputs.shape 
        bs = inputs.shape[0]
        d_embd = inputs.shape[-1]
        inputs = inputs.view(bs, -1, d_embd)

        return inputs, input_shape
    
    def _append_special_tokens(self, 
                              inputs, 
                              modality):
        batch_size = inputs.shape[0]
        agg_token = {
            "video": torch.nn.Parameter(torch.Tensor(256,)),#d_model
            "audio": torch.nn.Parameter(torch.Tensor(256,)),
        }
        special_embd = agg_token[modality][None, None, :].to(inputs.device)
        special_embd = special_embd.repeat(batch_size, 1, 1)

        return torch.cat([special_embd, inputs], dim=1)
    
    def _extend_attn_mask(self, 
                          attention_mask):
        attn_mask_shape = attention_mask.shape
        if len(attn_mask_shape) > 2:
            raise NotImplementedError

        batch_size = attn_mask_shape[0]
        extention_mask = torch.ones((batch_size, 1), dtype=attention_mask.dtype) 
        extended_attention_mask = torch.cat([extention_mask, attention_mask], dim=1)
        return extended_attention_mask

    def _modality_call(self, 
                       inputs, 
                       modality, 
                       training=False,
                       attention_mask=None, 
                       input_shape=None):
        embeddings = inputs
        if input_shape is None:
            embeddings, input_shape = self._flatten_inputs(embeddings)

        # print("pool:",embeddings)
        # print(features)

        # append modalities special tokens: [vid, aud, txt]
        tx_inputs = self._append_special_tokens(embeddings, modality)
        print("pool:",embeddings)

        # extend attention_mask accordingly
        if attention_mask is not None:
            attention_mask = self._extend_attn_mask(attention_mask)

        # call Transformer
        tx_outputs = self.tx(tx_inputs, attention_mask)

        # get last hidden states and perform final linear projection
        last_hidden_states = tx_outputs["hidden_states"][-1]
        modality_outputs = self.post_proj[modality](last_hidden_states)
        output_shape = list(input_shape[:-1]) + [modality_outputs.shape[-1]]
        # output_shape = list(256) + [modality_outputs.shape[-1]]

        features_pooled = modality_outputs[:, 0, :]
        features = modality_outputs[:, 1:, :].reshape(output_shape)
        
        # print("pool:",features_pooled)
        # print(features)

        # add token-level Transformer outputs
        outputs = {"features_pooled": features_pooled,
                   "features": features}

        return outputs
    
    def forward(self, video_semantic_result, audio_semantic_result):
        
        """
        outputs = {"features_pooled": features_pooled,
                   "features": features}
        """
        
        
        
        video_outputs = self._modality_call(inputs=video_semantic_result,
                                            modality='video',
                                            training=self.training,
                                            attention_mask=None)
        audio_outputs = self._modality_call(inputs=audio_semantic_result,
                                            modality='audio',
                                            training=self.training,
                                            attention_mask=None)
        
        # print("video_outputs:",video_outputs["features"].size(), video_outputs["features"].dtype)
        # print("video_semantic_result:",video_semantic_result.size(), video_semantic_result.dtype)
        
        # print(video_semantic_result)
        
        return video_outputs["features"], audio_outputs["features"]
