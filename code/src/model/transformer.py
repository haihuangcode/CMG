import torch
import torch.nn as nn 
import torch.nn.functional as F


class TemporalEmbeddings(nn.Module):
    """Construct the embeddings from temporal tokens."""

    def __init__(self,
                 hidden_size,
                 max_temporal_buckets,
                 dropout_rate=None,
                 layer_norm_eps=None):
        super(TemporalEmbeddings, self).__init__()
        self.max_temporal_positions = max_temporal_buckets
        self.hidden_size = hidden_size
        self.dropout_rate = 0.1 if dropout_rate is None else dropout_rate
        self.layer_norm_eps = 1e-6 if layer_norm_eps is None else layer_norm_eps
        
        self.temporal_position_embedding = nn.Embedding(self.max_temporal_positions,
                                                        self.hidden_size)
        
        self.layernorm = nn.LayerNorm(hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.dropout_rate)

    def forward(self, 
                inputs, 
                dimensions):
        """Get token embeddings of inputs.

        Args:
            inputs: input embeddings
            dimensions: a list of dimensions
        Returns:
            position_embeddings: output embedding tensor, float32 with
              shape [batch_size, length, embedding_size]
        """

        _, t, _ = dimensions
        temporal_position_ids = torch.arange(t, device=inputs.device)        

        position_embeddings = self.temporal_position_embedding(temporal_position_ids)

        position_embeddings = self.layernorm(position_embeddings)
        embeddings = inputs + position_embeddings
        embeddings = self.dropout(embeddings)

        return embeddings


class SpatioTemporalEmbeddings(nn.Module):
    """Construct the embeddings from spatio-temporal tokens."""

    def __init__(self,
                 hidden_size,
                 max_temporal_buckets,
                 max_vertical_buckets,
                 max_horizontal_buckets,
                 dropout_rate=None,
                 layer_norm_eps=None):
        super(SpatioTemporalEmbeddings, self).__init__()
        self.max_temporal_positions = max_temporal_buckets
        self.max_vertical_positions = max_vertical_buckets
        self.max_horizontal_positions = max_horizontal_buckets
        self.hidden_size = hidden_size
        self.dropout_rate = 0.1 if dropout_rate is None else dropout_rate
        self.layer_norm_eps = 1e-6 if layer_norm_eps is None else layer_norm_eps
        
        self.temporal_position_embedding = nn.Embedding(self.max_temporal_positions,
                                                        self.hidden_size)
        self.vertical_position_embedding = nn.Embedding(self.max_vertical_positions,
                                                        self.hidden_size)
        self.horizontal_position_embedding = nn.Embedding(self.max_horizontal_positions,
                                                          self.hidden_size)
        
        self.layernorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.dropout_rate)

    def _build_vid_pos_ids(self, t, h, w):
        """Creates and returns 3d positional ids.

        Args:
          t: time length
          h: height
          w: width


        Returns:
          pos_ids: outputs with shape [t * h * w, 3],
            where 3 = 1 + 1 + 1; 1 for temporal id, 1 for vertical id, and 1 for
            horizontal id, with the following order: [t, h, w]
        """

        # define pos_ids - a fixed tensor which is a function of input shape
        temporal_ids = torch.arange(t).unsqueeze(1).unsqueeze(1)  # (t, 1, 1)
        vertical_ids = torch.arange(h).unsqueeze(0).unsqueeze(2)  # (1, h, 1)
        horizontal_ids = torch.arange(w).unsqueeze(0).unsqueeze(0)  # (1, 1, w)

        temporal_ids = temporal_ids.repeat((1, h, w))  # (t, h, w)
        vertical_ids = vertical_ids.repeat((t, 1, w))  # (t, h, w)
        horizontal_ids = horizontal_ids.repeat((t, h, 1))  # (t, h, w)

        # (t, h, w, 3)
        pos_ids = torch.stack([temporal_ids, vertical_ids, horizontal_ids], dim=3)
        pos_ids = pos_ids.reshape(-1, 3)  # (t*h*w, 3)

        return pos_ids

    def forward(self, 
                inputs, 
                dimensions):
        """Get token embeddings of inputs.

        Args:
            inputs: input embeddings
            dimensions: a list of dimensions
        Returns:
            position_embeddings: output embedding tensor, float32 with
              shape [batch_size, length, embedding_size]
        """

        _, t, h, w, _ = dimensions
        pos_ids = self._build_vid_pos_ids(t, h, w)        

        temporal_position_ids = pos_ids[None, :, 0].to(inputs.device)
        vertical_position_ids = pos_ids[None, :, 1].to(inputs.device)
        horizontal_position_ids = pos_ids[None, :, 2].to(inputs.device)

        temporal_position_embeddings = self.temporal_position_embedding(temporal_position_ids)
        vertical_position_embeddings = self.vertical_position_embedding(vertical_position_ids)
        horizontal_position_embeddings = self.horizontal_position_embedding(horizontal_position_ids)

        position_embeddings = (
            temporal_position_embeddings +
            vertical_position_embeddings +
            horizontal_position_embeddings
        )

        position_embeddings = self.layernorm(position_embeddings)
        embeddings = inputs + position_embeddings
        embeddings = self.dropout(embeddings)

        return embeddings


class DenseReLUDense(nn.Module):
    """Construct Dense+ReLU+Dense module used in FeedForward layer."""

    def __init__(self,
                 d_ff,
                 d_model,
                 use_bias,
                 dropout_rate):
        super(DenseReLUDense, self).__init__()
        self.wi = nn.Linear(d_model, d_ff, bias=use_bias)
        self.wo = nn.Linear(d_ff, d_model, bias=use_bias)
        self.dropout = nn.Dropout(dropout_rate)
        self.act = nn.ReLU()

    def forward(self, 
                hidden_states):

        h = self.wi(hidden_states)
        h = self.act(h)
        h = self.dropout(h)
        h = self.wo(h)
        return h


class DenseGeLUDense(DenseReLUDense):
    """Construct Dense+GeLU+Dense module used in FeedForward layer."""

    def __init__(self,
                 d_ff,
                 d_model,
                 use_bias,
                 dropout_rate):
        super(DenseGeLUDense, self).__init__(d_ff=d_ff, 
                                             d_model=d_model, 
                                             use_bias=use_bias, 
                                             dropout_rate=dropout_rate)
        self.act = nn.GELU()


class DenseSwishDense(DenseReLUDense):
    """Construct Dense+Swish+Dense module used in FeedForward layer."""

    def __init__(self,
                 d_ff,
                 d_model,
                 use_bias,
                 dropout_rate):
        super(DenseSwishDense, self).__init__(d_ff=d_ff, 
                                              d_model=d_model, 
                                              use_bias=use_bias, 
                                              dropout_rate=dropout_rate)

        self.act = nn.SiLU() 


class DenseGeGLUDense(nn.Module):
  """Construct Dense+GeGLU+Dense module used in FeedForward layer."""

  def __init__(self,
               d_ff,
               d_model,
               use_bias,
               dropout_rate):
    super(DenseGeGLUDense, self).__init__()
    self.wi_0 = nn.Linear(d_model, d_ff, bias=use_bias)
    self.wi_1 = nn.Linear(d_model, d_ff, bias=use_bias)
    self.wo = nn.Linear(d_ff, d_model, bias=use_bias)
    self.dropout = nn.Dropout(dropout_rate)
    self.act = nn.GELU()

  def forward(self,
              hidden_states):

    h_g = self.act(self.wi_0(hidden_states))
    h_l = self.wi_1(hidden_states)
    h = h_g * h_l
    h = self.dropout(h) 
    h = self.wo(h)
    return h


class FeedForward(nn.Module):
  """Construct FeedForward module used in Transformer layers."""

  def __init__(self,
               d_ff,
               d_model,
               pre_norm,
               use_bias,
               activation,
               dropout_rate,
               layer_norm_epsilon,
               **kwargs):

    super().__init__(**kwargs)
    self.pre_norm = pre_norm
    if activation == "relu":
      self.mlp = DenseReLUDense(d_ff,
                                d_model,
                                use_bias,
                                dropout_rate)
    elif activation == "gelu":
      self.mlp = DenseGeLUDense(d_ff,
                                d_model,
                                use_bias,
                                dropout_rate)
    elif activation == "swish":
      self.mlp = DenseSwishDense(d_ff,
                                 d_model,
                                 use_bias,
                                 dropout_rate)
    elif activation == "geglu":
      self.mlp = DenseGeGLUDense(d_ff,
                                 d_model,
                                 use_bias,
                                 dropout_rate)

    self.layer_norm = nn.LayerNorm(d_model, eps=layer_norm_epsilon)
    self.dropout = nn.Dropout(dropout_rate)

  def forward(self,
              hidden_states):

    res_inputs = hidden_states
    if self.pre_norm:
      hidden_states = self.layer_norm(hidden_states)
    y = self.mlp(hidden_states)
    layer_output = res_inputs + self.dropout(y)
    if not self.pre_norm:
      layer_output = self.layer_norm(layer_output)
    return layer_output


class MultiHeadAttention(nn.Module):
    """Construct the main MHA module used in Transformer layers."""

    def __init__(self, 
                 d_model, 
                 d_kv, 
                 num_heads, 
                 use_bias, 
                 dropout_rate):
        super().__init__()
        self.d_model = d_model
        self.d_kv = d_kv
        self.n_heads = num_heads
        self.inner_dim = self.n_heads * self.d_kv

        # query, key, and value mapping
        self.q = nn.Linear(self.d_model, self.inner_dim, bias=use_bias)
        self.k = nn.Linear(self.d_model, self.inner_dim, bias=use_bias)
        self.v = nn.Linear(self.d_model, self.inner_dim, bias=use_bias)
        self.o = nn.Linear(self.inner_dim, self.d_model, bias=use_bias)
        self.dropout = nn.Dropout(dropout_rate)

    def _split_heads(self, x, bs):
        """Split heads and rearrange elements."""
        # output shape: (bs, n_heads, seq_len, d_kv)
        return x.view(bs, -1, self.n_heads, self.d_kv).permute(0, 2, 1, 3)

    def _join_heads(self, x, bs):
        """Join heads and rearrange elements."""
        # output shape: (bs, seq_len, inner_dim)
        return x.permute(0, 2, 1, 3).reshape(bs, -1, self.inner_dim)

    def forward(self, 
                query, 
                key, 
                value, 
                mask=None):

        bs = query.size(0)

        q = self.q(query)  # (bs, qlen, inner_dim)
        k = self.k(key)  # (bs, klen, inner_dim)
        v = self.v(value)  # (bs, klen, inner_dim)

        q = self._split_heads(q, bs)  # (bs, n_heads, qlen, dim_per_head)
        k = self._split_heads(k, bs)  # (bs, n_heads, klen, dim_per_head)
        v = self._split_heads(v, bs)  # (bs, n_heads, vlen, dim_per_head)

        # (bs, n_heads, seq_len, seq_len)
        scores = torch.einsum("bnqd,bnkd->bnqk", q, k)

        # scale attention_scores
        dk = torch.tensor(k.size(-1), dtype=scores.dtype)
        scores = scores / torch.sqrt(dk)
        
        if mask is not None: 
            scores = scores + mask

        # (bs, n_heads, seq_len, seq_len)
        attention_weights = F.softmax(scores, dim=-1)
        # (bs, n_heads, seq_len, seq_len)
        attention_weights = self.dropout(attention_weights)
        # (bs, n_heads, seq_len, dim_per_head)
        hidden_states = torch.matmul(attention_weights, v)

        # (bs, seq_len, dim)
        hidden_states = self._join_heads(hidden_states, bs)
        # (bs, seq_len, out_dim)
        hidden_states = self.o(hidden_states)

        outputs = {
            "hidden_states": hidden_states,
            "attention_weights": attention_weights,
        }

        return outputs


class TransformerEncoderLayer(nn.Module):
  """Construct the main Transformer module which includes MHA + FeedForward."""

  def __init__(self,
               d_model,
               d_kv,
               d_ff,
               num_heads,
               pre_norm,
               use_bias,
               activation,
               dropout_rate,
               layer_norm_epsilon,
               **kwargs):
    super(TransformerEncoderLayer, self).__init__()

    self.pre_norm = pre_norm
    self.layer_norm = nn.LayerNorm(d_model, eps=layer_norm_epsilon)

    self.mha = MultiHeadAttention(d_model=d_model,
                                  d_kv=d_kv,
                                  use_bias=use_bias,
                                  num_heads=num_heads,
                                  dropout_rate=dropout_rate)

    self.dropout = nn.Dropout(dropout_rate)

    self.feed_forward = FeedForward(d_ff=d_ff,
                                    d_model=d_model,
                                    pre_norm=pre_norm,
                                    use_bias=use_bias,
                                    activation=activation,
                                    dropout_rate=dropout_rate,
                                    layer_norm_epsilon=layer_norm_epsilon)

  def forward(self,
              inputs,
              attention_mask=None):

    res_inputs = inputs

    # apply layer_norm on inputs if pre_norm
    if self.pre_norm:
      inputs = self.layer_norm(inputs)

    # apply multi-head attention module
    attention_outputs = self.mha(query=inputs,
                                 key=inputs,
                                 value=inputs,
                                 mask=attention_mask)

    hidden_states = attention_outputs["hidden_states"]

    # apply residual + dropout
    hidden_states = res_inputs + self.dropout(hidden_states)

    # apply layer_norm if not pre_norm
    if not self.pre_norm:
      hidden_states = self.layer_norm(hidden_states)

    # apply Feed Forward layer
    hidden_states = self.feed_forward(hidden_states)

    # update hidden states
    attention_outputs["hidden_states"] = hidden_states

    return attention_outputs


class TransformerEncoder(nn.Module):
    """Construct the final Transformer stack."""

    def __init__(self,
                 d_model,
                 d_kv,
                 d_ff,
                 num_layers,
                 num_heads,
                 pre_norm,
                 use_bias,
                 activation,
                 dropout_rate,
                 layer_norm_epsilon):
        super(TransformerEncoder, self).__init__()

        self.d_model = d_model
        self.pre_norm = pre_norm
        self.num_hidden_layers = num_layers

        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model=d_model,
                                    d_kv=d_kv,
                                    d_ff=d_ff,
                                    num_heads=num_heads,
                                    pre_norm=pre_norm,
                                    use_bias=use_bias,
                                    activation=activation,
                                    dropout_rate=dropout_rate,
                                    layer_norm_epsilon=layer_norm_epsilon) 
            for n in range(self.num_hidden_layers)])

        if self.pre_norm:
            self.final_layer_norm = nn.LayerNorm(d_model, eps=layer_norm_epsilon)
            self.dropout = nn.Dropout(dropout_rate)

    def forward(self, 
                inputs, 
                attention_mask=None):

        if attention_mask is None:
            attention_mask = torch.ones(inputs.size()[:-1], 
                                        dtype=torch.float32, 
                              			device=inputs.device)

        # Provided a padding mask of dimensions [batch_size, seq_length]
        # make the mask broadcastable to
        #         [batch_size, num_heads, seq_length, seq_length]
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)
        attention_mask = (1.0 - attention_mask) * -1e9

        all_hidden_states = ()
        all_attentions = ()

        hidden_states = inputs

        for layer in self.layers:
            # temporary --- to reduce memory consumption
            # all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer(inputs=hidden_states,
                                  attention_mask=attention_mask)

            # layer_outputs is a dictionary with the following keys:
            # hidden_states, self_attention_weights
            hidden_states = layer_outputs["hidden_states"]
            all_attentions = all_attentions + (layer_outputs["attention_weights"],)

        if self.pre_norm:
            hidden_states = self.final_layer_norm(hidden_states)
            hidden_states = self.dropout(hidden_states)

        # Add last layer
        all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = {
            "hidden_states": all_hidden_states,
            "attention_weights": all_attentions,
        }

        return outputs
