from fast_transformers.attention import AttentionLayer, CausalLinearAttention
from fast_transformers.builders import TransformerEncoderBuilder, RecurrentEncoderBuilder
from fast_transformers.masking import TriangularCausalMask
from fast_transformers.transformers import TransformerEncoder, TransformerEncoderLayer
from torch import nn

# this import is essential to put CausalLinearRelativeAttention in the fast-transformer's
# attention registry
from torch.nn import LayerNorm

from BFT.attention.causal_linear_relative_attention import CausalLinearRelativeAttention
import torch


class LinearTransformer(nn.Module):
    def __init__(self,
                 d_model,
                 n_heads,
                 n_layers,
                 dim_feedforward,
                 recurrent
                 ):
        super(LinearTransformer, self).__init__()

        query_dimension = d_model // n_heads
        dropout = 0.1
        attention_dropout = 0.1
        if not recurrent:
            self.transformer = TransformerEncoderBuilder.from_kwargs(
                n_layers=n_layers,
                n_heads=n_heads,
                query_dimensions=query_dimension,
                value_dimensions=query_dimension,
                feed_forward_dimensions=dim_feedforward,
                dropout=dropout,
                attention_type="causal-linear",
                activation="gelu",
                final_normalization=True
            ).get()
        else:
            self.transformer = RecurrentEncoderBuilder.from_kwargs(
                n_layers=n_layers,
                n_heads=n_heads,
                query_dimensions=query_dimension,
                value_dimensions=query_dimension,
                feed_forward_dimensions=dim_feedforward,
                dropout=dropout,
                attention_type="causal-linear",
                activation="gelu",
                final_normalization=True
            ).get()

    def forward(self, x):
        """
        Here, transformer is non recurrent
        :param x: (batch_size, num_tokens, feature_dim)
        :return:
        """
        triangular_mask = TriangularCausalMask(x.size(1), device=x.device)
        return self.transformer(x, attn_mask=triangular_mask)

    def forward_step(self, x, state):
        """
        Here, transformer is recurrent
        :param x: (batch_size, feature_dim)
        :param state:
        :return: (output, state)
        """
        return self.transformer(x, state=state)
