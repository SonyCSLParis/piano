from BFT.positional_embeddings.positional_embedding import BasePositionalEmbedding
import torch
from torch import nn
import math


# class SinusoidalPositionalEmbedding(BasePositionalEmbedding):
#     def __init__(self, positional_embedding_size, num_channels, **kwargs):
#         super(SinusoidalPositionalEmbedding, self).__init__()
#         self.dropout = torch.nn.Dropout(p=kwargs['dropout'])
#         self.positional_embedding_size = positional_embedding_size
#         max_len = kwargs['num_tokens_max']
#         self.pe = torch.zeros(max_len, positional_embedding_size)
#         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, positional_embedding_size, 2).float() * (
#                 -math.log(10000.0) / positional_embedding_size))
#         self.pe[:, 0::2] = torch.sin(position * div_term)
#         self.pe[:, 1::2] = torch.cos(position * div_term)
#         self.pe = nn.Parameter(self.pe.unsqueeze(0), requires_grad=False)

#     def forward(self, x, i=0, h=None):
#         assert i == 0
#         pos_embedding = self.pe[:, :x.size(1), :]
#         pos_embedding = torch.repeat_interleave(pos_embedding, x.shape[0], dim=0)
#         x = torch.cat([x, pos_embedding], dim=2)
#         return self.dropout(x), h
    

class SinusoidalPositionalEmbedding(BasePositionalEmbedding):
    def __init__(self, 
                 positional_embedding_size=None,
                 num_channels=None, 
                 dropout=0., num_tokens_max=None):
        super(SinusoidalPositionalEmbedding, self).__init__()
        # total size of this positional embedding
        self.positional_embedding_size = positional_embedding_size

        self.dropout = torch.nn.Dropout(p=dropout)
        self.num_channels = num_channels
        max_len = num_tokens_max // num_channels + 1

        self.pe = torch.zeros(max_len, positional_embedding_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, positional_embedding_size, 2).float() * (
                -math.log(10000.0) / positional_embedding_size))
        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = nn.Parameter(self.pe.unsqueeze(0), requires_grad=False)


    def forward(self, x, i=0, h=None, target=None):
        assert i == 0
        pos_embedding = self.pe.repeat_interleave(
            self.num_channels, dim=1
        )
        pos_embedding = pos_embedding[:, :x.size(1), :]
        pos_embedding = pos_embedding.repeat(x.shape[0], 1, 1)

        x = torch.cat([x, pos_embedding], dim=2)
        return self.dropout(x), h

    def forward_step(self, x, i=0, h=None, target=None):
        pe_index = i // self.num_channels
        pos_embedding = self.pe[:, pe_index].repeat(x.size(0), 1)

        x = torch.cat(
            [x, pos_embedding],
            dim=1
        )
        return self.dropout(x), h