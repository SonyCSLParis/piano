import torch
from torch import nn
import math


class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, positional_embedding_size, num_channels, **kwargs):
        super(SinusoidalPositionalEmbedding, self).__init__()
        self.dropout = torch.nn.Dropout(p=kwargs['dropout'])
        self.positional_embedding_size = positional_embedding_size
        max_len = kwargs['num_tokens_max']
        self.pe = torch.zeros(max_len, positional_embedding_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, positional_embedding_size, 2).float() * (
                -math.log(10000.0) / positional_embedding_size))
        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = nn.Parameter(self.pe.unsqueeze(0), requires_grad=False)

    def forward(self, x, i=0, h=None):
        assert i == 0
        pos_embedding = self.pe[:, :x.size(1), :]
        pos_embedding = torch.repeat_interleave(pos_embedding, x.shape[0], dim=0)
        x = torch.cat([x, pos_embedding], dim=2)
        return self.dropout(x), h

    def forward_step(self, x, i):
        # TODO
        raise NotImplementedError
        pos_embedding = self.pe[0, i:i + 1]
        x = torch.cat(
            [x, pos_embedding.expand_as(x)],
            dim=1
        )
        return self.dropout(x)


# class SinusoidalPositionalEmbeddingChannels(nn.Module):
#     def __init__(self, positional_embedding_size, num_channels, **kwargs):
#         super(SinusoidalPositionalEmbeddingChannels, self).__init__()
#         self.dropout = torch.nn.Dropout(p=kwargs['dropout'])
#         self.positional_embedding_size = positional_embedding_size
#         self.num_channels = num_channels
#         max_len = kwargs['num_tokens_max'] // num_channels + 1
#
#         self.pe = torch.zeros(max_len, positional_embedding_size)
#         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, positional_embedding_size, 2).float() * (
#                 -math.log(10000.0) / positional_embedding_size))
#         self.pe[:, 0::2] = torch.sin(position * div_term)
#         self.pe[:, 1::2] = torch.cos(position * div_term)
#         self.pe = nn.Parameter(self.pe.unsqueeze(0), requires_grad=False)
#
#         self.channel_embeddings = nn.Parameter(
#             torch.randn(
#                 1, num_channels, positional_embedding_size
#             )
#         )
#
#     def forward(self, x, i=0, h=None):
#         assert i == 0
#         pos_embedding = self.pe.repeat_interleave(
#             self.num_channels, dim=1
#         )
#         pos_embedding = pos_embedding[:, :x.size(1), :]
#         pos_embedding = torch.repeat_interleave(pos_embedding, x.shape[0], dim=0)
#
#         channel_embeddings = self.channel_embeddings.repeat(
#             x.size(0),
#             x.size(1) // self.num_channels + 1,
#             1
#         )
#         channel_embeddings = channel_embeddings[:, :x.size(1)]
#         # Sum?
#         embeddings = pos_embedding + channel_embeddings
#         x = torch.cat([x, embeddings], dim=2)
#         return self.dropout(x), h
#
#     def forward_step(self, x, i):
#         # TODO
#         raise NotImplementedError
#         pos_embedding = self.pe[0, i:i + 1]
#         x = torch.cat(
#             [x, pos_embedding.expand_as(x)],
#             dim=1
#         )
#         return self.dropout(x)


class SinusoidalPositionalEmbeddingChannels(nn.Module):
    def __init__(self, positional_embedding_size, num_channels, **kwargs):
        super(SinusoidalPositionalEmbeddingChannels, self).__init__()
        assert positional_embedding_size % 2 == 0

        self.dropout = torch.nn.Dropout(p=kwargs['dropout'])
        individual_positional_embedding_size = positional_embedding_size // 2
        self.num_channels = num_channels
        max_len = kwargs['num_tokens_max'] // num_channels + 1

        self.pe = torch.zeros(max_len, individual_positional_embedding_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, individual_positional_embedding_size, 2).float() * (
                -math.log(10000.0) / positional_embedding_size))
        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = nn.Parameter(self.pe.unsqueeze(0), requires_grad=False)

        self.channel_embeddings = nn.Parameter(
            torch.randn(
                1, num_channels, individual_positional_embedding_size
            )
        )

    def forward(self, x, i=0, h=None):
        assert i == 0
        pos_embedding = self.pe.repeat_interleave(
            self.num_channels, dim=1
        )
        pos_embedding = pos_embedding[:, :x.size(1), :]
        pos_embedding = torch.repeat_interleave(pos_embedding, x.shape[0], dim=0)

        channel_embeddings = self.channel_embeddings.repeat(
            x.size(0),
            x.size(1) // self.num_channels + 1,
            1
        )
        channel_embeddings = channel_embeddings[:, :x.size(1)]

        x = torch.cat([x, pos_embedding, channel_embeddings], dim=2)
        return self.dropout(x), h

    def forward_step(self, x, i=0, h=None):
        pe_index = i // self.num_channels
        channel_index = i % self.num_channels
        pos_embedding = self.pe[:, pe_index].repeat(x.size(0), 1)
        channel_embeddings = self.channel_embeddings[:, channel_index].repeat(
            x.size(0), 1
        )
        x = torch.cat(
            [x, pos_embedding, channel_embeddings],
            dim=1
        )
        return self.dropout(x), h