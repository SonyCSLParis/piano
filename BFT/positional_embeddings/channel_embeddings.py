from torch import nn
import torch


class ChannelEmbeddings(nn.Module):

    def __init__(self,
                 positional_embedding_size,
                 num_channels,
                 # **kwargs
                 ):
        super(ChannelEmbeddings, self).__init__()
        self.num_channels = num_channels

        self.pe_0 = nn.Parameter(
            torch.randn(
                1, num_channels, positional_embedding_size
            )
        )

    def forward(self, x, i=0, h=None):
        """

        :param x: (
        batch_size,
        num_tokens
        d_model - positional_embedding_size
        )
        :param i:
        :return:
        """
        batch_size, num_tokens, _ = x.size()

        # create init sequence
        num_events = num_tokens // self.num_channels + 1
        positional_embeddings = self.pe_0.repeat(batch_size, num_events, 1)
        offset = i % self.num_channels

        # slice so that we have the correct offset
        positional_embeddings = positional_embeddings[:, offset: offset + num_tokens]

        x = torch.cat([
            x, positional_embeddings
        ], dim=2)

        return x, h

    def forward_step(self, x, i=0, h=None):
        """

        :param x: (
        batch_size,
        d_model - positional_embedding_size
        )
        :param i:
        :return:
        """
        # TODO can be done better
        batch_size, _ = x.size()
        x = x.unsqueeze(1)
        batch_size, num_tokens, _ = x.size()
        # create init sequence
        num_events = num_tokens // self.num_channels + 1
        positional_embeddings = self.pe_0.repeat(batch_size, num_events, 1)
        offset = i % self.num_channels

        # slice so that we have the correct offset
        positional_embeddings = positional_embeddings[:, offset: offset + num_tokens]

        x = torch.cat([
            x, positional_embeddings
        ], dim=2)

        x = x[:, 0]
        return x, h


