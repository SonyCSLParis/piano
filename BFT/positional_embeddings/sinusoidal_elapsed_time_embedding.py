from torch import nn
import torch
import math

class SinusoidalElapsedTimeEmbedding(nn.Module):
    def __init__(self, positional_embedding_size, num_channels, **kwargs):
        super(SinusoidalElapsedTimeEmbedding, self).__init__()
        assert positional_embedding_size % 2 == 0
        self.dataloader_generator = kwargs['dataloader_generator']

        self.dropout = torch.nn.Dropout(p=kwargs['dropout'])
        self.individual_positional_embedding_size = positional_embedding_size // 2
        self.num_channels = num_channels

        self.channel_embeddings = nn.Parameter(
            torch.randn(
                1, num_channels, self.individual_positional_embedding_size
            )
        )

    def forward(self, x_embed, i=0, h=None, target=None):
        assert i == 0
        x = target
        batch_size, num_events, num_channels = x.size()
        # batch_size, num_tokens, embedding_dim = x_embed.size()
        elapsed_time = self.dataloader_generator.get_elapsed_time(
            x
        )
        # add zeros
        elapsed_time = torch.cat(            
            [
                torch.zeros_like(elapsed_time)[:, :num_channels],
                elapsed_time[:, :-num_channels]
            ],
            dim=1
        )
        
        # add embedding_dim to elapsed tim
        elapsed_time = elapsed_time.unsqueeze(2)
        
        # elapsed_time * 4 en moyenne 4 notes par seconde
        elapsed_time = (elapsed_time * 4).detach()
        
        pe = torch.zeros(batch_size, 
                         num_events, self.individual_positional_embedding_size)
        pe = pe.to(device=x.device)
        
        div_term = torch.exp(torch.arange(0, self.individual_positional_embedding_size, 2).float() * (
                -math.log(10000.0) / self.individual_positional_embedding_size))
        div_term = div_term.to(device=x.device)
        div_term = div_term.unsqueeze(0).unsqueeze(0)
        pe[:, :, 0::2] = torch.sin(elapsed_time * div_term)
        pe[:, :, 1::2] = torch.cos(elapsed_time * div_term)
        
        
        pos_embedding = pe.repeat_interleave(
            self.num_channels, dim=1
        )

        channel_embeddings = self.channel_embeddings.repeat(
            x_embed.size(0),
            x_embed.size(1) // self.num_channels + 1,
            1
        )
        channel_embeddings = channel_embeddings[:, :x_embed.size(1)]

        x_embed = torch.cat([x_embed, pos_embedding, channel_embeddings], dim=2)
        return self.dropout(x_embed), h

    def forward_step(self, x, i=0, h=None):
        # TODO TODO
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