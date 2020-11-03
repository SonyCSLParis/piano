from BFT.positional_embeddings.positional_embedding import BasePositionalEmbedding
from torch import nn
import torch
import math

class SinusoidalElapsedTimeEmbedding(BasePositionalEmbedding):
    def __init__(self, positional_embedding_size, num_channels, 
                 dataloader_generator, dropout, **kwargs):
        super(SinusoidalElapsedTimeEmbedding, self).__init__()
        assert positional_embedding_size % 2 == 0
        self.dataloader_generator = dataloader_generator
        self.positional_embedding_size = positional_embedding_size

        self.dropout = torch.nn.Dropout(p=dropout)
        self.num_channels = num_channels

        

    def forward(self, x_embed, i=0, h=None, target=None):
        assert i == 0
        if h is None:
            h = torch.zeros_like(x_embed[:, 0, 0])
            
        x = target
        batch_size, num_events, num_channels = x.size()
        # batch_size, num_tokens, embedding_dim = x_embed.size()
        elapsed_time = self.dataloader_generator.get_elapsed_time(
            x
        )
        # add zeros
        elapsed_time = torch.cat(            
            [
                torch.zeros_like(elapsed_time)[:, :1],
                elapsed_time[:, :-1]
            ],
            dim=1
        )
        h = elapsed_time[:, -1]
        
        # add embedding_dim to elapsed time
        elapsed_time = elapsed_time.unsqueeze(2)
        
        pe = torch.zeros(batch_size, 
                         num_events, self.positional_embedding_size)
        pe = pe.to(device=x.device)
        
        div_term = torch.exp(torch.arange(0, self.positional_embedding_size, 2).float() * (
                -math.log(10000.0) / self.positional_embedding_size))
        div_term = div_term.to(device=x.device)
        div_term = div_term.unsqueeze(0).unsqueeze(0)
        pe[:, :, 0::2] = torch.sin(elapsed_time * div_term)
        pe[:, :, 1::2] = torch.cos(elapsed_time * div_term)
        
        
        pos_embedding = pe.repeat_interleave(
            self.num_channels, dim=1
        )

        x_embed = torch.cat([x_embed, pos_embedding], dim=2)
        return self.dropout(x_embed), h

    def forward_step(self, x, i=0, h=None, target=None):
        # time_shift must be the last feature
        assert self.dataloader_generator.features.index('time_shift') == len(self.dataloader_generator.features) - 1
        
        batch_size = x.size(0)
        # h represents the elapsed time
        if h is None:
            h = torch.zeros((batch_size, )).to(x.device)
            
        elapsed_time = h.unsqueeze(1)
        
        pe = torch.zeros(batch_size, 
                         self.positional_embedding_size)
        pe = pe.to(device=x.device)
        
        div_term = torch.exp(torch.arange(0, self.positional_embedding_size, 2).float() * (
                -math.log(10000.0) / self.positional_embedding_size))
        div_term = div_term.to(device=x.device)
        div_term = div_term.unsqueeze(0)
        pe[:, 0::2] = torch.sin(elapsed_time * div_term)
        pe[:, 1::2] = torch.cos(elapsed_time * div_term)
        
        x_embed = torch.cat([x, pe], dim=1)
        
        # update h if the current token is a time_shift:
        if i % self.num_channels == self.num_channels - 1:
            # add fake features so that we can call get_elapsed_time
            target = target.unsqueeze(1).unsqueeze(1)
            target = target.repeat(1, 1, self.num_channels)
            elapsed_time = self.dataloader_generator.get_elapsed_time(
            target
        )
            elapsed_time = elapsed_time.squeeze(1)
            h = h + elapsed_time
        
        return self.dropout(x_embed), h