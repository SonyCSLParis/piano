from torch import nn


class BasePositionalEmbedding(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # BasePositionalEmbedding must define
        # positional_embedding_size

    def forward(self, x_embed, i=0, h=None, target=None):
        return x_embed, h

    def forward_step(self, x_embed, i=0, h=None, target=None):
        return x_embed, h


class PositionalEmbedding(nn.Module):
    """Positional embeddings built from a list of
    "base" positional embeddings like
    - ChannelEmbedding,
    - SinusoidalEmbedding
    - etc.
    """
    def __init__(self, base_positional_embedding_list) -> None:
        super().__init__()
        self.base_positional_embeddings = nn.ModuleList(
            base_positional_embedding_list)
        self.positional_embedding_size = sum([
            pe.positional_embedding_size
            for pe in base_positional_embedding_list
        ])

    def forward(self, x_embed, i=0, h=None, target=None):
        """Concatenates all the simple_positional_embeddings
        on the last dim of x_embed

        Args:
            x_embed (batch_size, num_tokens, embedding_dim): embedded_sequence
            i (int, optional): index of the first token. Defaults to 0.
            h (list of tensors, optional): cached values, one for each embedding. Defaults to None.
            target (batch_size, num_events_num_channels, optional):
            The target tensor (not embedded), can be used compute some quantities. Defaults to None.
            
        Output:
            x_embed_with
        """
        if isinstance(h, list):
            pass
        else:
            if h is None:
                h = [None] * len(
                    self.base_positional_embeddings
                    )
                
        new_h_list = []
        for positional_embedding, h_pe in zip(self.base_positional_embeddings, h):
            x_embed, new_h_pe = positional_embedding.forward(x_embed,
                                                   i=i,
                                                   h=h_pe,
                                                   target=target)
            new_h_list.append(new_h_pe)
        return x_embed, new_h_list

    def forward_step(self, x_embed, i=0, h=None, target=None):
        """Concatenates all the simple_positional_embeddings
        on the last dim of x_embed

        Args:
            x_embed (batch_size, embedding_dim): embedded_sequence
            i (int, optional): index of the token in the whole sequence. Defaults to 0.
            h (list of tensors, optional): cached values, one for each embedding. Defaults to None.
            target (batch_size, num_events_num_channels, optional):
            The target tensor (not embedded), can be used compute some quantities. Defaults to None.
        """
        if isinstance(h, list):
            pass
        else:
            if h is None:
                h = [None] * len(
                    self.base_positional_embeddings
                    )
                
                
        new_h_list = []
        for positional_embedding, h_pe in zip(self.base_positional_embeddings, h):
            x_embed, new_h_pe = positional_embedding.forward_step(x_embed,
                                                   i=i,
                                                   h=h_pe,
                                                   target=target)
            new_h_list.append(new_h_pe)
        return x_embed, new_h_list