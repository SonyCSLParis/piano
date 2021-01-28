from fast_transformers.builders import TransformerEncoderBuilder, RecurrentEncoderBuilder, TransformerDecoderBuilder, RecurrentDecoderBuilder, TransformerDiagonalDecoderBuilder, RecurrentDiagonalDecoderBuilder
from fast_transformers.builders.transformer_builders import TransformerDiagonalDecoderBuilderWithStates, TransformerEncoderBuilderWithStates

from fast_transformers.masking import TriangularCausalMask

from torch import nn

import torch


class LinearTransformerCausalEncoder(nn.Module):
    def __init__(self, d_model, n_heads, n_layers, dim_feedforward, recurrent):
        super(LinearTransformerCausalEncoder, self).__init__()

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
                final_normalization=True).get()
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
                final_normalization=True).get()

            self.transformer_with_states = TransformerEncoderBuilderWithStates.from_kwargs(
                n_layers=n_layers,
                n_heads=n_heads,
                query_dimensions=query_dimension,
                value_dimensions=query_dimension,
                feed_forward_dimensions=dim_feedforward,
                dropout=dropout,
                attention_type="causal-linear-states",
                activation="gelu",
                final_normalization=True).get()

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

    def forward_with_states(self, x):
        triangular_mask = TriangularCausalMask(x.size(1), device=x.device)
        return self.transformer_with_states(x, attn_mask=triangular_mask)


class LinearTransformerEncoder(nn.Module):
    def __init__(self, d_model, n_heads, n_layers, dim_feedforward, recurrent):
        super(LinearTransformerEncoder, self).__init__()

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
                attention_type="linear",
                activation="gelu",
                final_normalization=True).get()
        else:
            self.transformer = RecurrentEncoderBuilder.from_kwargs(
                n_layers=n_layers,
                n_heads=n_heads,
                query_dimensions=query_dimension,
                value_dimensions=query_dimension,
                feed_forward_dimensions=dim_feedforward,
                dropout=dropout,
                attention_type="linear",
                activation="gelu",
                final_normalization=True).get()

    def forward(self, x):
        """
        Here, transformer is non recurrent
        :param x: (batch_size, num_tokens, feature_dim)
        :return:
        """
        return self.transformer(x, attn_mask=None)

    def forward_step(self, x, state):
        """
        Here, transformer is recurrent
        :param x: (batch_size, feature_dim)
        :param state:
        :return: (output, state)
        """
        # This method should never be called
        raise NotImplementedError


class LinearTransformerAnticausalEncoder(nn.Module):
    def __init__(self, d_model, n_heads, n_layers, dim_feedforward, recurrent):
        super(LinearTransformerAnticausalEncoder, self).__init__()

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
                final_normalization=True).get()
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
                final_normalization=True).get()

    def forward(self, x):
        """
        Here, transformer is non recurrent
        :param x: (batch_size, num_tokens, feature_dim)
        :return:
        """
        triangular_mask = TriangularCausalMask(x.size(1), device=x.device)
        x = torch.flip(x, dims=[1])
        x = self.transformer(x, attn_mask=triangular_mask)
        return torch.flip(x, dims=[1])

    def forward_step(self, x, state):
        """
        Here, transformer is recurrent
        :param x: (batch_size, feature_dim)
        :param state:
        :return: (output, state)
        """
        # This method should never be called
        raise NotImplementedError


class LinearTransformerCausalDecoder(nn.Module):
    def __init__(self, d_model, n_heads, n_layers, dim_feedforward, recurrent):
        super(LinearTransformerCausalDecoder, self).__init__()

        query_dimension = d_model // n_heads
        dropout = 0.1
        attention_dropout = 0.1
        if not recurrent:
            self.transformer = TransformerDecoderBuilder.from_kwargs(
                n_layers=n_layers,
                n_heads=n_heads,
                query_dimensions=query_dimension,
                value_dimensions=query_dimension,
                feed_forward_dimensions=dim_feedforward,
                dropout=dropout,
                self_attention_type="causal-linear",
                cross_attention_type='linear',
                activation="gelu",
                final_normalization=True).get()
        else:
            self.transformer = RecurrentDecoderBuilder.from_kwargs(
                n_layers=n_layers,
                n_heads=n_heads,
                query_dimensions=query_dimension,
                value_dimensions=query_dimension,
                feed_forward_dimensions=dim_feedforward,
                dropout=dropout,
                self_attention_type="causal-linear",
                cross_attention_type='linear',
                activation="gelu",
                final_normalization=True).get()

    def forward(self, memory, target):
        """
        Here, transformer is non recurrent
        :param x: (batch_size, num_tokens, feature_dim)
        :return:
        """
        triangular_mask = TriangularCausalMask(target.size(1),
                                               device=target.device)
        return self.transformer(x=target,
                                memory=memory,
                                x_mask=triangular_mask,
                                memory_mask=None)

    def forward_step(self, memory, x, state):
        """
        Here, transformer is recurrent
        :param x: (batch_size, feature_dim)
        :param state:
        :return: (output, state)
        """

        return self.transformer(x=x, memory=memory, state=state)


class LinearTransformerCausalDiagonalDecoder(nn.Module):
    def __init__(self, d_model, n_heads, n_layers, dim_feedforward, recurrent):
        super(LinearTransformerCausalDiagonalDecoder, self).__init__()

        query_dimension = d_model // n_heads
        dropout = 0.1
        attention_dropout = 0.1
        if not recurrent:
            self.transformer = TransformerDiagonalDecoderBuilder.from_kwargs(
                n_layers=n_layers,
                n_heads=n_heads,
                query_dimensions=query_dimension,
                value_dimensions=query_dimension,
                feed_forward_dimensions=dim_feedforward,
                dropout=dropout,
                self_attention_type="causal-linear",
                cross_attention_type='diagonal',
                activation="gelu",
                final_normalization=True).get()
        else:
            self.transformer = RecurrentDiagonalDecoderBuilder.from_kwargs(
                n_layers=n_layers,
                n_heads=n_heads,
                query_dimensions=query_dimension,
                value_dimensions=query_dimension,
                feed_forward_dimensions=dim_feedforward,
                dropout=dropout,
                self_attention_type="causal-linear",
                cross_attention_type='diagonal',
                activation="gelu",
                final_normalization=True).get()

            self.transformer_with_states = TransformerDiagonalDecoderBuilderWithStates.from_kwargs(
                n_layers=n_layers,
                n_heads=n_heads,
                query_dimensions=query_dimension,
                value_dimensions=query_dimension,
                feed_forward_dimensions=dim_feedforward,
                dropout=dropout,
                self_attention_type="causal-linear-states",
                cross_attention_type='diagonal-states',
                activation="gelu",
                final_normalization=True).get()

    def forward(self, memory, target):
        """
        Here, transformer is non recurrent
        :param x: (batch_size, num_tokens, feature_dim)
        :return:
        """
        triangular_mask = TriangularCausalMask(target.size(1),
                                               device=target.device)
        return self.transformer(x=target,
                                memory=memory,
                                x_mask=triangular_mask,
                                memory_mask=None)

    def forward_with_states(self, memory, target):
        """
        Same as forward, but state is returned. Used only during inference
        Here, transformer is non recurrent
        :param x: (batch_size, num_tokens, feature_dim)
        :return:
        """
        triangular_mask = TriangularCausalMask(target.size(1),
                                               device=target.device)
        return self.transformer_with_states(x=target,
                                            memory=memory,
                                            x_mask=triangular_mask,
                                            memory_mask=None)

    def forward_step(self, memory, x, state):
        """
        Here, transformer is recurrent
        :param x: (batch_size, feature_dim)
        :param state:
        :return: (output, state)
        """

        return self.transformer(x=x, memory=memory, state=state)
