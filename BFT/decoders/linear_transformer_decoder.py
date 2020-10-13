from datetime import datetime


from fast_transformers.masking import TriangularCausalMask
from torch import nn
import numpy as np

from tqdm import tqdm

from BFT.data_processors.data_processor import DataProcessor
from BFT.dataloaders.dataloader import DataloaderGenerator
from BFT.positional_embeddings.channel_embeddings import ChannelEmbeddings
from BFT.positional_embeddings.learnt_embeddings import LearntEmbeddings
from BFT.positional_embeddings.recurrent_positional_embedding import RecurrentPositionalEmbedding
from BFT.positional_embeddings.sinusoidal_positional_embedding import SinusoidalPositionalEmbedding, \
    SinusoidalPositionalEmbeddingChannels
from BFT.transformers.linear_transformer import LinearTransformer
from BFT.utils import flatten, categorical_crossentropy, dict_pretty_print, top_k_top_p_filtering, \
    to_numpy, cuda_variable
import os
import torch


class LinearTransformerDecoder(nn.Module):
    def __init__(self,
                 data_processor: DataProcessor,
                 d_model,
                 num_decoder_layers,
                 n_head,
                 dim_feedforward,
                 positional_embedding_size,
                 num_channels_decoder,
                 num_events_decoder,
                 dropout,
                 # TODO pass arguments
                 recurrent=False):
        # TODO Signature
        super(LinearTransformerDecoder, self).__init__()
        self.data_processor = data_processor

        # Compute num_tokens for source and target
        self.num_tokens_per_channel = self.data_processor.num_tokens_per_channel
        self.num_channels_target = len(self.num_tokens_per_channel)
        assert self.num_channels_target == num_channels_decoder
        self.d_model = d_model
        self.num_tokens_target = self.data_processor.num_tokens

        assert self.num_tokens_target == num_channels_decoder * num_events_decoder

        ######################################################
        # Embeddings
        self.target_positional_embedding = SinusoidalPositionalEmbeddingChannels(
            positional_embedding_size=positional_embedding_size,
            num_channels=num_channels_decoder,
            num_tokens_max=1024 * 4,  # TODO hard coded
            dropout=0.1
        )

        linear_target_input_size = self.d_model - positional_embedding_size
        self.linear_target = nn.Linear(
            self.data_processor.embedding_size,
            linear_target_input_size
        )

        ########################################################
        # Start of sentence
        self.sos_target = nn.Parameter(torch.randn((1, 1, self.d_model)))

        ######################################################
        self.transformer = LinearTransformer(
            d_model=d_model,
            n_heads=n_head,
            n_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            recurrent=recurrent
        )

        ######################################################
        # Output dimension adjustment
        self.pre_softmaxes = nn.ModuleList([nn.Linear(self.d_model, num_tokens_of_channel)
                                            for num_tokens_of_channel in self.num_tokens_per_channel
                                            ]
                                           )
        
    def __repr__(self) -> str:
        return 'LinearTransformerDecoder'

    def forward(self, target, h_pe_init=None):
        """
        :param target: sequence of tokens (batch_size, num_events, num_channels)
        :return:
        """
        batch_size, num_events, num_channels = target.size()

        target = self.data_processor.preprocess(target)
        target_embedded = self.data_processor.embed(target)
        target_embedded = self.linear_target(target_embedded)
        target_seq = flatten(target_embedded)

        num_tokens_target = target_seq.size(1)
        # add positional embeddings
        target_seq, h_pe = self.target_positional_embedding(target_seq, h=h_pe_init)

        # shift target_seq by one
        # Pad

        dummy_input_target = self.sos_target.repeat(batch_size, 1, 1)
        target_seq = torch.cat(
            [
                dummy_input_target,
                target_seq
            ],
            dim=1)
        target_seq = target_seq[:, :-1]

        output = self.transformer(
            target_seq
        )

        output = output.view(batch_size,
                             -1,
                             self.num_channels_target,
                             self.d_model)
        weights_per_category = [
            pre_softmax(t[:, :, 0, :])
            for t, pre_softmax in zip(output.split(1, 2), self.pre_softmaxes)
        ]

        # we can change loss mask
        loss = categorical_crossentropy(
            value=weights_per_category,
            target=target,
            mask=torch.ones_like(target)
        )

        loss = loss.mean()
        return {
            'loss':                 loss,
            'h_pe':                 h_pe,
            'weights_per_category': weights_per_category,
            'monitored_quantities': {
                'loss': loss.item()
            }
        }

    def forward_step(self, target, state, i, h_pe):
        """
        if i == 0, target is not used: SOS instead
        :param target: sequence of tokens (batch_size,)
        :param state:
        :param i:
        :param h_pe:
        :return:
        """
        # deal with the SOS token embedding
        if i == 0:
            target_seq = self.sos_target.repeat(target.size(0), 1, 1)[:, 0, :]
        else:
            channel_index_input = (i - 1) % self.num_channels_target
            target = self.data_processor.preprocess(target)
            target_embedded = self.data_processor.embed_step(
                target,
                channel_index=channel_index_input)
            target_embedded = self.linear_target(target_embedded)
            # add positional embeddings
            target_seq, h_pe = self.target_positional_embedding.forward_step(
                target_embedded,
                i=(i - 1),
                h=h_pe)

        output, state = self.transformer.forward_step(
            target_seq, state=state
        )

        channel_index_output = i % self.num_channels_target

        weights = self.pre_softmaxes[channel_index_output](output)

        # no need for a loss
        return {
            'loss':    None,
            'state':   state,
            'h_pe':    h_pe,
            'weights': weights,
        }

