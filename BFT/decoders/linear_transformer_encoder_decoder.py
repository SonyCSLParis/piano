
from BFT.positional_embeddings.positional_embedding import PositionalEmbedding
from BFT.positional_embeddings.sinusoidal_elapsed_time_embedding import SinusoidalElapsedTimeEmbedding
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
from BFT.positional_embeddings.sinusoidal_positional_embedding import SinusoidalPositionalEmbedding
from BFT.transformers.linear_transformer import LinearTransformerCausalDecoder, LinearTransformerCausalEncoder, LinearTransformerEncoder
from BFT.utils import flatten, categorical_crossentropy, dict_pretty_print, top_k_top_p_filtering, \
    to_numpy, cuda_variable
import os
import torch


class EncoderDecoder(nn.Module):
    def __init__(self,
                 data_processor: DataProcessor,
                 dataloader_generator: DataloaderGenerator,
                 positional_embedding_source: PositionalEmbedding,
                 positional_embedding_target: PositionalEmbedding,
                 d_model_encoder,
                 d_model_decoder,
                 num_layers_encoder,
                 num_layers_decoder,
                 n_head_encoder,
                 n_head_decoder,
                 dim_feedforward_encoder,
                 dim_feedforward_decoder,
                 num_channels_source,
                 num_channels_target,
                 num_events_source,
                 num_events_target,
                 dropout,
                 label_smoothing,
                 recurrent=False):
        # TODO Signature
        super(EncoderDecoder, self).__init__()
        self.data_processor = data_processor
        # can be useful
        self.dataloader_generator = dataloader_generator

        # Compute num_tokens for source and target
        
        self.num_channels_target = num_channels_target
        self.num_channels_source = num_channels_source

        ######################################################
        # Positional Embeddings
        
        self.positional_embedding_source = positional_embedding_source
        self.positional_embedding_target = positional_embedding_target
        
        assert d_model_encoder == d_model_decoder
        self.d_model = d_model_encoder
        
        self.linear_source = nn.Linear(
            self.data_processor.embedding_size_source 
            + self.positional_embedding_source.positional_embedding_size,
            self.d_model
        )
        
        self.linear_target = nn.Linear(
            self.data_processor.embedding_size_target 
            + self.positional_embedding_target.positional_embedding_size,
            self.d_model
        )

        ########################################################
        # Start of sentence
        self.sos_target = nn.Parameter(torch.randn((1, 1, self.d_model)))

        ######################################################
        self.encoder = LinearTransformerEncoder(
            d_model=d_model_encoder,
            n_heads=n_head_encoder,
            n_layers=num_layers_encoder,
            dim_feedforward=dim_feedforward_encoder,
            recurrent=recurrent
        )
        
        self.decoder = LinearTransformerCausalDecoder(
            d_model=d_model_decoder,
            n_heads=n_head_decoder,
            n_layers=num_layers_decoder,
            dim_feedforward=dim_feedforward_decoder,
            recurrent=recurrent
        )
        self.label_smoothing = label_smoothing
        self.recurrent = recurrent
        
        ######################################################
        # Output dimension adjustment
        self.pre_softmaxes = nn.ModuleList([nn.Linear(self.d_model, num_tokens_of_channel)
                                            for num_tokens_of_channel in self.data_processor.decoder_data_processor.num_tokens_per_channel
                                            ]
                                           )
        
    def __repr__(self) -> str:
        return 'LinearTransformerEncoderDecoder'

    def forward(self, source, target, h_pe_init=None):
        """
        :param source: sequence of tokens (batch_size, num_events_source, num_channels_source)
        :param target: sequence of tokens (batch_size, num_events_target, num_channels_target)
        :return:
        """
        batch_size, num_events_target, num_channels_target = target.size()
        batch_size, num_events_source, num_channels_source = source.size()
        
        # --- Source
        source_embedded = self.data_processor.embed_source(source)
        
        # add positional embeddings and flatten and to d_model
        source_seq = flatten(source_embedded)
        source_seq, h_pe_source = self.positional_embedding_source(source_seq, i=0, h=h_pe_init, target=source)        
        source_seq = self.linear_source(source_seq)
        
        # encode
        memory = self.encoder(source_seq)
        
        # --- Target
        target_embedded = self.data_processor.embed_target(target)
        
        # add positional embeddings
        target_seq = flatten(target_embedded)
        target_seq, h_pe_target = self.positional_embedding_target(target_seq, i=0, h=h_pe_init, target=target)
        target_seq = self.linear_target(target_seq)

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

        output = self.decoder(
            memory=memory,
            target=target_seq
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
            mask=torch.ones_like(target),
            label_smoothing=self.label_smoothing
        )

        loss = loss.mean()
        return {
            'loss':                 loss,
            'h_pe_target':                 h_pe_target,
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
        # TODO
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
                h=h_pe,
                target=target)

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

