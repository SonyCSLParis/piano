from BFT.start_of_sequence_embeddings import SOSEmbedding
from BFT.positional_embeddings import PositionalEmbedding
from BFT.data_processors import SourceTargetDataProcessor
from torch import nn

from BFT.dataloaders import DataloaderGenerator

from BFT.transformers.linear_transformer import LinearTransformerAnticausalEncoder, LinearTransformerCausalDiagonalDecoder
from BFT.utils import flatten, categorical_crossentropy
import torch


class EncoderDecoder(nn.Module):
    def __init__(self,
                 data_processor: SourceTargetDataProcessor,
                 dataloader_generator: DataloaderGenerator,
                 positional_embedding_source: PositionalEmbedding,
                 positional_embedding_target: PositionalEmbedding,
                 sos_embedding: SOSEmbedding,
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
        """EncoderDecoder with linear attention trained on a next-character prediction task
        
        AC/D/C model
        For now, assumes num_tokens_source == num_tokens_target

        Args:
            data_processor (SourceTargetDataProcessor): 
            dataloader_generator (DataloaderGenerator): 
            positional_embedding_source (PositionalEmbedding): 
            positional_embedding_target (PositionalEmbedding): 
            d_model_encoder (int): 
            d_model_decoder (int): 
            num_layers_encoder (int): 
            num_layers_decoder (int): 
            n_head_encoder (int): 
            n_head_decoder (int): 
            dim_feedforward_encoder (int): 
            dim_feedforward_decoder (int): 
            num_channels_source (int): 
            num_channels_target (int): 
            num_events_source (int): 
            num_events_target (int): 
            dropout (float): 
            label_smoothing (int): 
            recurrent (bool, optional): If True, uses a recurrent linear transformer for the DECODER PART (usage is like an RNN) for inference. Use only forward_step() in this case. Otherwise, standard linear transformer used for training. Use only forward() in this case. Defaults to False.
        """
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
            self.data_processor.embedding_size_source +
            self.positional_embedding_source.positional_embedding_size,
            self.d_model)

        self.linear_target = nn.Linear(
            self.data_processor.embedding_size_target +
            self.positional_embedding_target.positional_embedding_size,
            self.d_model)

        ########################################################
        # Start of sentence
        self.sos_embedding = sos_embedding

        ######################################################
        self.encoder = LinearTransformerAnticausalEncoder(
            d_model=d_model_encoder,
            n_heads=n_head_encoder,
            n_layers=num_layers_encoder,
            dim_feedforward=dim_feedforward_encoder,
            recurrent=False)

        self.decoder = LinearTransformerCausalDiagonalDecoder(
            d_model=d_model_decoder,
            n_heads=n_head_decoder,
            n_layers=num_layers_decoder,
            dim_feedforward=dim_feedforward_decoder,
            recurrent=recurrent)
        self.label_smoothing = label_smoothing
        self.recurrent = recurrent

        ######################################################
        # Output dimension adjustment
        self.pre_softmaxes = nn.ModuleList([
            nn.Linear(self.d_model, num_tokens_of_channel)
            for num_tokens_of_channel in
            self.data_processor.decoder_data_processor.num_tokens_per_channel
        ])

    def __repr__(self) -> str:
        return 'EncoderDecoder'

    def forward_source(self, source, metadata_dict):
        source_embedded = self.data_processor.embed_source(source)

        # add positional embeddings and flatten and to d_model
        source_seq = flatten(source_embedded)

        # since Encoder is bidirectionnal or anticausal, h is always None
        source_seq, h_pe_source = self.positional_embedding_source(
            source_seq, metadata_dict=metadata_dict, i=0, h=None)
        source_seq = self.linear_source(source_seq)

        # encode
        memory = self.encoder(source_seq)
        return memory

    def forward_memory_target(self,
                              memory,
                              target,
                              metadata_dict,
                              h_pe_init=None):
        """Call decoder on target conditionned on memory (output of the encoder)

        Args:
            memory (FloatTensor): (batch_size, num_tokens_source, d_model_encoder)
            target (LongTensor): [description]
            h_pe_init ([type], optional): [description]. Defaults to None.

        Returns:
            [type]: [description]
        """
        batch_size, num_events_target, num_channels_source = target.size()
        target_embedded = self.data_processor.embed_target(target)

        # add positional embeddings
        target_seq = flatten(target_embedded)
        target_seq, h_pe_target = self.positional_embedding_target(
            target_seq, i=0, h=h_pe_init, metadata_dict=metadata_dict)
        target_seq = self.linear_target(target_seq)

        # shift target_seq by one
        # Pad
        # sos_embedding is (batch_size, d_model_decoder)
        dummy_input_target = self.sos_embedding(metadata_dict).unsqueeze(1)
        target_seq = torch.cat([dummy_input_target, target_seq], dim=1)
        target_seq = target_seq[:, :-1]

        output = self.decoder(memory=memory, target=target_seq)

        output = output.view(batch_size, -1, self.num_channels_target,
                             self.d_model)
        weights_per_category = [
            pre_softmax(t[:, :, 0, :])
            for t, pre_softmax in zip(output.split(1, 2), self.pre_softmaxes)
        ]
        return weights_per_category, h_pe_target
    
    def forward_memory_target_with_states(self,
                              memory,
                              target,
                              metadata_dict,
                              h_pe_init=None):
        """Call decoder on target conditionned on memory (output of the encoder)

        Args:
            memory (FloatTensor): (batch_size, num_tokens_source, d_model_encoder)
            target (LongTensor): [description]
            h_pe_init ([type], optional): [description]. Defaults to None.

        Returns:
            [type]: [description]
        """
        batch_size, num_events_target, num_channels_source = target.size()
        target_embedded = self.data_processor.embed_target(target)

        # add positional embeddings
        target_seq = flatten(target_embedded)
        target_seq, h_pe_target = self.positional_embedding_target(
            target_seq, i=0, h=h_pe_init, metadata_dict=metadata_dict)
        target_seq = self.linear_target(target_seq)

        # shift target_seq by one
        # Pad
        # sos_embedding is (batch_size, d_model_decoder)
        dummy_input_target = self.sos_embedding(metadata_dict).unsqueeze(1)
        target_seq = torch.cat([dummy_input_target, target_seq], dim=1)
        target_seq = target_seq[:, :-1]

        output, states = self.decoder.forward_with_states(memory=memory, target=target_seq)

        output = output.view(batch_size, -1, self.num_channels_target,
                             self.d_model)
        weights_per_category = [
            pre_softmax(t[:, :, 0, :])
            for t, pre_softmax in zip(output.split(1, 2), self.pre_softmaxes)
        ]
        return weights_per_category, h_pe_target, states

    def forward(self, source, target, metadata_dict, h_pe_init=None):
        """
        :param source: sequence of tokens (batch_size, num_events_source, num_channels_source)
        :param target: sequence of tokens (batch_size, num_events_target, num_channels_target)
        :param metadata_dict: dictionnary of tensors
        :return: dict containing loss, output, and monitored_quantities
        """

        # --- Source
        memory = self.forward_source(source, metadata_dict)

        # --- Target
        weights_per_category, h_pe_target = self.forward_memory_target(
            memory=memory,
            target=target,
            metadata_dict=metadata_dict,
            h_pe_init=h_pe_init)

        # we can change loss mask
        # mask = torch.ones_like(target)
        mask = metadata_dict['masked_positions']
        
        # so that we do not predict PAD and START symbols
        if 'loss_mask' in metadata_dict:
            mask = torch.logical_and(mask.bool(), torch.logical_not(metadata_dict['loss_mask']))
        loss = categorical_crossentropy(value=weights_per_category,
                                        target=target,
                                        mask=mask,
                                        label_smoothing=self.label_smoothing)

        # loss = loss.mean()
        return {
            'loss': loss,
            'h_pe_target': h_pe_target,
            'weights_per_category': weights_per_category,
            'monitored_quantities': {
                'loss': loss.item()
            }
        }


    def forward_step(self, memory, target, metadata_dict, state, i, h_pe):
        """
        Recurrent version of forward_memory_target
        Assumes memory is the output of the encoder
        
        i is the index of target that is to BE predicted:
        if i == 0, target is not used: SOS instead
        
        :param memory: (batch_size, num_tokens_source, d_model)
        :param target: sequence of tokens (batch_size,)
        :param state:
        :param i:
        :param h_pe:
        :return:
        """
        # deal with the SOS token embedding
        if i == 0:
            target_seq = self.sos_embedding(metadata_dict)
        else:
            channel_index_input = (i - 1) % self.num_channels_target
            # embed target
            target_embedded = self.data_processor.embed_step_target(
                target, channel_index=channel_index_input)
            # add positional embeddings
            metadata_dict['original_token'] = target
            target_embedded, h_pe = self.positional_embedding_target.forward_step(
                target_embedded,
                i=(i - 1),
                h=h_pe,
                metadata_dict=metadata_dict)
            target_seq = self.linear_target(target_embedded)
            
        # TODO(gaetan) check state with diagonal
        output, state = self.decoder.forward_step(memory=memory,
                                                  x=target_seq,
                                                  state=state)

        channel_index_output = i % self.num_channels_target

        weights = self.pre_softmaxes[channel_index_output](output)

        # no need for a loss
        return {
            'loss': None,
            'state': state,
            'h_pe': h_pe,
            'weights': weights,
        }
