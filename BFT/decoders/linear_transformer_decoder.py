from BFT.positional_embeddings import PositionalEmbedding

from torch import nn

from BFT.data_processors import DataProcessor
from BFT.dataloaders.dataloader import DataloaderGenerator

from BFT.transformers.linear_transformer import LinearTransformerCausalEncoder
from BFT.utils import flatten, categorical_crossentropy
import torch


class CausalEncoder(nn.Module):
    def __init__(self,
                 data_processor: DataProcessor,
                 dataloader_generator: DataloaderGenerator,
                 positional_embedding: PositionalEmbedding,
                 d_model,
                 num_decoder_layers,
                 n_head,
                 dim_feedforward,
                 num_channels_decoder,
                 num_events_decoder,
                 dropout,
                 label_smoothing,
                 recurrent=False):
        """CausalEncoder with linear attention trained on a next-character prediction task

        Args:
            data_processor (DataProcessor): 
            dataloader_generator (DataloaderGenerator): 
            positional_embedding (PositionalEmbedding): 
            d_model (int): 
            num_decoder_layers (int):
            n_head (int):
            dim_feedforward (int): 
            num_channels_decoder ([int]):
            num_events_decoder ([int]): 
            dropout ([float]): 
            label_smoothing ([bool]): 
            recurrent (bool, optional): If True, uses a recurrent linear transformer encoder (usage is like an RNN) for inference. Use only forward_step() in this case. Otherwise, standard linear transformer used for training. Use only forward() in this case. Defaults to False.
        """
        super(CausalEncoder, self).__init__()
        self.data_processor = data_processor
        # can be useful
        self.dataloader_generator = dataloader_generator

        # Compute num_tokens for source and target
        self.num_tokens_per_channel = self.data_processor.num_tokens_per_channel
        self.num_channels_target = len(self.num_tokens_per_channel)
        assert self.num_channels_target == num_channels_decoder
        self.d_model = d_model
        self.num_tokens_target = self.data_processor.num_tokens

        assert self.num_tokens_target == num_channels_decoder * num_events_decoder

        ######################################################
        # Embeddings
        self.target_positional_embedding = positional_embedding
        positional_embedding_size = self.target_positional_embedding.positional_embedding_size
        
        linear_target_input_size = self.d_model - positional_embedding_size
        self.linear_target = nn.Linear(
            self.data_processor.embedding_size,
            linear_target_input_size
        )

        ########################################################
        # Start of sentence
        self.sos_target = nn.Parameter(torch.randn((1, 1, self.d_model)))

        ######################################################
        self.transformer = LinearTransformerCausalEncoder(
            d_model=d_model,
            n_heads=n_head,
            n_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            recurrent=recurrent
        )
        
        self.label_smoothing = label_smoothing
        self.recurrent = recurrent
        
        ######################################################
        # Output dimension adjustment
        self.pre_softmaxes = nn.ModuleList([nn.Linear(self.d_model, num_tokens_of_channel)
                                            for num_tokens_of_channel in self.num_tokens_per_channel
                                            ]
                                           )
        
    def __repr__(self) -> str:
        return 'CausalEncoder'

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


        # add positional embeddings
        target_seq, h_pe = self.target_positional_embedding(target_seq, i=0, h=h_pe_init, target=target)

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
            mask=torch.ones_like(target),
            label_smoothing=self.label_smoothing
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

