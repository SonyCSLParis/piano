from .source_target_data_processor import SourceTargetDataProcessor
from .data_processor import DataProcessor
import torch
import random
from torch import nn
from BFT.utils import cuda_variable


class PianoDataProcessor(DataProcessor):
    def __init__(self,
                 embedding_size,
                 num_events,
                 num_tokens_per_channel,
                 add_mask_token=False): 
        super(PianoDataProcessor,
              self).__init__(embedding_size=embedding_size,
                             num_events=num_events,
                             num_tokens_per_channel=num_tokens_per_channel,
                             add_mask_token=add_mask_token)

    def postprocess(self, reconstruction, original=None):
        if original is not None:
            # Just concatenate along batch dimension original and reconstruction
            original = original.long()
            tensor_score = torch.cat(
                [original[0].unsqueeze(0),
                 reconstruction.cpu()], dim=0)
            #  Add a first empty dimension as everything will be written in one score
            return tensor_score.unsqueeze(0)
        else:
            return reconstruction

class MaskedPianoSourceTargetDataProcessor(SourceTargetDataProcessor):
    def __init__(self, embedding_size, num_events, num_tokens_per_channel):
        
        
        encoder_data_processor = PianoDataProcessor(
            embedding_size=embedding_size,
            num_events=num_events,
            num_tokens_per_channel=num_tokens_per_channel,
            add_mask_token=True)

        decoder_data_processor = PianoDataProcessor(
            embedding_size=embedding_size,
            num_events=num_events,
            num_tokens_per_channel=num_tokens_per_channel,
            add_mask_token=False)
        
        super(MaskedPianoSourceTargetDataProcessor, self).__init__(
            encoder_data_processor=encoder_data_processor,
            decoder_data_processor=decoder_data_processor
            )
        
        # (num_channels, ) LongTensor
        self.mask_symbols = nn.Parameter(
            torch.LongTensor(self.encoder_data_processor.num_tokens_per_channel),
            requires_grad=False
        )

    def _mask_source(self, x, masked_positions=None):
        """Add a MASK symbol

        Args:
            x (batch_size, num_events, num_channels) LongTensor: non-embeded source input
            masked_positions ([type], optional): if None, masked_positions are sampled. Defaults to None.

        Returns:
            [type]: masked_x
        """
        if masked_positions is None:
            p = random.random() * 0.5
            masked_positions = (torch.rand_like(x.float()) < p).long()
        
        batch_size, num_events, num_channels = x.size()
        mask_symbols = self.mask_symbols.unsqueeze(0).unsqueeze(0).repeat(
            batch_size, num_events, 1
        )
                
        return (
            x * (1 - masked_positions) + masked_positions * mask_symbols
        )

    def preprocess(self, x):
        """
        :param x: ? -> tuple source, target of size (batch_size, num_events_source, num_channels_source)
        (batch_size, num_events_target, num_channels_target)
        :return:
        """
        source = cuda_variable(x.long())
        target = cuda_variable(x.long())
        source = self._mask_source(source)
        return source, target
