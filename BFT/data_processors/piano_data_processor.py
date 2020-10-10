import numpy as np
import torch

from BFT.data_processors.data_processor import DataProcessor


class PianoDataProcessor(DataProcessor):
    def __init__(self, embedding_size, num_events, num_tokens_per_channel):
        super(PianoDataProcessor, self).__init__(embedding_size=embedding_size,
                                                 num_events=num_events,
                                                 num_tokens_per_channel=num_tokens_per_channel
                                                 )

    def postprocess(self, reconstruction, original=None):
        if original is not None:
            # Just concatenate along batch dimension original and reconstruction
            original = original.long()
            tensor_score = torch.cat([
                original[0].unsqueeze(0),
                reconstruction.cpu()
            ], dim=0)
            #  Add a first empty dimension as everything will be written in one score
            return tensor_score.unsqueeze(0)
        else:
            return reconstruction