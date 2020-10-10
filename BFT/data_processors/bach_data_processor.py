import torch

from BFT.data_processors.data_processor import DataProcessor
from BFT.utils import to_numpy


class BachDataProcessor(DataProcessor):
    def __init__(self, embedding_size, num_events, num_tokens_per_channel):
        super(BachDataProcessor, self).__init__(embedding_size=embedding_size,
                                                num_events=num_events,
                                                num_tokens_per_channel=num_tokens_per_channel
                                                )

    def postprocess(self, reconstruction, original=None):
        if original is not None:
            tensor_score = torch.cat([
                original.long(),
                reconstruction.cpu()
            ], dim=1)
        else:
            tensor_score = reconstruction
        tensor_score = to_numpy(tensor_score)
        return tensor_score
