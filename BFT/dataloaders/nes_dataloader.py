from torch import random
from torch.utils import data
from BFT.dataloaders.dataloader import DataloaderGenerator
import torch

from DatasetManager.nes.nes_dataset import END_SYMBOL, PAD_SYMBOL, SimpleNESDataset


class NESDataloader(DataloaderGenerator):
    def __init__(self, sequences_size, voices=(0, 1, 2, 3)):
        dataset = SimpleNESDataset(voices=voices,
                                   sequences_size=sequences_size)

        self.sequences_size = sequences_size
        self.features = ['pitch', 'velocity', 'duration', 'time_shift']
        self.num_channels = 4
        super(NESDataloader, self).__init__(dataset=dataset)

    def write(self, x, path):
        # TODO
        raise NotImplementedError

    def get_elapsed_time(self, x):
        """
        x is (batch_size, num_events, num_channels)
        """
        assert 'time_shift' in self.features

        timeshift_indices = x[:, :, self.features.index('time_shift')]
        # convert timeshift indices to their actual duration:
        y = self.dataset.timeshift_indices_to_elapsed_time(
            timeshift_indices, smallest_time_shift=0.02)
        return y.cumsum(dim=-1)

    def dataloaders(self,
                    batch_size,
                    num_workers=0,
                    shuffle_train=True,
                    shuffle_val=False):
        dataloaders = self.dataset.data_loaders(batch_size,
                                                shuffle_train=shuffle_train,
                                                shuffle_val=shuffle_val,
                                                num_workers=num_workers)

        def _build_dataloader(dataloader):
            for data in dataloader:
                x = torch.stack([data[e] for e in self.features], dim=-1)
                ret = {'x': x}
                yield ret

        dataloaders = [
            _build_dataloader(dataloaders[split])
            for split in ['train', 'val', 'test']
        ]
        return dataloaders