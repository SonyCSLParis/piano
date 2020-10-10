import torch
from DatasetManager.piano.piano_helper import MaestroIteratorGenerator
from DatasetManager.piano.piano_midi_dataset import PianoMidiDataset

from BFT.dataloaders.dataloader import DataloaderGenerator


class PianoDataloaderGenerator(DataloaderGenerator):
    def __init__(self,
                 sequences_size,
                 transformations,
                 *args, **kwargs):
        """
        :param num_tokens_per_block:
        :param num_blocks_left:
        :param num_blocks_right:
        :param num_negative_samples:
        :param negative_sampling_method:
        :param args:
        :param kwargs:
        """

        legacy = True

        corpus_it_gen = MaestroIteratorGenerator(
            composers_filter=[],
            num_elements=None
        )
        # Positive dataset
        dataset: PianoMidiDataset = PianoMidiDataset(
            corpus_it_gen=corpus_it_gen,
            sequence_size=sequences_size,
            smallest_time_shift=0.02,
            max_transposition=6,
            time_dilation_factor=0.1,
            velocity_shift=20,
            transformations=transformations,
            different_time_table_ts_duration=not legacy
        )
        super(PianoDataloaderGenerator, self).__init__(dataset=dataset)
        self.features = ['pitch', 'velocity', 'duration', 'time_shift']
        self.num_channels = 4

    def dataloaders(self, batch_size, num_workers=0, shuffle_train=True,
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
            for split
            in ['train', 'val', 'test']
        ]
        return dataloaders

    def write(self, x, path):
        """
        :param x: (batch_size, num_events, num_channels)
        :return: list of music21.Score
        """
        # TODO add back when fixing signatures for write
        # xs = self.dataset.interleave_silences_batch(x, self.features)
        xs = x
        # values
        sequences = {feature: xs[:, feature_index] for feature_index, feature in
                     enumerate(self.features)}
        score = self.dataset.tensor_to_score(sequences, fill_features=None)
        score.write(f'{path}.mid')
        return torch
