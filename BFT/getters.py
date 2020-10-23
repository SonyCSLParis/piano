from BFT.data_processors.bach_data_processor import BachDataProcessor
from BFT.data_processors.piano_data_processor import PianoDataProcessor
from BFT.dataloaders.bach_dataloader import BachDataloaderGenerator
import numpy as np

from BFT.dataloaders.piano_dataloader import PianoDataloaderGenerator
from BFT.decoders.linear_transformer_decoder import LinearTransformerDecoder


def get_dataloader_generator(dataset, dataloader_generator_kwargs):
    if dataset.lower() == 'bach':
        return BachDataloaderGenerator(
            sequences_size=dataloader_generator_kwargs['sequences_size'])
    elif dataset.lower() == 'piano':
        return PianoDataloaderGenerator(
            sequences_size=dataloader_generator_kwargs['sequences_size'],
            transformations=dataloader_generator_kwargs['transformations'])
    else:
        raise NotImplementedError


def get_data_processor(dataloader_generator, data_processor_type,
                       data_processor_kwargs):
    if data_processor_type == 'bach':
        # compute num_events num_tokens_per_channel
        dataset = dataloader_generator.dataset
        num_events = dataset.sequences_size * dataset.subdivision
        num_tokens_per_channel = [len(d) for d in dataset.index2note_dicts]
        data_processor = BachDataProcessor(
            embedding_size=data_processor_kwargs['embedding_size'],
            num_events=num_events,
            num_tokens_per_channel=num_tokens_per_channel)
    elif data_processor_type == 'piano':
        num_events = dataloader_generator.dataset.sequence_size
        value2index = dataloader_generator.dataset.value2index
        num_tokens_per_channel = [
            len(value2index[feature])
            for feature in dataloader_generator.features
        ]
        data_processor = PianoDataProcessor(
            embedding_size=data_processor_kwargs['embedding_size'],
            num_events=num_events,
            num_tokens_per_channel=num_tokens_per_channel)
    else:
        raise NotImplementedError

    return data_processor


# todo write Decoder base class
def get_decoder(data_processor, 
                dataloader_generator,
                decoder_type,
                decoder_kwargs, training_phase):
    num_channels_decoder = data_processor.num_channels
    num_events_decoder = data_processor.num_events
    # TODO add get positional embedding
    
    if decoder_type == 'linear_transformer':
        decoder = LinearTransformerDecoder(
            data_processor=data_processor,
            dataloader_generator=dataloader_generator,
            d_model=decoder_kwargs['d_model'],
            num_decoder_layers=decoder_kwargs['num_decoder_layers'],
            n_head=decoder_kwargs['n_head'],
            dim_feedforward=decoder_kwargs['dim_feedforward'],
            dropout=decoder_kwargs['dropout'],
            positional_embedding_size=decoder_kwargs[
                'positional_embedding_size'],
            num_channels_decoder=num_channels_decoder,
            num_events_decoder=num_events_decoder,
            label_smoothing=decoder_kwargs['label_smoothing'],
            recurrent=not training_phase)
    else:
        raise NotImplementedError

    return decoder
