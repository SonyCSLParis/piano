from BFT.positional_embeddings import ChannelEmbeddings, BasePositionalEmbedding, PositionalEmbedding, SinusoidalElapsedTimeEmbedding, SinusoidalPositionalEmbedding
from BFT.data_processors import BachDataProcessor, MaskedPianoSourceTargetDataProcessor, PianoDataProcessor
from BFT.dataloaders import BachDataloaderGenerator, PianoDataloaderGenerator

from BFT.decoders.linear_transformer_decoder import CausalEncoder
from BFT.decoders.linear_transformer_encoder_decoder import EncoderDecoder


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


def get_source_target_data_processor(dataloader_generator, data_processor_type,
                          data_processor_kwargs):

    if data_processor_type == 'bach':
        # TODO
        raise NotImplementedError
    elif data_processor_type == 'masked_piano':
        num_events = dataloader_generator.dataset.sequence_size
        value2index = dataloader_generator.dataset.value2index
        num_tokens_per_channel = [
            len(value2index[feature])
            for feature in dataloader_generator.features
        ]
        data_processor = MaskedPianoSourceTargetDataProcessor(
            embedding_size=data_processor_kwargs['embedding_size'],
            num_events=num_events,
            num_tokens_per_channel=num_tokens_per_channel)
    else:
        raise NotImplementedError

    return data_processor


def get_positional_embedding(dataloader_generator,
                             positional_embedding_dict) -> PositionalEmbedding:
    base_positional_embedding_list = []
    for pe_name, pe_kwargs in positional_embedding_dict.items():
        if pe_name == 'sinusoidal_embedding':
            num_tokens_max = (dataloader_generator.dataset.sequence_size *
                              dataloader_generator.num_channels)
            base_pe: BasePositionalEmbedding = SinusoidalPositionalEmbedding(
                positional_embedding_size=pe_kwargs[
                    'positional_embedding_size'],
                num_tokens_max=num_tokens_max,
                num_channels=pe_kwargs['num_channels'],
                dropout=pe_kwargs['dropout'])
        elif pe_name == 'channel_embedding':
            base_pe = ChannelEmbeddings(**pe_kwargs)
        elif pe_name == 'sinusoidal_elapsed_time_embedding':
            base_pe: BasePositionalEmbedding = SinusoidalElapsedTimeEmbedding(
                dataloader_generator=dataloader_generator, **pe_kwargs)
        else:
            raise NotImplementedError
        base_positional_embedding_list.append(base_pe)

    return PositionalEmbedding(
        base_positional_embedding_list=base_positional_embedding_list)


# todo write Decoder base class
def get_decoder(data_processor, dataloader_generator, positional_embedding,
                decoder_type, decoder_kwargs, training_phase):
    num_channels_decoder = data_processor.num_channels
    num_events_decoder = data_processor.num_events

    if decoder_type == 'linear_transformer':
        decoder = CausalEncoder(
            data_processor=data_processor,
            dataloader_generator=dataloader_generator,
            positional_embedding=positional_embedding,
            d_model=decoder_kwargs['d_model'],
            num_decoder_layers=decoder_kwargs['num_decoder_layers'],
            n_head=decoder_kwargs['n_head'],
            dim_feedforward=decoder_kwargs['dim_feedforward'],
            dropout=decoder_kwargs['dropout'],
            num_channels_decoder=num_channels_decoder,
            num_events_decoder=num_events_decoder,
            label_smoothing=decoder_kwargs['label_smoothing'],
            recurrent=not training_phase)
    else:
        raise NotImplementedError

    return decoder


def get_encoder_decoder(data_processor, 
                        dataloader_generator,
                        positional_embedding_source,
                        positional_embedding_target,
                        encoder_decoder_type,
                        encoder_decoder_kwargs,
                        training_phase):

    if encoder_decoder_type == 'linear_transformer':
        decoder = EncoderDecoder(
            data_processor=data_processor,
            dataloader_generator=dataloader_generator,
            positional_embedding_target=positional_embedding_target,
            positional_embedding_source=positional_embedding_source,
            d_model_encoder=encoder_decoder_kwargs['d_model_encoder'],
            d_model_decoder=encoder_decoder_kwargs['d_model_decoder'],
            num_layers_decoder=encoder_decoder_kwargs['num_layers_decoder'],
            num_layers_encoder=encoder_decoder_kwargs['num_layers_encoder'],
            n_head_encoder=encoder_decoder_kwargs['n_head_encoder'],
            n_head_decoder=encoder_decoder_kwargs['n_head_decoder'],
            dim_feedforward_encoder=encoder_decoder_kwargs['dim_feedforward_encoder'],
            dim_feedforward_decoder=encoder_decoder_kwargs['dim_feedforward_decoder'],
            dropout=encoder_decoder_kwargs['dropout'],
            num_channels_target=data_processor.num_channels_target,
            num_channels_source=data_processor.num_channels_source,
            num_events_target=data_processor.num_events_target,
            num_events_source=data_processor.num_events_source,
            label_smoothing=encoder_decoder_kwargs['label_smoothing'],
            recurrent=not training_phase)
    else:
        raise NotImplementedError

    return decoder