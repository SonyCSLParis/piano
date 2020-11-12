"""
@author: Gaetan Hadjeres
"""
from BFT.handlers import EncoderDecoderHandler
from BFT.positional_embeddings import PositionalEmbedding
import importlib
import os
import shutil
from datetime import datetime

import click
import torch

import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from BFT.getters import get_dataloader_generator, get_sos_embedding, get_source_target_data_processor, get_encoder_decoder, get_positional_embedding


@click.command()
@click.option('-t', '--train', is_flag=True)
@click.option('-l', '--load', is_flag=True)
@click.option('-o', '--overfitted', is_flag=True)
@click.option('-c', '--config', type=click.Path(exists=True))
@click.option('-n', '--num_workers', type=int, default=0)
def launcher(train, load, overfitted, config, num_workers):

    # === Set shared parameters

    # always use the maximum number of available GPUs for training
    if train:
        world_size = torch.cuda.device_count()
        assert world_size > 0
    else:
        # only use 1 GPU for inference
        world_size = 1

    # Load config as dict
    config_path = config
    config_module_name = os.path.splitext(config)[0].replace('/', '.')
    config = importlib.import_module(config_module_name).config

    # Compute time stamp
    if config['timestamp'] is not None:
        timestamp = config['timestamp']
    else:
        timestamp = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
        config['timestamp'] = timestamp

    # Create or retreive model_dir
    if load:
        model_dir = os.path.dirname(config_path)
    else:
        model_dir = f'models/{config["savename"]}_{timestamp}'

    # Copy .py config file in the save directory before training
    if not load:
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        shutil.copy(config_path, f'{model_dir}/config.py')

    print(f'Using {world_size} GPUs')
    mp.spawn(main,
             args=(train, load, overfitted, config, num_workers, world_size,
                   model_dir),
             nprocs=world_size,
             join=True)


def main(rank, train, load, overfitted, config, num_workers, world_size,
         model_dir):
    # === Init process group
    os.environ['MASTER_ADDR'] = 'localhost'
    # os.environ['MASTER_PORT'] = '12355'
    os.environ['MASTER_PORT'] = '12356'
    # os.environ['MASTER_PORT'] = '12357'
    dist.init_process_group(backend='nccl', world_size=world_size, rank=rank)
    torch.cuda.set_device(rank)
    device = f'cuda:{rank}'

    # === Decoder ====
    # dataloader generator
    dataloader_generator = get_dataloader_generator(
        dataset=config['dataset'],
        dataloader_generator_kwargs=config['dataloader_generator_kwargs'])

    # data processor
    data_processor = get_source_target_data_processor(
        dataloader_generator=dataloader_generator,
        data_processor_type=config['data_processor_type'],
        data_processor_kwargs=config['data_processor_kwargs'])

    # positional embedding
    positional_embedding_source: PositionalEmbedding = get_positional_embedding(
        dataloader_generator=dataloader_generator,
        positional_embedding_dict=config['positional_embedding_source_dict'])
    positional_embedding_target: PositionalEmbedding = get_positional_embedding(
        dataloader_generator=dataloader_generator,
        positional_embedding_dict=config['positional_embedding_target_dict'])
    # sos embedding
    sos_embedding = get_sos_embedding(
        dataloader_generator=dataloader_generator,
        sos_embedding_dict=config['sos_embedding_dict']
    )
    encoder_decoder = get_encoder_decoder(
        data_processor=data_processor,
        dataloader_generator=dataloader_generator,
        positional_embedding_source=positional_embedding_source,
        positional_embedding_target=positional_embedding_target,
        sos_embedding=sos_embedding,
        encoder_decoder_type=config['encoder_decoder_type'],
        encoder_decoder_kwargs=config['encoder_decoder_kwargs'],
        training_phase=train)

    encoder_decoder.to(device)
    encoder_decoder = DistributedDataParallel(module=encoder_decoder,
                                      device_ids=[rank],
                                      output_device=rank,
                                    #   find_unused_parameters=True
                                      )

    handler = EncoderDecoderHandler(model=encoder_decoder,
                                     model_dir=model_dir,
                                     dataloader_generator=dataloader_generator
                                     )

    if load:
        if overfitted:
            handler.load(early_stopped=False)
        else:
            handler.load(early_stopped=True)

    if train:
        handler.train_model(
            batch_size=config['batch_size'],
            num_batches=config['num_batches'],
            num_epochs=config['num_epochs'],
            lr=config['lr'],
            plot=True,
            num_workers=num_workers,
        )
        exit()

    # scores = decoder.generate_non_recurrent(
    #     temperature=1.,
    #     batch_size=3,
    #     top_p=0.9,
    #     top_k=0)

    scores = handler.generate_completion(num_completions=3,
                                         temperature=1.,
                                         top_p=0.9,
                                         top_k=0,
                                         midi_file='inputs/Test_X_1.mid')
    scores = handler.generate(temperature=1.,
                                      batch_size=3,
                                      top_p=0.8,
                                      top_k=0)
    # midi_file = 'inputs/br_rhap_format0.mid')
    # midi_file='/home/gaetan/Data/databases/Piano/ecomp_piano_dataset/BENABD02.mid')
    # midi_file='/home/gaetan/Data/databases/Piano/ecomp_piano_dataset/Denisova04.MID')

    # for score in scores:
    #     score.show()

    # scores = decoder.generate_reharmonisation(
    #     temperature=1.0,
    #     num_reharmonisations=3,
    #     top_k=0,
    #     top_p=0.8
    # )
    # for score in scores:
    #     score.show()

    # # Body code: need do check cluster before adding values
    # start_cluster = 7
    # end_cluster = 21
    # pad_cluster = 12
    #
    # start_codes = [pad_cluster] * 5 + [start_cluster]
    # end_codes = [end_cluster] + [pad_cluster] * 5
    # body_codes = [1] * 16   # put what u want here
    # scores = decoder.generate_alla_mano(
    #     start_codes=start_codes,
    #     end_codes=end_codes,
    #     body_codes=body_codes,
    #     temperature=1.2,
    # )
    # for score in scores:
    #     score.show()


if __name__ == '__main__':
    launcher()
