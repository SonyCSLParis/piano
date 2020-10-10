"""
@author: Gaetan Hadjeres
"""
import importlib
import os
import shutil
from datetime import datetime

import click
import torch

from BFT.data_processors.data_processor import DataProcessor
from BFT.getters import get_dataloader_generator, get_data_processor, get_decoder


@click.command()
@click.option('-t', '--train', is_flag=True)
@click.option('-l', '--load', is_flag=True)
@click.option('-o', '--overfitted', is_flag=True)
@click.option('-c', '--config', type=click.Path(exists=True))
@click.option('-n', '--num_workers', type=int, default=0)
def main(train,
         load,
         overfitted,
         config,
         num_workers
         ):
    # Use all gpus available
    gpu_ids = [int(gpu) for gpu in range(torch.cuda.device_count())]
    print(f'Using GPUs {gpu_ids}')
    if len(gpu_ids) > 0:
        device = 'cuda'
    else:
        device = 'cpu'

    # Load config
    config_path = config
    config_module_name = os.path.splitext(config)[0].replace('/', '.')
    config = importlib.import_module(config_module_name).config

    # compute time stamp
    if config['timestamp'] is not None:
        timestamp = config['timestamp']
    else:
        timestamp = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
        config['timestamp'] = timestamp

    if load:
        model_dir = os.path.dirname(config_path)
    else:
        model_dir = f'models/{config["savename"]}_{timestamp}'

    # === Decoder ====
    dataloader_generator = get_dataloader_generator(
        dataset=config['dataset'],
        dataloader_generator_kwargs=config['dataloader_generator_kwargs']
    )

    data_processor: DataProcessor = get_data_processor(
        dataloader_generator=dataloader_generator,
        data_processor_type=config['data_processor_type'],
        data_processor_kwargs=config['data_processor_kwargs']
    )

    decoder = get_decoder(
        model_dir=model_dir,
        dataloader_generator=dataloader_generator,
        data_processor=data_processor,
        decoder_type=config['decoder_type'],
        decoder_kwargs=config['decoder_kwargs'],
        training_phase=train
        # training_phase = True
    )

    if load:
        if overfitted:
            decoder.load(early_stopped=False)
        else:
            decoder.load(early_stopped=True)
        decoder.to(device)

    if train:
        # Copy .py config file in the save directory before training
        if not load:
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            shutil.copy(config_path, f'{model_dir}/config.py')
        decoder.to(device)
        decoder.train_model(
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

    scores = decoder.generate(temperature=1.,
                              batch_size=3,
                              top_p=0.98,
                              top_k=0)
    scores = decoder.generate_completion(
        num_completions=3,
        temperature=1.,
        top_p=0.95,
        top_k=0,
        midi_file='inputs/Test_X_1.mid')
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
    main()
