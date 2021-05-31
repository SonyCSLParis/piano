"""
@author: Gaetan Hadjeres
"""
from torch.utils import data
from tqdm import tqdm
from BFT.handlers import EncoderDecoderHandler
from BFT.positional_embeddings import PositionalEmbedding
import importlib
import os
import shutil
from datetime import datetime, time

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
    # os.environ['MASTER_PORT'] = '12356'
    # os.environ['MASTER_PORT'] = '12357'
    os.environ['MASTER_PORT'] = '12358'
    # os.environ['MASTER_PORT'] = '12359'

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
        sos_embedding_dict=config['sos_embedding_dict'])
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
    encoder_decoder = DistributedDataParallel(
        module=encoder_decoder,
        device_ids=[rank],
        output_device=rank,
        #   find_unused_parameters=True
    )

    handler = EncoderDecoderHandler(model=encoder_decoder,
                                    model_dir=model_dir,
                                    dataloader_generator=dataloader_generator)

    if load:
        if overfitted:
            handler.load(early_stopped=False, recurrent=not train)
        else:
            handler.load(early_stopped=True, recurrent=not train)

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

    # load and preprocess
    def generate_transitions(part_1, part_2, timestamp=None, id=None):
        # _edit or _repitched
        begin = dataloader_generator.dataset.process_score(
            f'/home/gaetan/Data/russel/corpus/section{part_1}.mid')
        end = dataloader_generator.dataset.process_score(
            f'/home/gaetan/Data/russel/corpus/section{part_2}.mid')
        middle = dataloader_generator.dataset.process_score(
            f'/home/gaetan/Data/russel/corpus/{part_1}to{part_2}.mid')

        num_events_begin = len(begin['pitch'])
        num_events_middle = len(middle['pitch'])
        num_events_end = len(end['pitch'])

        # concat
        x = {f: begin[f] + middle[f] + end[f] for f in begin}
        
        # x['duration'] = [d / 2 + 1e-3 for d in x['duration']]
        # x['time_shift'] = [d / 2  for d in x['time_shift']]

        x = dataloader_generator.dataset.add_start_end_symbols(
            x,
            start_time=0,
            sequence_size=dataloader_generator.dataset.sequence_size)

        x = dataloader_generator.dataset.tokenize(x)
        x = torch.stack(
            [torch.LongTensor(x[e]) for e in dataloader_generator.features],
            dim=-1)
        # add batch dim
        x = x.unsqueeze(0)

        _, x, _ = data_processor.preprocess(x)
        x = x.repeat(1, 1, 1)

        masked_positions = torch.zeros_like(x)
        # inpainting
        # pitch only
        with open(f'/home/gaetan/Data/russel/corpus/{part_1}to{part_2}.txt', 'r') as f:
            for l in f:
                try:
                    a, b = [int(c) for c in l.split(' ')]
                    assert a <= num_events_middle
                    assert b <= num_events_end
                    masked_positions[:,
                                    num_events_begin + a:num_events_begin + b,
                                    0] = 1
                except ValueError:
                    pass

        x = handler.inpaint_region_optimized(x=x,
                                        masked_positions=masked_positions,
                                        start_event=num_events_begin,
                                        end_event=num_events_begin + num_events_middle,
                                        temperature=1.,
                                        top_p=0.95,
                                        top_k=0)
        
        region = x[:, num_events_begin:num_events_begin + num_events_middle]
        
        # test debug
        # region = x
        
        # mute some notes
        all_regions_mask = region.clone()
        with open(f'/home/gaetan/Data/russel/corpus/{part_1}to{part_2}.txt', 'r') as f:
            last_b = 0
            for k, l in enumerate(f):
                try:
                    a, b = [int(c) for c in l.split(' ')]
                    assert a <= num_events_middle
                    assert b <= num_events_end                
                    all_regions_mask[:,
                                        last_b: a,
                                        1] = 1
                    last_b = b
                except ValueError:
                    pass
            all_regions_mask[:, b:, 1] = 1
        
        # to score
        all_regions = data_processor.postprocess(region.cpu())
        all_regions_mask = data_processor.postprocess(all_regions_mask.cpu())
                
        
        save_folder = f'/home/gaetan/Data/russel/generations/{part_1}to{part_2}/'
        if timestamp is None:
            timestamp = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
        save_folder_with_time = save_folder+ str(timestamp)
        ###############################
        # Saving        
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)
        if not os.path.exists(save_folder_with_time):
            os.mkdir(save_folder_with_time)
        
        # Write scores
        scores = []
        if id is None:
            id = ''
        for k, tensor_score in enumerate(all_regions):
            path_no_extension = f'{save_folder_with_time}/{id}'
            scores.append(
                dataloader_generator.write(tensor_score,
                                                path_no_extension))
        scores = []
        for k, tensor_score in enumerate(all_regions_mask):
            path_no_extension = f'{save_folder_with_time}/{id}_mask'
            scores.append(
                dataloader_generator.write(tensor_score,
                                                path_no_extension))


    def generate_pitches():
        # _edit or _repitched
        # x = dataloader_generator.dataset.process_score(
        #     f'/home/gaetan/Data/russel/offsets_and_velocity.mid')
        # x = dataloader_generator.dataset.process_score(
        #     f'/home/gaetan/Data/databases/Piano/ecomp_piano_dataset/Shen09.MID')
        x = dataloader_generator.dataset.process_score(
            f'/home/gaetan/Data/russel/moz.mid')
        
        # x = dataloader_generator.dataset.process_score(
        #     f'/home/gaetan/Data/databases/Piano/ecomp_piano_dataset/Shamray06.MID')
        
        # x = dataloader_generator.dataset.process_score(
        #     f'/home/gaetan/Data/databases/Piano/ecomp_piano_dataset/MATSUM02.mid')


        x = dataloader_generator.dataset.add_start_end_symbols(
            x,
            start_time=0,
            sequence_size=dataloader_generator.dataset.sequence_size)

        x = dataloader_generator.dataset.tokenize(x)
        x = torch.stack(
            [torch.LongTensor(x[e]) for e in dataloader_generator.features],
            dim=-1)
        # add batch dim
        x = x.unsqueeze(0)

        _, x, _ = data_processor.preprocess(x)
        x = x.repeat(1, 1, 1)
        
        # slice
        x = x[:, :512]
        num_events_x = x.size(1)

        masked_positions = torch.zeros_like(x)
        # inpainting
        # pitch only
        masked_positions[:,
                         :,
                         0:3] = 1

        x = handler.inpaint_region_optimized(x=x,
                                        masked_positions=masked_positions,
                                        start_event=0,
                                        end_event=num_events_x,
                                        temperature=1.,
                                        top_p=0.95,
                                        top_k=0)
        
        # to score
        all_regions = data_processor.postprocess(x.cpu())
        
        save_folder = f'/home/gaetan/Data/russel/pitches_only'
        timestamp = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
        save_folder_with_time = save_folder+ str(timestamp)
        ###############################
        # Saving        
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)
        if not os.path.exists(save_folder_with_time):
            os.mkdir(save_folder_with_time)
        
        # Write scores
        scores = []
        for k, tensor_score in enumerate(all_regions):
            path_no_extension = f'{save_folder_with_time}/{k}'
            scores.append(
                dataloader_generator.write(tensor_score,
                                                path_no_extension))
    
    timestamp = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    for i in range(1, 7):
        for j in range(1, 7):
            print(f'#{i}to#{j}')
            if i==1 and j < 6:
                continue
            for k in tqdm(range(100)):
                generate_transitions(part_1=str(j), part_2=str(i),
                                    timestamp=timestamp, id=k)
    # generate_pitches()

    # TEST
    # metadata_dict = dict(original_sequence=x,
    #                          masked_positions=masked_positions)
    # handler.test_decoder_with_states(
    #     source=x,
    #     metadata_dict=metadata_dict,
    #     temperature=1.,
    #     top_p=0.95
    # )

    # midi_file = 'inputs/br_rhap_format0.mid')
    # midi_file='/home/gaetan/Data/databases/Piano/ecomp_piano_dataset/BENABD02.mid')
    # midi_file='/home/gaetan/Data/databases/Piano/ecomp_piano_dataset/Denisova04.MID')

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
