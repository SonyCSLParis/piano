from BFT.positional_embeddings.sinusoidal_progress_bar_embedding import SinusoidalProgressBarEmbedding
from BFT.utils import cuda_variable
from BFT.getters import get_data_processor, get_dataloader_generator
import unittest
import torch
import os
import torch.distributed as dist
from BFT.positional_embeddings import SinusoidalElapsedTimeEmbedding
import random

random.seed(a=44, version=2)

def max_relative_error(a, b, eps=1e-6):
    return torch.abs((a - b) / (torch.abs(a) + eps)).max().item()


class TestSinusoidalElapsedTimeEmbedding(unittest.TestCase):
    def test_forward_step():
        dataloader_generator = get_dataloader_generator(
            dataset='piano',
            dataloader_generator_kwargs=dict(sequences_size=1024,
                                             transformations={
                                                 'time_dilation': True,
                                                 'velocity_shift': True,
                                                 'transposition': True
                                             },
                                             pad_before=True))

        data_processor = get_data_processor(
            dataloader_generator=dataloader_generator,
            data_processor_type='piano_prefix',
            data_processor_kwargs=dict(embedding_size=64,
                                       num_events_before=256,
                                       num_events_after=256))

        (generator_train, generator_val,
         _) = dataloader_generator.dataloaders(batch_size=1,
                                               num_workers=0,
                                               shuffle_val=True)
        x = next(generator_val)['x']

        x, metadata_dict = data_processor.preprocess(x)
        pe = SinusoidalElapsedTimeEmbedding(
            positional_embedding_size=32,
            num_channels=4,
            dataloader_generator=dataloader_generator)

        x_embedded = torch.randn(x.size(0), x.size(1) * x.size(2), 32)
        output_parallel, h_parallel = pe.forward(
            x_embed=x_embedded,
            i=0,
            h=None,
            metadata_dict=metadata_dict
        )
        
        # recurrent
        # for i in range(x.size(1)):
        #     pe.forward_step()




if __name__ == "__main__":
    # unittest.main()
    # === Init process group
    os.environ['MASTER_ADDR'] = 'localhost'
    # os.environ['MASTER_PORT'] = '12355'
    # os.environ['MASTER_PORT'] = '12356'
    # os.environ['MASTER_PORT'] = '12357'
    os.environ['MASTER_PORT'] = '12358'
    rank = 0
    dist.init_process_group(backend='nccl', world_size=1, rank=rank)
    torch.cuda.set_device(rank)
    device = f'cuda:{rank}'
    dataloader_generator = get_dataloader_generator(
        dataset='piano',
        dataloader_generator_kwargs=dict(sequences_size=1024,
                                            transformations={
                                                'time_dilation': True,
                                                'velocity_shift': True,
                                                'transposition': True
                                            },
                                            pad_before=True))

    data_processor = get_data_processor(
        dataloader_generator=dataloader_generator,
        data_processor_type='piano_prefix',
        data_processor_kwargs=dict(embedding_size=64,
                                    num_events_before=256,
                                    num_events_after=256))
    data_processor.to(device)

    (generator_train, generator_val,
        _) = dataloader_generator.dataloaders(batch_size=1,
                                            num_workers=0,
                                            shuffle_val=True)
    x = next(generator_val)['x']

    x, metadata_dict = data_processor.preprocess(x)
    print(f'Placeholder duration: {metadata_dict["placeholder_duration"]}')
    
    # pe = SinusoidalElapsedTimeEmbedding(
    #     positional_embedding_size=32,
    #     num_channels=4,
    #     dropout=0.,
    #     dataloader_generator=dataloader_generator,
    #     mask_positions=False
    #     )
    
    pe = SinusoidalProgressBarEmbedding(
        positional_embedding_size=32,
        num_channels=4,
        dropout=0.,
        dataloader_generator=dataloader_generator,
        )
    
    
    pe.to(device)

    x_embedded = cuda_variable(torch.randn(x.size(0), x.size(1) * x.size(2), 32))
    output_parallel, h_parallel = pe.forward(
        x_embed=x_embedded,
        i=0,
        h=None,
        metadata_dict=metadata_dict
    )
    pos_embedding_parallel = output_parallel[:, :, 32:]
    
    
    h = None
    pos_embedding_recurrent = []
    for event_index in range(x.size(1)):
        for channel_index in range(4):
            i = event_index * 4 + channel_index
            metadata_dict['original_token'] = x[:, event_index, channel_index]
            output, h = pe.forward_step(
                x=x_embedded[:, i],
                i=i,
                h=h,
                metadata_dict=metadata_dict
            )
            pos_embedding_recurrent.append(output[:, 32:])
            
            # if i < 20:
            #     print(f'{i} : {h[0]}')
    pos_embedding_recurrent = torch.stack(pos_embedding_recurrent, dim=1)
    h_recurrent = h
    
    
    
    diff = (torch.abs(pos_embedding_recurrent - pos_embedding_parallel) > 0.001).sum()
    for i in range(pos_embedding_recurrent.size(1)):
        diff_i = torch.abs(pos_embedding_recurrent[0, i] - pos_embedding_parallel[0, i]).sum().item()
        if diff_i > 1e-3:
            print(f'{i}: {diff_i}')
            # break
    
    
