from torch.distributed.distributed_c10d import get_world_size
from BFT.dataloaders.dataloader import DataloaderGenerator
from BFT.utils import all_reduce_scalar, dict_pretty_print, display_monitored_quantities, is_main_process, to_numpy, top_k_top_p_filtering
import torch
import os
from tqdm import tqdm
from itertools import islice
from datetime import datetime
import numpy as np
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist


class Handler:
    def __init__(self, model: DistributedDataParallel,
                 model_dir: str,
                 dataloader_generator: DataloaderGenerator) -> None:
        self.model = model
        self.model_dir = model_dir
        self.dataloader_generator = dataloader_generator        
        # optim
        self.optimizer = None
        self.scheduler = None
        
    #     self.init_weights()
        
    # def init_weights(self):
    #     for name, param in self.model.named_parameters():
    #         if name.split('.')[-1] == 'weight':
    #             if ('embed' not in name) and ('norm' not in name) and ('gating' not in name):
    #                 torch.nn.init.kaiming_normal_(param, nonlinearity='relu')
    #                 param.data = param.data * 0.01
                    
            

    def init_optimizers(self, lr=1e-3):
        # self.optimizer = torch.optim.Adam(list(self.parameters()), lr=lr)
        self.optimizer = torch.optim.AdamW(list(self.parameters()),
                                           lr=lr,
                                           weight_decay=1e-3)

    # ==== Wrappers
    def forward(self, target, metadata_dict, h_pe_init=None):
        return self.model.forward(target, metadata_dict, h_pe_init=h_pe_init)
    
    def forward_step(self, target, metadata_dict, state, i, h_pe):
        return self.model.module.forward_step(target, metadata_dict, state, i, h_pe)

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def parameters(self):
        return self.model.parameters()
    
    # expose useful attributes for generation
    @property
    def recurrent(self):
        return self.model.module.recurrent
    
    @property
    def num_tokens_per_channel_target(self):
        return self.model.module.data_processor.num_tokens_per_channel
    
    @property
    def num_channels_target(self):
        return self.model.module.data_processor.num_channels

    @property
    def data_processor(self):
        return self.model.module.data_processor

    # ==== Save and Load methods
    def __repr__(self):
        return self.model.module.__repr__()

    def save(self, early_stopped):
        # Only save on process 0
        if dist.get_rank() == 0:
            # This saves also the encoder
            if early_stopped:
                model_dir = f'{self.model_dir}/early_stopped'
            else:
                model_dir = f'{self.model_dir}/overfitted'
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            torch.save(self.model.state_dict(), f'{model_dir}/model')
        dist.barrier()

    def load(self, early_stopped):
        map_location = {'cuda:0': f'cuda:{dist.get_rank()}'}
        print(f'Loading models {self.__repr__()}')
        if early_stopped:
            print('Load early stopped model')
            model_dir = f'{self.model_dir}/early_stopped'
        else:
            print('Load over-fitted model')
            model_dir = f'{self.model_dir}/overfitted'
            
        state_dict = torch.load(f'{model_dir}/model',
                                map_location=map_location)
        
        # TODO, Remove?!
        # # copy transformer_with_states during inference
        
        # transformer_with_states_dict = {}
        # for k, v in state_dict.items():
        #     if 'transformer' in k:
        #         new_key = k.replace('decoder.transformer', 'decoder.transformer_with_states')
        #         transformer_with_states_dict[new_key] = v
        # state_dict.update(transformer_with_states_dict)

        self.model.load_state_dict(
            state_dict=state_dict
            )

    def plot(self, epoch_id, monitored_quantities_train,
             monitored_quantities_val) -> None:
        if is_main_process():
            for k, v in monitored_quantities_train.items():
                self.writer.add_scalar(f'{k}/train', v, epoch_id)
            for k, v in monitored_quantities_val.items():
                self.writer.add_scalar(f'{k}/val', v, epoch_id)


    def train_model(self,
                    batch_size,
                    num_batches=None,
                    num_epochs=10,
                    lr=1e-3,
                    plot=False,
                    num_workers=0,
                    **kwargs):
        if plot and is_main_process():
            self.writer = SummaryWriter(f'{self.model_dir}')

        best_val = 1e8
        self.init_optimizers(lr=lr)
        for epoch_id in range(num_epochs):
            (generator_train, generator_val,
             generator_test) = self.dataloader_generator.dataloaders(
                 batch_size=batch_size,
                 num_workers=num_workers,
                 shuffle_val=True)

            monitored_quantities_train = self.epoch(
                data_loader=generator_train,
                train=True,
                num_batches=num_batches,
            )
            del generator_train

            with torch.no_grad():
                monitored_quantities_val = self.epoch(
                    data_loader=generator_val,
                    train=False,
                    num_batches=num_batches //
                    2 if num_batches is not None else None,
                )
                del generator_val
            valid_loss = monitored_quantities_val['loss']
            # self.scheduler.step(monitored_quantities_val["loss"])

            display_monitored_quantities(
                epoch_id=epoch_id,
                monitored_quantities_train=monitored_quantities_train,
                monitored_quantities_val=monitored_quantities_val)

            self.save(early_stopped=False)

            if valid_loss < best_val:
                self.save(early_stopped=True)
                best_val = valid_loss

            if plot:
                self.plot(epoch_id, monitored_quantities_train,
                          monitored_quantities_val)
                
    def epoch(self,
        data_loader,
        train=True,
        num_batches=None):
        raise NotImplementedError
