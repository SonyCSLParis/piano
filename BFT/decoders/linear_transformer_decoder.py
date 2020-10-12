from datetime import datetime
from itertools import islice

from fast_transformers.masking import TriangularCausalMask
from torch import nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from BFT.data_processors.data_processor import DataProcessor
from BFT.dataloaders.dataloader import DataloaderGenerator
from BFT.positional_embeddings.channel_embeddings import ChannelEmbeddings
from BFT.positional_embeddings.learnt_embeddings import LearntEmbeddings
from BFT.positional_embeddings.recurrent_positional_embedding import RecurrentPositionalEmbedding
from BFT.positional_embeddings.sinusoidal_positional_embedding import SinusoidalPositionalEmbedding, \
    SinusoidalPositionalEmbeddingChannels
from BFT.transformers.linear_transformer import LinearTransformer
from BFT.utils import flatten, categorical_crossentropy, dict_pretty_print, top_k_top_p_filtering, \
    to_numpy, cuda_variable
import os
import torch


class LinearTransformerDecoder(nn.Module):
    def __init__(self,
                 model_dir,
                 dataloader_generator: DataloaderGenerator,
                 data_processor: DataProcessor,
                 d_model,
                 num_decoder_layers,
                 n_head,
                 dim_feedforward,
                 positional_embedding_size,
                 num_channels_decoder,
                 num_events_decoder,
                 dropout,
                 # TODO pass arguments
                 recurrent=False):
        """
        TODO
        :param model_dir:
        :param dataloader_generator:
        :param data_processor:
        :param d_model:
        :param num_decoder_layers:
        :param n_head:
        :param dim_feedforward:
        :param positional_embedding_size:
        :param num_channels_decoder:
        :param num_events_decoder:
        :param dropout:
        """
        super(LinearTransformerDecoder, self).__init__()
        self.model_dir = model_dir
        self.recurrent = recurrent

        self.dataloader_generator = dataloader_generator
        self.data_processor = data_processor

        # Compute num_tokens for source and target
        self.num_tokens_per_channel = self.data_processor.num_tokens_per_channel
        self.num_channels_target = len(self.num_tokens_per_channel)
        assert self.num_channels_target == num_channels_decoder
        self.d_model = d_model
        self.num_tokens_target = self.data_processor.num_tokens

        assert self.num_tokens_target == num_channels_decoder * num_events_decoder

        ######################################################
        # Embeddings
        # self.target_positional_embedding = RecurrentPositionalEmbedding(
        #     positional_embedding_size=positional_embedding_size,
        #     num_channels=num_channels_decoder * 4
        # )

        # self.target_positional_embedding = ChannelEmbeddings(
        #     positional_embedding_size=positional_embedding_size,
        #     num_channels=num_channels_decoder
        # )

        # for Bach
        # self.target_positional_embedding = LearntEmbeddings(
        #     positional_embedding_size=positional_embedding_size,
        #     num_channels=num_channels_decoder * 4,
        #     num_tokens_max=1024 # TODO hard coded
        # )

        # self.target_positional_embedding = LearntEmbeddings(
        #     positional_embedding_size=positional_embedding_size,
        #     num_channels=num_channels_decoder,
        #     num_tokens_max=1024 * 4 # TODO hard coded
        # # num_tokens_max = 1024  # TODO hard coded
        # )
        # self.target_positional_embedding = SinusoidalPositionalEmbedding(
        #     positional_embedding_size=positional_embedding_size,
        #     num_channels=num_channels_decoder,
        #     num_tokens_max=1024 * 4, # TODO hard coded
        #     dropout=0.1
        # )

        # LAST USED
        self.target_positional_embedding = SinusoidalPositionalEmbeddingChannels(
            positional_embedding_size=positional_embedding_size,
            num_channels=num_channels_decoder,
            num_tokens_max=1024 * 4,  # TODO hard coded
            dropout=0.1
        )

        linear_target_input_size = self.d_model - positional_embedding_size
        self.linear_target = nn.Linear(
            self.data_processor.embedding_size,
            linear_target_input_size
        )

        ########################################################
        # Start of sentence
        self.sos_target = nn.Parameter(torch.randn((1, 1, self.d_model)))

        ######################################################
        self.transformer = LinearTransformer(
            d_model=d_model,
            n_heads=n_head,
            n_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            recurrent=recurrent
        )

        ######################################################
        # Output dimension adjustment
        self.pre_softmaxes = nn.ModuleList([nn.Linear(self.d_model, num_tokens_of_channel)
                                            for num_tokens_of_channel in self.num_tokens_per_channel
                                            ]
                                           )

        ######################################################
        # optim
        self.optimizer = None
        self.scheduler = None

    def init_optimizers(self, lr=1e-3):
        self.optimizer = torch.optim.Adam(
            list(self.parameters())
            ,
            lr=lr
        )

    def __repr__(self):
        return 'DecoderRelative'

    def save(self, early_stopped):
        # This saves also the encoder
        if early_stopped:
            model_dir = f'{self.model_dir}/early_stopped'
        else:
            model_dir = f'{self.model_dir}/overfitted'
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        torch.save(self.state_dict(), f'{model_dir}/decoder')

    def load(self, early_stopped):
        print(f'Loading models {self.__repr__()}')
        if early_stopped:
            print('Load early stopped model')
            model_dir = f'{self.model_dir}/early_stopped'
        else:
            print('Load over-fitted model')
            model_dir = f'{self.model_dir}/overfitted'
            self.load_state_dict(
                torch.load(f'{model_dir}/decoder',
                           map_location=lambda storage, loc: storage
                           )
            )

    def forward(self, target, h_pe_init=None):
        """
        :param target: sequence of tokens (batch_size, num_events, num_channels)
        :return:
        """
        batch_size, num_events, num_channels = target.size()

        target = self.data_processor.preprocess(target)
        target_embedded = self.data_processor.embed(target)
        target_embedded = self.linear_target(target_embedded)
        target_seq = flatten(target_embedded)

        num_tokens_target = target_seq.size(1)
        # add positional embeddings
        target_seq, h_pe = self.target_positional_embedding(target_seq, h=h_pe_init)

        # shift target_seq by one
        # Pad

        dummy_input_target = self.sos_target.repeat(batch_size, 1, 1)
        target_seq = torch.cat(
            [
                dummy_input_target,
                target_seq
            ],
            dim=1)
        target_seq = target_seq[:, :-1]

        output = self.transformer(
            target_seq
        )

        output = output.view(batch_size,
                             -1,
                             self.num_channels_target,
                             self.d_model)
        weights_per_category = [
            pre_softmax(t[:, :, 0, :])
            for t, pre_softmax in zip(output.split(1, 2), self.pre_softmaxes)
        ]

        # we can change loss mask
        loss = categorical_crossentropy(
            value=weights_per_category,
            target=target,
            mask=torch.ones_like(target)
        )

        loss = loss.mean()
        return {
            'loss':                 loss,
            'h_pe':                 h_pe,
            'weights_per_category': weights_per_category,
            'monitored_quantities': {
                'loss': loss.item()
            }
        }

    def forward_step(self, target, state, i, h_pe):
        """
        if i == 0, target is not used: SOS instead
        :param target: sequence of tokens (batch_size,)
        :param state:
        :param i:
        :param h_pe:
        :return:
        """
        # deal with the SOS token embedding
        if i == 0:
            target_seq = self.sos_target.repeat(target.size(0), 1, 1)[:, 0, :]
        else:
            channel_index_input = (i - 1) % self.num_channels_target
            target = self.data_processor.preprocess(target)
            target_embedded = self.data_processor.embed_step(
                target,
                channel_index=channel_index_input)
            target_embedded = self.linear_target(target_embedded)
            # add positional embeddings
            target_seq, h_pe = self.target_positional_embedding.forward_step(
                target_embedded,
                i=(i - 1),
                h=h_pe)

        output, state = self.transformer.forward_step(
            target_seq, state=state
        )

        channel_index_output = i % self.num_channels_target

        weights = self.pre_softmaxes[channel_index_output](output)

        # no need for a loss
        return {
            'loss':    None,
            'state':   state,
            'h_pe':    h_pe,
            'weights': weights,
        }

    def epoch(self, data_loader,
              train=True,
              num_batches=None,
              ):
        means = None

        if train:
            self.train()
        else:
            self.eval()

        h_pe_init = None
        for sample_id, tensor_dict in tqdm(enumerate(
                islice(data_loader, num_batches)),
                ncols=80):

            # ==========================
            with torch.no_grad():
                x = tensor_dict['x']

            # ========Train decoder =============
            self.optimizer.zero_grad()
            forward_pass = self.forward(
                target=x, h_pe_init=h_pe_init
            )
            loss = forward_pass['loss']
            # h_pe_init = forward_pass['h_pe'].detach()

            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 5)
                self.optimizer.step()

            # Monitored quantities
            monitored_quantities = forward_pass['monitored_quantities']

            # average quantities
            if means is None:
                means = {key: 0
                         for key in monitored_quantities}
            means = {
                key: value + means[key]
                for key, value in monitored_quantities.items()
            }

            del loss

        # renormalize monitored quantities
        means = {
            key: value / (sample_id + 1)
            for key, value in means.items()
        }
        return means

    def train_model(self,
                    batch_size,
                    num_batches=None,
                    num_epochs=10,
                    lr=1e-3,
                    plot=False,
                    num_workers=0,
                    **kwargs
                    ):
        if plot:
            self.writer = SummaryWriter(f'{self.model_dir}')

        best_val = 1e8
        self.init_optimizers(lr=lr)
        for epoch_id in range(num_epochs):
            (generator_train,
             generator_val,
             generator_test) = self.dataloader_generator.dataloaders(
                batch_size=batch_size,
                num_workers=num_workers)

            monitored_quantities_train = self.epoch(
                data_loader=generator_train,
                train=True,
                num_batches=num_batches,
            )
            del generator_train

            # monitored_quantities_val = self.epoch(
            #     data_loader=generator_val,
            #     train=False,
            #     num_batches=num_batches // 2 if num_batches is not None else None,
            # )
            # del generator_val
            # valid_loss = monitored_quantities_val['loss']

            valid_loss = 0.
            monitored_quantities_val = {}

            # self.scheduler.step(monitored_quantities_val["loss"])

            print(f'======= Epoch {epoch_id} =======')
            print(f'---Train---')
            dict_pretty_print(monitored_quantities_train, endstr=' ' * 5)
            print()
            print(f'---Val---')
            dict_pretty_print(monitored_quantities_val, endstr=' ' * 5)
            print('\n')

            self.save(early_stopped=False)
            if valid_loss < best_val:
                self.save(early_stopped=True)
                best_val = valid_loss

            if plot:
                self.plot(epoch_id,
                          monitored_quantities_train,
                          monitored_quantities_val)

    def plot(self, epoch_id, monitored_quantities_train,
             monitored_quantities_val):
        for k, v in monitored_quantities_train.items():
            self.writer.add_scalar(f'{k}/train', v, epoch_id)
        for k, v in monitored_quantities_val.items():
            self.writer.add_scalar(f'{k}/val', v, epoch_id)

    def generate_non_recurrent(self, temperature, batch_size=1, plot_attentions=False,
                               top_k=0, top_p=1.):
        self.eval()

        with torch.no_grad():
            x = self.init_generation(num_events=self.data_processor.num_events)
            # Duplicate along batch dimension
            x = x.repeat(batch_size, 1, 1)

            h_pe_init = None

            for event_index in range(x.size(1)):
                # for event_index in range(self.data_processor.num_events):
                for channel_index in range(self.num_channels_target):
                    forward_pass = self.forward(x, h_pe_init=h_pe_init)

                    weights_per_voice = forward_pass['weights_per_category']
                    weights = weights_per_voice[channel_index]

                    # Keep only the last token predictions of the first batch item (batch size 1), apply a temperature coefficient and filter
                    logits = weights[:, event_index, :] / temperature

                    # # exclude non note symbols:
                    exclude_symbols = ['START', 'END', 'XX']
                    for sym in exclude_symbols:
                        sym_index = self.dataloader_generator.dataset.note2index_dicts[
                            channel_index][sym]
                        logits[:, sym_index] = -np.inf

                    filtered_logits = []
                    for logit in logits:
                        filter_logit = top_k_top_p_filtering(logit, top_k=top_k, top_p=top_p)
                        filtered_logits.append(filter_logit)
                    filtered_logits = torch.stack(filtered_logits, dim=0)
                    # Sample from the filtered distribution
                    p = to_numpy(torch.softmax(filtered_logits, dim=-1))

                    # update generated sequence
                    for batch_index in range(batch_size):
                        new_pitch_index = np.random.choice(np.arange(
                            self.num_tokens_per_channel[channel_index]
                        ), p=p[batch_index])
                        x[batch_index, event_index, channel_index] = int(new_pitch_index)

        # to score
        original_and_reconstruction = self.data_processor.postprocess(x.cpu())

        ###############################
        # Saving
        timestamp = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
        if not os.path.exists(f'{self.model_dir}/generations'):
            os.mkdir(f'{self.model_dir}/generations')

        # Write scores
        scores = []
        for k, tensor_score in enumerate(original_and_reconstruction):
            path_no_extension = f'{self.model_dir}/generations/{timestamp}_{k}'
            scores.append(self.dataloader_generator.write(tensor_score, path_no_extension))
        ###############################

        return scores

    def generate(self, temperature, batch_size=1,
                 top_k=0, top_p=1.):
        assert self.recurrent
        self.eval()
        # num_events = 4 * 4 * 24
        # num_events = 240
        num_events = 1024

        x = torch.zeros(batch_size, num_events, self.num_channels_target).long()
        with torch.no_grad():

            # init
            xi = torch.zeros_like(x)[:, 0, 0]
            state = None
            h_pe = None

            # i corresponds to the position of the token BEING generated
            for event_index in range(num_events):
                for channel_index in range(self.num_channels_target):
                    i = event_index * self.num_channels_target + channel_index

                    forward_pass = self.forward_step(xi,
                                                     state=state,
                                                     i=i, h_pe=h_pe)
                    weights = forward_pass['weights']

                    logits = weights / temperature

                    # # Removing these lines make the method applicable to all datasets
                    # TODO separate method in dataprocessor?
                    # # exclude non note symbols:
                    # exclude_symbols = ['START', 'END', 'XX']
                    # for sym in exclude_symbols:
                    #     sym_index = self.dataloader_generator.dataset.note2index_dicts[
                    #         channel_index][sym]
                    #     logits[:, sym_index] = -np.inf



                    filtered_logits = []
                    for logit in logits:
                        filter_logit = top_k_top_p_filtering(logit, top_k=top_k, top_p=top_p)
                        filtered_logits.append(filter_logit)
                    filtered_logits = torch.stack(filtered_logits, dim=0)
                    # Sample from the filtered distribution
                    p = to_numpy(torch.softmax(filtered_logits, dim=-1))


                    # update generated sequence
                    for batch_index in range(batch_size):
                        new_pitch_index = np.random.choice(np.arange(
                            self.num_tokens_per_channel[channel_index]
                        ), p=p[batch_index])
                        x[batch_index, event_index, channel_index] = int(new_pitch_index)

                    # update
                    xi = x[:, event_index, channel_index]
                    h_pe = forward_pass['h_pe']
                    state = forward_pass['state']

        # to score
        original_and_reconstruction = self.data_processor.postprocess(x.cpu())

        ###############################
        # Saving
        timestamp = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
        if not os.path.exists(f'{self.model_dir}/generations'):
            os.mkdir(f'{self.model_dir}/generations')

        # Write scores
        scores = []
        for k, tensor_score in enumerate(original_and_reconstruction):
            path_no_extension = f'{self.model_dir}/generations/{timestamp}_{k}'
            # TODO fix write signature
            scores.append(self.dataloader_generator.write(tensor_score, path_no_extension))
            # scores.append(self.dataloader_generator.write(tensor_score.unsqueeze(0),
            #                                               path_no_extension))
        ###############################

        return scores

    def generate_from_code_long(self, encoding_indices,
                                temperature,
                                top_k=0,
                                top_p=1.,
                                exclude_meta_symbols=False,
                                num_decodings=1,
                                code_index_start=None,
                                code_index_end=None):
        """
        Returns a list of music21 scores
        """
        self.eval()
        size_encoding = encoding_indices.size(1)

        total_upscaling = self.total_upscaling
        num_tokens_indices = self.data_processor.num_tokens // total_upscaling

        num_events_full_chorale = size_encoding * total_upscaling // self.data_processor.num_channels
        num_events_before_start = code_index_start * total_upscaling // self.num_channels_target
        num_events_before_end = code_index_end * total_upscaling // self.num_channels_target

        batch_size = num_decodings * encoding_indices.size(0)

        if code_index_start is None:
            code_index_start = 0
        if code_index_end is None:
            code_index_end = size_encoding

        with torch.no_grad():
            chorale = self.init_generation(
                num_events=num_events_full_chorale
            )
            # chorale = self.init_generation_chorale(num_events=num_events_full_chorale,
            #                                        start_index=num_events_before_start)
            # Duplicate along batch dimension
            chorale = chorale.repeat(batch_size, 1, 1)
            encoding_indices = encoding_indices.repeat_interleave(num_decodings, dim=0)

            for code_index in range(code_index_start, code_index_end):
                for relative_event in range(self.num_events_per_code):
                    for channel_index in range(self.data_processor.num_channels):
                        t_begin, t_end, t_relative = self.compute_start_end_times(
                            code_index, num_blocks=size_encoding,
                            num_blocks_model=num_tokens_indices
                        )

                        input_encoding_indices = encoding_indices[:, t_begin:t_end]

                        input_chorale = chorale[:,
                                        t_begin * self.num_events_per_code: t_end * self.num_events_per_code,
                                        :]
                        weights_per_voice = self.forward(input_encoding_indices,
                                                         input_chorale)['weights_per_category']

                        # Keep only the last token predictions of the first batch item (batch size 1), apply a
                        # temperature coefficient and filter
                        weights = weights_per_voice[channel_index]
                        logits = weights[:, t_relative * self.num_events_per_code + relative_event,
                                 :] / temperature

                        # Top-p sampling
                        filtered_logits = []
                        for logit in logits:
                            filter_logit = top_k_top_p_filtering(logit, top_k=top_k, top_p=top_p)
                            filtered_logits.append(filter_logit)
                        filtered_logits = torch.stack(filtered_logits, dim=0)
                        # Sample from the filtered distribution
                        p = to_numpy(torch.softmax(filtered_logits, dim=-1))

                        for batch_index in range(batch_size):
                            new_pitch_index = np.random.choice(np.arange(
                                self.num_tokens_per_channel[channel_index]
                            ), p=p[batch_index])
                            chorale[batch_index,
                                    code_index * self.num_events_per_code + relative_event,
                                    channel_index] = int(
                                new_pitch_index)

        # slice
        chorale = chorale[:, num_events_before_start:num_events_before_end]
        # Write scores
        scores = self.dataloader_generator.to_score(chorale)
        return scores

    def check_duplicate(self, generation, original):
        from difflib import SequenceMatcher
        s1 = self.data_processor.dump(generation)
        s2 = self.data_processor.dump(original)

        match = SequenceMatcher(None, s1, s2).find_longest_match(0, len(s1), 0, len(s2))
        print(match)
        print(s1[match.a: match.a + match.size])
        print(f'Num tokens plagiarisms: {(match.size - 1) / 3}')

    def check_duplicate_all_corpus(self, generation):
        from difflib import SequenceMatcher
        s1 = self.data_processor.dump(generation)
        (generator_train, generator_val, _) = self.dataloader_generator.dataloaders(
            batch_size=1,
            shuffle_val=True,
            shuffle_train=False
        )
        # todo use generator train!
        best_x = None
        best_size = 0
        for tensor_dict in tqdm(generator_train):
            x = tensor_dict['x']
            s2 = self.data_processor.dump(x[0])
            match = SequenceMatcher(None, s1, s2, autojunk=False).find_longest_match(0, len(s1),
                                                                                     0, len(s2))
            if match.size > best_size:
                best_x = x
                best_size = match.size
            # print(match)
            # print(s1[match.a: match.a + match.size])

        print(f'Num tokens plagiarisms: {(best_size - 1) / 3}')
        print(f'Num beats plagiarisms: {(best_size - 1) / 3 / 4 / 4}')

        return best_x

    def plot_attention(self,
                       attentions_list,
                       timestamp,
                       name):
        """
        Helper function

        :param attentions_list: list of (batch_size, num_heads, num_tokens_encoder

        :return:
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        # to (batch_size, num_heads, num_tokens_decoder, num_tokens_encoder)
        attentions_batch = torch.cat(
            [t.unsqueeze(2)
             for t in attentions_list
             ], dim=2
        )

        # plot only batch 0 for now
        for batch_index, attentions in enumerate(attentions_batch):
            plt.clf()
            plt.cla()
            num_heads = attentions.size(0)
            for head_index, t in enumerate(attentions):
                plt.subplot(1, num_heads, head_index + 1)
                plt.title(f'Head {head_index}')
                mat = t.detach().cpu().numpy()
                sns.heatmap(mat, vmin=0, vmax=1, cmap="YlGnBu")
                plt.grid(True)
            plt.savefig(f'{self.model_dir}/generations/{timestamp}_{batch_index}_{name}.pdf')
            # plt.show()
        plt.close()

    def init_generation(self, num_events):
        return cuda_variable(
            torch.zeros(1, num_events, self.num_channels_target).long()
        )

    # TODO put this in data_processor/dataloader_generator
    # but hard!
    def init_generation_chorale(self, num_events, start_index):
        PAD = [d[PAD_SYMBOL] for d in self.dataloader_generator.dataset.note2index_dicts]
        START = [d[START_SYMBOL] for d in self.dataloader_generator.dataset.note2index_dicts]
        aa = torch.Tensor(PAD).unsqueeze(0).unsqueeze(0).repeat(1, start_index - 1, 1).long()
        bb = torch.Tensor(START).unsqueeze(0).unsqueeze(0).long()
        cc = torch.Tensor(PAD).unsqueeze(0).unsqueeze(0).repeat(1, num_events - start_index,
                                                                1).long()
        init_sequence = torch.cat([aa, bb, cc], 1)
        return init_sequence

    def compute_start_end_times(self, t, num_blocks, num_blocks_model):
        """

        :param t:
        :param num_blocks: num_blocks of the sequence to be generated
        :param num_blocks_model:
        :return:
        """
        # t_relative
        if num_blocks_model // 2 <= t < num_blocks - num_blocks_model // 2:
            t_relative = (num_blocks_model // 2)
        else:
            if t < num_blocks_model // 2:
                t_relative = t
            elif t >= num_blocks - num_blocks_model // 2:
                t_relative = num_blocks_model - (num_blocks - t)
            else:
                NotImplementedError

        # choose proper block to use
        t_begin = min(max(0, t - num_blocks_model // 2), num_blocks - num_blocks_model)
        t_end = t_begin + num_blocks_model

        return t_begin, t_end, t_relative

    def generate_reharmonisation(self, num_reharmonisations,
                                 temperature,
                                 top_k=0,
                                 top_p=1.
                                 ):
        """
        This method only works on bach chorales
        :param num_reharmonisations:
        :param temperature:
        :return:
        """
        import music21
        cl = music21.corpus.chorales.ChoraleList()
        print(cl.byBWV.keys())
        # chorale_m21 = music21.corpus.chorales.getByTitle(cl.byBWV[437]['title'])
        # chorale_m21 = music21.corpus.chorales.getByTitle(cl.byBWV[289]['title'])

        for bwv in cl.byBWV.keys():
            chorale_m21 = music21.corpus.chorales.getByTitle(cl.byBWV[bwv]['title'])
            x = self.dataloader_generator.dataset.transposed_score_and_metadata_tensors(
                chorale_m21, semi_tone=0)[0].transpose(1, 0).unsqueeze(0)
            # remove metadata
            # and put num_channels at the end
            # and add batch_dim

            x_chunks = list(x.split(self.data_processor.num_events, 1))

            last_chunk = x_chunks[-1]

            # compute START and END clusters
            PAD = [d[PAD_SYMBOL] for d in self.dataloader_generator.dataset.note2index_dicts]
            START = [d[START_SYMBOL] for d in self.dataloader_generator.dataset.note2index_dicts]
            END = [d[END_SYMBOL] for d in self.dataloader_generator.dataset.note2index_dicts]

            # start
            start_chunk_ = torch.Tensor(START).unsqueeze(0).unsqueeze(0).long()
            pad_chunk_beginning = torch.Tensor(PAD).unsqueeze(0).unsqueeze(0).repeat(
                1, self.data_processor.num_events - 1, 1
            ).long()
            start_chunk = torch.cat([pad_chunk_beginning, start_chunk_], 1)

            # end
            end_chunk_ = torch.Tensor(END).unsqueeze(0).unsqueeze(0).long()
            pad_chunk_end = torch.Tensor(PAD).unsqueeze(0).unsqueeze(0).repeat(
                1, self.data_processor.num_events - 1, 1
            ).long()
            end_chunk = torch.cat([end_chunk_, pad_chunk_end], 1)

            # last chunk
            completion_chunk = torch.Tensor(PAD).unsqueeze(0).unsqueeze(0).repeat(
                1, self.data_processor.num_events - last_chunk.size(1) - 1, 1
            ).long()
            last_chunk = torch.cat([last_chunk, end_chunk_, completion_chunk], 1)
            x_chunks[-1] = last_chunk

            x_chunks = torch.cat([start_chunk] + x_chunks + [end_chunk], dim=0)
            encoding_indices = self.encoders_stack(x_chunks)

            # glue all encoding indices
            encoding_indices = torch.cat(
                encoding_indices.split(1, 0),
                1
            )
            # compute code_index start and stop
            total_upscaling = int(np.prod(self.encoders_stack.downscale_factors))
            code_index_start = start_chunk.size(1) * self.num_channels_target // total_upscaling
            code_index_end = encoding_indices.size(1) - (
                    end_chunk.size(1) + completion_chunk.size(1)) * self.num_channels_target // \
                             total_upscaling

            scores = self.generate_from_code_long(encoding_indices,
                                                  num_decodings=num_reharmonisations,
                                                  temperature=temperature,
                                                  code_index_start=code_index_start,
                                                  code_index_end=code_index_end
                                                  )

            reharmonisation_dir = f'{self.model_dir}/reharmonisations'
            if not os.path.exists(reharmonisation_dir):
                os.mkdir(reharmonisation_dir)
            for k, score in enumerate(scores):
                score.write('xml', f'{reharmonisation_dir}/BWV{bwv}_{k}.xml')
                # score.show()
            chorale_m21.write('xml', f'{reharmonisation_dir}/BWV{bwv}_original.xml')
        return scores

    def generate_alla_mano(self, start_codes, end_codes, body_codes, temperature):
        code_index_start = len(start_codes)
        encoding_indices = start_codes + \
                           body_codes
        code_index_end = len(encoding_indices)
        encoding_indices += end_codes
        encoding_indices = torch.Tensor(encoding_indices).unsqueeze(0).long().to('cuda')

        scores = self.generate_from_code_long(
            encoding_indices=encoding_indices,
            temperature=temperature,
            num_decodings=3,
            code_index_start=code_index_start,
            code_index_end=code_index_end)

        save_dir = f'{self.model_dir}/alla_mano'
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        for k, score in enumerate(scores):
            score.write('xml', f'{save_dir}/{k}.xml')

        return scores

    def generate_completion(self,
                            num_completions,
                            temperature,
                            top_k,
                            top_p,
                            midi_file,
                            ):
        """
        This method only works on harpsichord
        :param num_completions:
        :param temperature:
        :return:
        """
        self.eval()
        original = self.dataloader_generator.dataset.process_score(midi_file)
        original = self.dataloader_generator.dataset.tokenize(original)
        # TODO put in preprocess
        x = torch.stack([torch.LongTensor(original[e])
                         for e in self.dataloader_generator.features], dim=-1)
        # todo filter
        # add batch_size
        num_events = 1024

        x = x.unsqueeze(0).repeat(num_completions, 1, 1)
        start_event_index = x.size(1)
        x = torch.cat([x,
                       torch.zeros(num_completions, num_events - start_event_index,
                                   self.num_channels_target).long()],
                      dim=1)

        with torch.no_grad():
            # init
            xi = torch.zeros_like(x)[:, 0, 0]
            state = None
            h_pe = None

            # i corresponds to the position of the token BEING generated
            for event_index in range(num_events):
                for channel_index in range(self.num_channels_target):
                    i = event_index * self.num_channels_target + channel_index

                    forward_pass = self.forward_step(xi,
                                                     state=state,
                                                     i=i, h_pe=h_pe)
                    weights = forward_pass['weights']

                    logits = weights / temperature

                    # # Removing these lines make the method applicable to all datasets
                    # TODO separate method in dataprocessor?
                    # # exclude non note symbols:
                    # exclude_symbols = ['START', 'END', 'XX']
                    # for sym in exclude_symbols:
                    #     sym_index = self.dataloader_generator.dataset.note2index_dicts[
                    #         channel_index][sym]
                    #     logits[:, sym_index] = -np.inf

                    filtered_logits = []
                    for logit in logits:
                        filter_logit = top_k_top_p_filtering(logit, top_k=top_k, top_p=top_p)
                        filtered_logits.append(filter_logit)
                    filtered_logits = torch.stack(filtered_logits, dim=0)
                    # Sample from the filtered distribution
                    p = to_numpy(torch.softmax(filtered_logits, dim=-1))

                    # update generated sequence
                    for batch_index in range(num_completions):
                        new_pitch_index = np.random.choice(np.arange(
                            self.num_tokens_per_channel[channel_index]
                        ), p=p[batch_index])

                        # complete only
                        if event_index >= start_event_index:
                            x[batch_index, event_index, channel_index] = int(new_pitch_index)

                    # update
                    xi = x[:, event_index, channel_index]
                    h_pe = forward_pass['h_pe']
                    state = forward_pass['state']


        # Saving
        original_and_reconstruction = self.data_processor.postprocess(x.cpu())
        timestamp = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
        if not os.path.exists(f'{self.model_dir}/completions'):
            os.mkdir(f'{self.model_dir}/completions')

        # Write scores
        scores = []
        for k, tensor_score in enumerate(original_and_reconstruction):
            path_no_extension = f'{self.model_dir}/completions/{timestamp}_{k}'
            scores.append(self.dataloader_generator.write(tensor_score, path_no_extension))

        return scores
