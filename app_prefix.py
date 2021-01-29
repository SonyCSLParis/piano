import json
from flask import Flask
from flask import request
from flask.helpers import make_response
from flask.json import JSONDecoder, jsonify
from torch.utils import data
from BFT.utils import cuda_variable
from flask_cors import CORS
from BFT.handlers import DecoderPrefixHandler

app = Flask(__name__)
CORS(app)
"""
@author: Gaetan Hadjeres
"""
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
from BFT.getters import get_data_processor, get_dataloader_generator, get_decoder, get_sos_embedding, get_positional_embedding

DEBUG = False


@click.command()
@click.argument('cmd')
@click.option('-o', '--overfitted', is_flag=True)
@click.option('-c', '--config', type=click.Path(exists=True))
@click.option('-n', '--num_workers', type=int, default=0)
def launcher(cmd, overfitted, config, num_workers):
    # === Set shared parameters
    # only use 1 GPU for inference
    print(cmd)
    assert cmd == 'serve'
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
    model_dir = os.path.dirname(config_path)

    print(f'Using {world_size} GPUs')
    mp.spawn(main,
             args=(overfitted, config, num_workers, world_size, model_dir),
             nprocs=world_size,
             join=True)


def main(rank, overfitted, config, num_workers, world_size, model_dir):
    # === Init process group
    os.environ['MASTER_ADDR'] = 'localhost'
    # os.environ['MASTER_PORT'] = '12355'
    # os.environ['MASTER_PORT'] = '12356'
    os.environ['MASTER_PORT'] = '12357'
    dist.init_process_group(backend='nccl', world_size=world_size, rank=rank)
    torch.cuda.set_device(rank)
    device = f'cuda:{rank}'

    # === Decoder ====
    # dataloader generator
    dataloader_generator = get_dataloader_generator(
        dataset=config['dataset'],
        dataloader_generator_kwargs=config['dataloader_generator_kwargs'])

    # data processor
    global data_processor
    data_processor = get_data_processor(
        dataloader_generator=dataloader_generator,
        data_processor_type=config['data_processor_type'],
        data_processor_kwargs=config['data_processor_kwargs'])

    # positional embedding
    positional_embedding_target: PositionalEmbedding = get_positional_embedding(
        dataloader_generator=dataloader_generator,
        positional_embedding_dict=config['positional_embedding_dict'])
    # sos embedding
    sos_embedding = get_sos_embedding(
        dataloader_generator=dataloader_generator,
        sos_embedding_dict=config['sos_embedding_dict'])
    encoder_decoder = get_decoder(
        data_processor=data_processor,
        dataloader_generator=dataloader_generator,
        positional_embedding=positional_embedding_target,
        sos_embedding=sos_embedding,
        decoder_type=config['decoder_type'],
        decoder_kwargs=config['decoder_kwargs'],
        training_phase=False)

    encoder_decoder.to(device)
    encoder_decoder = DistributedDataParallel(
        module=encoder_decoder,
        device_ids=[rank],
        output_device=rank,
        #   find_unused_parameters=True
    )

    global handler
    handler = DecoderPrefixHandler(model=encoder_decoder,
                                   model_dir=model_dir,
                                   dataloader_generator=dataloader_generator)
    # Load model
    if overfitted:
        handler.load(early_stopped=False, recurrent=True)
    else:
        handler.load(early_stopped=True, recurrent=True)

    local_only = False
    if local_only:
        # accessible only locally:
        app.run(threaded=True)
    else:
        # accessible from outside:
        port = 5000 if DEBUG else 8080

        app.run(host='0.0.0.0',
                port=port,
                threaded=True,
                debug=DEBUG,
                use_reloader=False)


@app.route('/ping', methods=['GET'])
def ping():
    return 'pong'


@app.route('/invocations', methods=['POST'])
def invocations():

    # === Parse request ===
    # common components
    d = json.loads(request.data)
    case = d['case']
    assert case in ['start', 'continue']
    if DEBUG:
        print(d)

    notes = d['notes']
    selected_region = d['selected_region']
    clip_start = d['clip_start']
    tempo = d['tempo']
    beats_per_second = tempo / 60
    seconds_per_beat = 1 / beats_per_second

    # two different parsing methods
    if case == 'start':
        num_max_generated_events = 15
        (x, metadata_dict, unused_before, before, after, unsused_after,
         clip_start,
         selected_region) = ableton_to_tensor(notes, clip_start,
                                              seconds_per_beat,
                                              selected_region)
    elif case == 'continue':
        num_max_generated_events = 30
        json_notes = d['notes']
        event_start = d['next_event_start']
        event_end = d['next_event_end']
        x, num_events_before_padding = json_to_tensor(json_notes,
                                                      seconds_per_beat,
                                                      event_start, event_end,
                                                      selected_region)
    else:
        raise NotImplementedError

    global handler

    generated_region, done = handler.inpaint(
        x=x,
        metadata_dict=metadata_dict,
        num_max_generated_events=None,  # TODO change
        temperature=1.,
        top_p=0.95,
        top_k=0)

    new_x = torch.cat([
        unused_before[0], before[0], generated_region[0], after[0],
        unsused_after[0]
    ],
                      dim=0).detach().cpu()

    # TODO use done to rescale

    ableton_notes, track_duration = tensor_to_ableton(
        new_x, start_time=clip_start, beats_per_second=beats_per_second)

    ableton_notes_region, _ = tensor_to_ableton(
        generated_region[0].detach().cpu(),
        start_time=selected_region['start'],
        beats_per_second=beats_per_second)

    after_region = torch.cat([after[0], unsused_after[0]],
                             dim=0).detach().cpu()

    ableton_notes_after_region, _ = tensor_to_ableton(
        after_region,
        start_time=selected_region['end'],  # TODO WRONG!,
        beats_per_second=beats_per_second)

    print(f'albeton notes: {ableton_notes}')
    print(f'region start: {ableton_notes_region}')
    d = {
        'id': d['id'],
        'notes': ableton_notes,
        'track_duration': track_duration,
        'done': done,
        'selected_region': selected_region,
        'notes_region': ableton_notes_region,
        'notes_after_region': ableton_notes_after_region,
        'clip_start': clip_start,
        'clip_id': d['clip_id'],
        'clip_end': d['clip_end'],
        'detail_clip_id': d['detail_clip_id'],
        'tempo': d['tempo']
    }
    return jsonify(d)


def preprocess_input(x, event_start, event_end):
    """

    Args:
        x ([type]): original sequence (num_events, num_channels )
        event_start ([type]): indicates the beginning of the recomposed region
        event_end ([type]): indicates the end of the recomposed region
        note_density ([type]): [description]

    Returns:
        [type]: [description]
    """
    global data_processor
    global handler
    # add batch_dim
    # only one proposal for now
    num_examples = 1
    x = x.unsqueeze(0).repeat(num_examples, 1, 1)
    _, x, _ = data_processor.preprocess(x)

    total_length = x.size(1)

    # if x is too large
    # x is always >= 1024 since we pad
    num_events_model = handler.dataloader_generator.sequences_size
    if total_length > num_events_model:
        # slice
        slice_begin = max((event_start - num_events_model // 2), 0)
        slice_end = slice_begin + num_events_model

        x_beginning = x[:, :slice_begin]
        x_end = x[:, slice_end:]
        x = x[:, slice_begin:slice_end]
    else:
        x_beginning = torch.zeros(1, 0).to(x.device)
        x_end = torch.zeros(1, 0).to(x.device)

    offset = slice_begin
    masked_positions = torch.zeros_like(x).long()
    masked_positions[:, event_start - offset:event_end - offset] = 1

    # the last time shift should be known:
    # TODO check this condition : should be done in conjunction with setting the correct duration of the inpainted region
    # if event_end < total_length:
    #     masked_positions[:, event_end - offset - 1, 3] = 0

    return (x_beginning, x, x_end), masked_positions, slice_begin


def json_to_tensor(json_note_list, seconds_per_beat, event_start, event_end,
                   selected_region):
    # TODO!
    d = {
        'pitch': [],
        'time': [],
        'duration': [],
        'velocity': [],
        'muted': [],
    }
    # pitch time duration velocity muted

    for n in json_note_list:
        for k, v in n.items():
            d[k].append(v)

    # we now have to sort
    l = [[p, t, d, v] for p, t, d, v in zip(d['pitch'], d['time'],
                                            d['duration'], d['velocity'])]

    l = sorted(l, key=lambda x: (x[1], -x[0]))

    d = dict(pitch=torch.LongTensor([x[0] for x in l]),
             time=torch.FloatTensor([x[1] for x in l]),
             duration=torch.FloatTensor([max(float(x[2]), 0.05) for x in l]),
             velocity=torch.LongTensor([x[3] for x in l]))

    # multiply by tempo
    d['time'] = d['time'] * seconds_per_beat
    d['duration'] = d['duration'] * seconds_per_beat

    # compute time_shift
    d['time_shift'] = torch.cat(
        [d['time'][1:] - d['time'][:-1],
         torch.zeros(1, )], dim=0)

    # Recompute time shifts in the selected_region
    # Set correct time shifts and compute masked_positions
    # TODO THIS CAN BE NEGATIVE!
    start_time = d['time'][event_start]
    end_time = selected_region['end'] * seconds_per_beat
    num_events_to_compose = event_end - event_start
    d['time_shift'][event_start:event_end] = ((end_time - start_time.item()) /
                                              num_events_to_compose)

    global handler
    num_events_before_padding = d['pitch'].size(0)
    # to numpy :(
    d = {k: t.numpy() for k, t in d.items()}

    # delete unnecessary entries in dict
    del d['time']
    # TODO over pad?
    d = handler.dataloader_generator.dataset.add_start_end_symbols(
        sequence=d, start_time=0, sequence_size=1024 + 512)

    sequence_dict = handler.dataloader_generator.dataset.tokenize(d)
    # to pytorch :)
    sequence_dict = {k: torch.LongTensor(t) for k, t in sequence_dict.items()}

    x = torch.stack(
        [sequence_dict[e] for e in handler.dataloader_generator.features],
        dim=-1).long()
    return x, num_events_before_padding


def ableton_to_tensor(ableton_note_list,
                      clip_start,
                      seconds_per_beat,
                      selected_region=None):
    """[summary]

    Args:
        ableton_note_list ([type]): [description]
        note_density ([type]): [description]
        clip_start ([type]): [description]
        selected_region ([type], optional): [description]. Defaults to None.

    Returns:
        x [type]: x is at least of size (1024, 4), it is padded if necessary
    """
    d = {
        'pitch': [],
        'time': [],
        'duration': [],
        'velocity': [],
        'muted': [],
    }
    mod = -1

    # pitch time duration velocity muted
    ableton_features = ['pitch', 'time', 'duration', 'velocity', 'muted']

    if selected_region is not None:
        start_time = selected_region['start']
        end_time = selected_region['end']

    for msg in ableton_note_list:
        if msg == 'notes':
            pass
        elif msg == 'note':
            mod = 0
        elif msg == 'done':
            break
        else:
            if mod >= 0:
                d[ableton_features[mod]].append(msg)
                mod = (mod + 1) % 5

    # we now have to sort
    l = [[p, t, d, v] for p, t, d, v in zip(d['pitch'], d['time'],
                                            d['duration'], d['velocity'])]

    l = sorted(l, key=lambda x: (x[1], -x[0]))

    d = dict(pitch=torch.LongTensor([x[0] for x in l]),
             time=torch.FloatTensor([x[1] for x in l]),
             duration=torch.FloatTensor([max(float(x[2]), 0.05) for x in l]),
             velocity=torch.LongTensor([x[3] for x in l]))

    # compute event_start, event_end
    # num_notes is the number of notes in the original sequence
    epsilon = 1e-4
    num_notes = d['time'].size(0)

    event_start, event_end = None, None
    if selected_region is not None:
        i = 0
        flag = True
        while flag:
            if i == num_notes:
                event_start = num_notes
                break
            if d['time'][i].item() >= start_time - epsilon:
                flag = False
                event_start = i
            else:
                i = i + 1

        i = 0
        flag = True
        while flag:
            if i == num_notes:
                event_end = num_notes
                break
            if i > d['time'].size(0):
                flag = False
                event_end = i
            elif d['time'][i].item() >= end_time - epsilon:
                flag = False
                event_end = i
            else:
                i = i + 1

    # multiply by tempo
    d['time'] = d['time'] * seconds_per_beat
    d['duration'] = d['duration'] * seconds_per_beat

    # compute time_shift
    d['time_shift'] = torch.cat(
        [d['time'][1:] - d['time'][:-1],
         torch.zeros(1, )], dim=0)

    # Remove selected region and replace it with a placeholder
    # end_time must be the first starting time after the selected region
    if event_end < num_notes:
        end_time = d['time'][event_end].item() / seconds_per_beat
        # update_selected_region
        selected_region['end'] = end_time
    placeholder_duration = (end_time - start_time) * seconds_per_beat
    placeholder_duration = cuda_variable(torch.Tensor([placeholder_duration]))
    global data_processor

    placeholder = data_processor.compute_placeholder(
        placeholder_duration=placeholder_duration, batch_size=1)

    # delete unnecessary entries in dict
    del d['time']
    before = {k: v[:event_start] for k, v in d.items()}
    after = {k: v[event_end:] for k, v in d.items()}

    global handler

    # format and pad
    # If we need to pad "before"
    if event_start < data_processor.num_events_before:
        before = {k: t.numpy() for k, t in before.items()}
        before = handler.dataloader_generator.dataset.add_start_end_symbols(
            sequence=before,
            start_time=event_start - data_processor.num_events_before,
            sequence_size=data_processor.num_events_before)

        before = handler.dataloader_generator.dataset.tokenize(before)
        before = {k: torch.LongTensor(t) for k, t in before.items()}
        before = torch.stack(
            [before[e] for e in handler.dataloader_generator.features],
            dim=-1).long()

        before = cuda_variable(before)
        unused_before = before[:0]
    else:
        before = {k: t.numpy() for k, t in before.items()}
        before = handler.dataloader_generator.dataset.add_start_end_symbols(
            sequence=before, start_time=0, sequence_size=event_start)
        before = handler.dataloader_generator.dataset.tokenize(before)
        before = {k: torch.LongTensor(t) for k, t in before.items()}
        before = torch.stack(
            [before[e] for e in handler.dataloader_generator.features],
            dim=-1).long()
        before = cuda_variable(before)

        unused_before, before = (before[:-data_processor.num_events_before],
                                 before[-data_processor.num_events_before:])

    # same for "after"
    num_notes_after = after['pitch'].size(0)

    # After cannot contain 'START' symbol
    after = {k: t.numpy() for k, t in after.items()}
    after = handler.dataloader_generator.dataset.add_start_end_symbols(
        sequence=after,
        start_time=0,
        sequence_size=max(num_notes_after, data_processor.num_events_after))

    after = handler.dataloader_generator.dataset.tokenize(after)
    after = {k: torch.LongTensor(t) for k, t in after.items()}
    after = torch.stack(
        [after[e] for e in handler.dataloader_generator.features],
        dim=-1).long()
    after = cuda_variable(after)

    after, unused_after = (after[:data_processor.num_events_after],
                           after[data_processor.num_events_after:])

    middle_length = (data_processor.dataloader_generator.sequences_size -
                     data_processor.num_events_before -
                     data_processor.num_events_after - 2)

    # add batch dim
    unused_before = unused_before.unsqueeze(0)
    before = before.unsqueeze(0)
    after = after.unsqueeze(0)
    unused_after = unused_after.unsqueeze(0)

    # create x:
    x = torch.cat([
        before, placeholder, after,
        data_processor.sod_symbols.unsqueeze(0).unsqueeze(0),
        cuda_variable(
            torch.zeros(1, middle_length, data_processor.num_channels))
    ],
                  dim=1).long()

    # if "before" was padded
    if event_start < data_processor.num_events_before:
        # (then event_start is the size of "before")
        before = before[:, -event_start:]

        # slicing does not work in this case
        if event_start == 0:
            before = before[:, :0]

    # if "after" was padded:
    if num_notes_after < data_processor.num_events_after:
        after = after[:, :num_notes_after]

    # update clip start if necessary
    if clip_start > start_time:
        clip_start = start_time

    metadata_dict = dict(original_sequence=x,
                         placeholder_duration=placeholder_duration,
                         decoding_start=data_processor.num_events_before +
                         data_processor.num_events_after + 2)

    return x, metadata_dict, unused_before, before, after, unused_after, clip_start, selected_region


def tensor_to_ableton(tensor, start_time, beats_per_second):
    """
    convert back a tensor to ableton format.
    Then shift all notes by clip start

    Args:
        tensor (num_events, num_channels)):
        clip_start
    """
    num_events, num_channels = tensor.size()
    if num_events == 0:
        return [], 0
    # channels are ['pitch', 'velocity', 'duration', 'time_shift']
    notes = []

    tensor = tensor.detach().cpu()
    global handler
    index2value = handler.dataloader_generator.dataset.index2value

    timeshifts = torch.FloatTensor(
        [index2value['time_shift'][ts.item()] for ts in tensor[:, 3]])
    time = torch.cumsum(timeshifts, dim=0)
    time = (torch.cat([torch.zeros(
        (1, )), time[:-1]], dim=0) * beats_per_second + start_time)
    for i in range(num_events):
        note = dict(pitch=index2value['pitch'][tensor[i, 0].item()],
                    time=time[i].item(),
                    duration=index2value['duration'][tensor[i, 2].item()] *
                    beats_per_second,
                    velocity=index2value['velocity'][tensor[i, 1].item()],
                    muted=0)
        notes.append(note)

    track_duration = time[-1].item() + notes[-1]['duration'].item()
    return notes, track_duration


if __name__ == "__main__":
    launcher()

# Response format
# {'id': '14', 'notes': ['notes', 10, 'note', 64, 0.5, 0.25, 100, 0, 'note', 64, 0.75, 0.25, 100, 0, 'note', 64, 1, 0.25, 100, 0, 'note', 65, 0.25, 0.25, 100, 0, 'note', 68, 1, 0.25, 100, 0, 'note', 69, 0, 0.25, 100, 0, 'note', 69, 0.75, 0.25, 100, 0, 'note', 69, 1.25, 2, 100, 0, 'note', 70, 0.5, 0.25, 100, 0, 'note', 71, 0.25, 0.25, 100, 0, 'done'], 'duration': 4}