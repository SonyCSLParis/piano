import json
from flask import Flask
from flask import request
from flask.helpers import make_response
from flask.json import JSONDecoder, jsonify
from torch.utils import data
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
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
    global data_processor
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
        training_phase=False)

    encoder_decoder.to(device)
    encoder_decoder = DistributedDataParallel(
        module=encoder_decoder,
        device_ids=[rank],
        output_device=rank,
        #   find_unused_parameters=True
    )

    global handler
    handler = EncoderDecoderHandler(model=encoder_decoder,
                                    model_dir=model_dir,
                                    dataloader_generator=dataloader_generator)
    # Load model
    if overfitted:
        handler.load(early_stopped=False)
    else:
        handler.load(early_stopped=True)

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
        note_density = d['note_density'] 
        (x, (event_start, event_end), num_events_before_padding,
         clip_start) = ableton_to_tensor(notes, note_density, clip_start,
                                         seconds_per_beat, selected_region)
    elif case == 'continue':
        num_max_generated_events = 30
        json_notes = d['notes']
        event_start = d['next_event_start']
        event_end = d['next_event_end']
        x, num_events_before_padding = json_to_tensor(json_notes, seconds_per_beat, event_start, event_end, selected_region)
    else:
        raise NotImplementedError

    (x_beginning, x, x_end), masked_positions, offset = preprocess_input(
        x, event_start, event_end)

    global handler
    print(f'event: {event_start, event_end}')
    if event_start + num_max_generated_events >= event_end:
        done = True
        event_end_region = event_end
    else:
        done = False
        event_end_region = event_start + num_max_generated_events

    output = handler.inpaint_region_optimized(
        x=x,
        masked_positions=masked_positions,
        start_event=event_start - offset,
        end_event=event_end_region - offset,
        temperature=1.,
        top_p=0.95,
        top_k=0)

    # join (x_beginning, x, x_end)
    new_x = torch.cat(
        [
            x_beginning[0].detach().cpu(), output[0].detach().cpu(),
            x_end[0].detach().cpu()
        ],
        dim=0)[:num_events_before_padding]  # removes padding and end symbols

    ableton_notes, ableton_notes_region, ableton_notes_after_region, track_duration = tensor_to_ableton(
        new_x,
        clip_start=clip_start,
        start_event=event_start,
        end_event=event_end_region,
        beats_per_second=beats_per_second)

    print(f'albeton notes: {ableton_notes}')
    print(f'region start: {ableton_notes_region}')
    d = {
        'id': d['id'],
        'notes': ableton_notes,
        'track_duration': track_duration,
        'done': done,
        'next_event_start': event_end_region,
        'next_event_end': event_end,
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



def json_to_tensor(json_note_list, seconds_per_beat,
                   event_start, event_end,
                   selected_region):
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
                      note_density,
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

    # Remove selected region and replace with the correct number of events
    # recompute event_start and event_end
    num_events_to_compose = int((end_time - start_time) * note_density)
    # TODO restrict to a maximum number of events to recompose
    assert num_events_to_compose < 256
    for k in d:
        d[k] = torch.cat([
            d[k][:event_start],
            torch.zeros(num_events_to_compose), d[k][event_end:]
        ],
                         dim=0)
    event_end = event_start + num_events_to_compose

    # Set correct time shifts and compute masked_positions
    d['time_shift'][event_start:event_end] = ((end_time - start_time) *
                                              seconds_per_beat /
                                              num_events_to_compose)

    # --- handle special cases
    # the selected region starts BEFORE the first notes of the clip
    if event_end == num_events_to_compose:
        if clip_start >= end_time:
            # the final time_shift is set to clip_start - end_time if the WHOLE region is before the start
            d['time_shift'][event_end -
                            1] = (clip_start - end_time) * seconds_per_beat
        # clip_start must be updated!
        clip_start = start_time
    elif event_start == num_notes:
        # the whole selected region is AFTER the clip
        # the final time shift is set to start_time - d[time] of the final note
        # in the original sequence
        assert start_time * seconds_per_beat >= d['time'][event_start - 1]
        d['time_shift'][
            event_start -
            1] = start_time * seconds_per_beat - d['time'][event_start - 1]
    else:
        assert start_time * seconds_per_beat >= d['time'][event_start - 1]
        # TODO we could regenerate this token + filtering
        # make the first note start exactly at start_time
        d['time_shift'][
            event_start -
            1] = start_time * seconds_per_beat - d['time'][event_start - 1]
        # make the last time shift to be exactly the gap between the region and the note
        # TODO check this! It's about the 
        # if d['time'].size(0) > event_end:
        #     d['time_shift'][
        #         event_end -
        #         1] = d['time'][event_end] - end_time * seconds_per_beat

    # adjustments
    # zero is not in pitch
    d['pitch'][event_start:event_end] = 60
    d['pitch'] = d['pitch'].long()
    d['velocity'] = d['velocity'].long()
    # zero duration does not exist
    d['duration'] = torch.max(d['duration'],
                              torch.ones_like(d['duration']) * 0.05)

    global handler
    num_events_before_padding = d['pitch'].size(0)

    # delete unnecessary entries in dict
    del d['time']

    # Pad and END symbol, returns a dict of lists
    # to numpy :(
    d = {k: t.numpy() for k, t in d.items()}

    # We overpad
    d = handler.dataloader_generator.dataset.add_start_end_symbols(
        sequence=d, start_time=0, sequence_size=1024 + 512)

    sequence_dict = handler.dataloader_generator.dataset.tokenize(d)
    # to pytorch :)
    sequence_dict = {k: torch.LongTensor(t) for k, t in sequence_dict.items()}

    x = torch.stack(
        [sequence_dict[e] for e in handler.dataloader_generator.features],
        dim=-1).long()
    return x, (event_start, event_end), num_events_before_padding, clip_start


def tensor_to_ableton(tensor, clip_start, start_event, end_event,
                      beats_per_second):
    """
    convert back a tensor to ableton format.
    Then shift all notes by clip start

    Args:
        tensor (num_events, num_channels)):
        clip_start
    """
    # channels are ['pitch', 'velocity', 'duration', 'time_shift']
    notes = []
    notes_region = []
    notes_after_region = []

    tensor = tensor.detach().cpu()
    global handler
    index2value = handler.dataloader_generator.dataset.index2value
    num_events, num_channels = tensor.size()
    timeshifts = torch.FloatTensor(
        [index2value['time_shift'][ts.item()] for ts in tensor[:, 3]])
    time = torch.cumsum(timeshifts, dim=0)
    time = (torch.cat([torch.zeros(
        (1, )), time[:-1]], dim=0) * beats_per_second + clip_start)
    for i in range(num_events):
        note = dict(pitch=index2value['pitch'][tensor[i, 0].item()],
                    time=time[i].item(),
                    duration=index2value['duration'][tensor[i, 2].item()] *
                    beats_per_second,
                    velocity=index2value['velocity'][tensor[i, 1].item()],
                    muted=0)
        notes.append(note)
        if start_event <= i < end_event:
            notes_region.append(note)
        if i >= end_event:
            notes_after_region.append(note)

    track_duration = time[-1].item() + notes[-1]['duration'].item()
    return notes, notes_region, notes_after_region, track_duration

if __name__ == "__main__":
    launcher()

# Response format
# {'id': '14', 'notes': ['notes', 10, 'note', 64, 0.5, 0.25, 100, 0, 'note', 64, 0.75, 0.25, 100, 0, 'note', 64, 1, 0.25, 100, 0, 'note', 65, 0.25, 0.25, 100, 0, 'note', 68, 1, 0.25, 100, 0, 'note', 69, 0, 0.25, 100, 0, 'note', 69, 0.75, 0.25, 100, 0, 'note', 69, 1.25, 2, 100, 0, 'note', 70, 0.5, 0.25, 100, 0, 'note', 71, 0.25, 0.25, 100, 0, 'done'], 'duration': 4}