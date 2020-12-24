from functools import total_ordering
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


# def main(rank, overfitted, config, num_workers, world_size, model_dir):
#     # === Init process group
#     os.environ['MASTER_ADDR'] = 'localhost'
#     # os.environ['MASTER_PORT'] = '12355'
#     # os.environ['MASTER_PORT'] = '12356'
#     os.environ['MASTER_PORT'] = '12357'
#     dist.init_process_group(backend='nccl', world_size=world_size, rank=rank)
#     torch.cuda.set_device(rank)
#     device = f'cuda:{rank}'

#     # === Decoder ====
#     # dataloader generator
#     dataloader_generator = get_dataloader_generator(
#         dataset=config['dataset'],
#         dataloader_generator_kwargs=config['dataloader_generator_kwargs'])

#     # data processor
#     global data_processor
#     data_processor = get_source_target_data_processor(
#         dataloader_generator=dataloader_generator,
#         data_processor_type=config['data_processor_type'],
#         data_processor_kwargs=config['data_processor_kwargs'])

#     # positional embedding
#     positional_embedding_source: PositionalEmbedding = get_positional_embedding(
#         dataloader_generator=dataloader_generator,
#         positional_embedding_dict=config['positional_embedding_source_dict'])
#     positional_embedding_target: PositionalEmbedding = get_positional_embedding(
#         dataloader_generator=dataloader_generator,
#         positional_embedding_dict=config['positional_embedding_target_dict'])
#     # sos embedding
#     sos_embedding = get_sos_embedding(
#         dataloader_generator=dataloader_generator,
#         sos_embedding_dict=config['sos_embedding_dict'])
#     encoder_decoder = get_encoder_decoder(
#         data_processor=data_processor,
#         dataloader_generator=dataloader_generator,
#         positional_embedding_source=positional_embedding_source,
#         positional_embedding_target=positional_embedding_target,
#         sos_embedding=sos_embedding,
#         encoder_decoder_type=config['encoder_decoder_type'],
#         encoder_decoder_kwargs=config['encoder_decoder_kwargs'],
#         training_phase=False)

#     encoder_decoder.to(device)
#     encoder_decoder = DistributedDataParallel(
#         module=encoder_decoder,
#         device_ids=[rank],
#         output_device=rank,
#         #   find_unused_parameters=True
#     )

#     global handler
#     handler = EncoderDecoderHandler(model=encoder_decoder,
#                                     model_dir=model_dir,
#                                     dataloader_generator=dataloader_generator)
#     # Load model
#     if overfitted:
#         handler.load(early_stopped=False)
#     else:
#         handler.load(early_stopped=True)

#     # debug_test()

#     local_only = False
#     if local_only:
#         # accessible only locally:
#         app.run(threaded=True)
#     else:
#         # accessible from outside:
#         app.run(host='0.0.0.0', port=80, threaded=True)


def main(rank, overfitted, config, num_workers, world_size, model_dir):
    # === Init process group
    os.environ['MASTER_ADDR'] = 'localhost'
    # os.environ['MASTER_PORT'] = '12355'
    # os.environ['MASTER_PORT'] = '12356'
    os.environ['MASTER_PORT'] = '12357'
    dist.init_process_group(backend='nccl', world_size=world_size, rank=rank)
    torch.cuda.set_device(rank)
    device = f'cuda:{rank}'

    local_only = False
    if local_only:
        # accessible only locally:
        app.run(threaded=True)
    else:
        # accessible from outside:
        app.run(host='0.0.0.0', port=80, threaded=True)


@app.route('/ping', methods=['GET'])
def ping():
    return 'pong'

@app.route('/inpaint', methods=['GET'])
def inpaint():
    global handler
    scores = handler.inpaint(x=x,
                             masked_positions=masked_positions,
                             temperature=1.,
                             top_p=0.95,
                             top_k=0)
    return scores


@app.route('/invocations', methods=['POST'])
def invocations():
    d = json.loads(request.data)
    print(d)
    num_max_generated_events = 15
    notes = d['notes']
    selected_region = d['selected_region']
    x, (event_start, event_end) = ableton_to_tensor(notes, selected_region)

    original_size = x.size(0)
    # add batch_dim
    global data_processor
    # TODO different number of batches?
    num_examples = 1
    x = x.unsqueeze(0).repeat(num_examples, 1, 1)
    _, x, _ = data_processor.preprocess(x)

    # TODO case where x is bigger than 1024
    # TODO add num_notes
    # TODO exclude pad and start symbols
    # TODO PAD BEFORE
    # TODO compute state before calling inpaint_region
    # TODO high pitched notes are discarded
    # TODO tempo
    # TODO send track duration
    # TODO pb when selection out of bounds
    # TODO not zeros after
    # add correct size
    x = torch.cat([
        x,
        torch.zeros(num_examples, 1024 - x.size(1), 4).long().to(x.device)
    ],
                  dim=1)

    masked_positions = torch.zeros_like(x).long()
    # TODO remove mask end
    # masked_positions[:, event_start:event_end] = 1
    masked_positions[:, event_start:] = 1


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
        start_event=event_start,
        end_event=event_end_region,
        temperature=1.,
        top_p=0.95,
        top_k=0)
    # output = x.detach().cpu()

    ableton_notes, ableton_notes_region, track_duration = tensor_to_ableton(
        output[0, :original_size],
        clip_start=d['clip_start'],
        start_event=event_start,
        end_event=event_end_region)

    # _, region_start = tensor_to_ableton(
    #     beginning[0],
    #     clip_start=d['clip_start']
    # )

    # region_ableton_notes, _ = tensor_to_ableton(
    #     region[0],    clip_start=region_start

    # )

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
        'clip_id': d['clip_id'],
        'clip_start': d['clip_start'],
        'clip_end': d['clip_end'],
        'detail_clip_id': d['detail_clip_id']
    }
    return jsonify(d)


@app.route('/continue', methods=['GET', 'POST'])
def update():
    d = json.loads(request.data)
    print(d)
    num_max_generated_events = 30
    json_notes = d['notes']
    x = json_to_tensor(json_notes)
    event_start = d['next_event_start']
    event_end = d['next_event_end']

    original_size = x.size(0)
    # add batch_dim
    global data_processor
    # TODO different number of batches?
    num_examples = 1
    x = x.unsqueeze(0).repeat(num_examples, 1, 1)
    _, x, _ = data_processor.preprocess(x)

    # TODO case where x is bigger than 1024
    # TODO add num_notes
    # TODO exclude pad and start symbols
    # TODO PAD BEFORE
    # TODO compute state before calling inpaint_region
    # TODO high pitched notes are discarded
    # TODO tempo
    # TODO send track duration
    # TODO pb when selection out of bounds
    # TODO not zeros after
    # add correct size
    x = torch.cat([
        x,
        torch.zeros(num_examples, 1024 - x.size(1), 4).long().to(x.device)
    ],
                  dim=1)

    masked_positions = torch.zeros_like(x).long()
    # TODO remove mask end
    # masked_positions[:, event_start:event_end] = 1
    masked_positions[:, event_start:] = 1
    # masked_positions[:, event_start:event_end] = 1


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
        start_event=event_start,
        end_event=event_end_region,
        temperature=1.,
        top_p=0.95,
        top_k=0)
    # output = x.detach().cpu()

    ableton_notes, ableton_notes_region, track_duration = tensor_to_ableton(
        output[0, :original_size],
        clip_start=d['clip_start'],
        start_event=event_start,
        end_event=event_end_region)

    # _, region_start = tensor_to_ableton(
    #     beginning[0],
    #     clip_start=d['clip_start']
    # )

    # region_ableton_notes, _ = tensor_to_ableton(
    #     region[0],    clip_start=region_start

    # )

    print(f'albeton notes: {ableton_notes}')
    print(f'region start: {ableton_notes_region}')
    d = {
        'id': d['id'],
        'notes': ableton_notes,
        'track_duration': track_duration,
        'done': done,
        'next_event_start': event_end_region,
        'next_event_end': event_end,
        'notes_region': ableton_notes_region,
        'clip_id': d['clip_id'],
        'clip_start': d['clip_start'],
        'clip_end': d['clip_end'],
        'detail_clip_id': d['detail_clip_id']
    }
    return jsonify(d)

def json_to_tensor(json_note_list):
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
    # TODO remove debug
    print(f'Max pitch: {d["pitch"].max()}')
    
    # multiply by tempo
    tempo = 0.5  # 120 bpm
    # tempo = 1  # absolute timing?
    d['time'] = d['time'] * tempo
    d['duration'] = d['duration'] * tempo

    # compute time_shift
    d['time_shift'] = torch.cat(
        [d['time'][1:] - d['time'][:-1],
         torch.zeros(1, )], dim=0)

    global handler

    # to numpy :(
    d = {k: t.numpy() for k, t in d.items()}
    sequence_dict = handler.dataloader_generator.dataset.tokenize(d)
    # to pytorch :)
    sequence_dict = {k: torch.LongTensor(t) for k, t in sequence_dict.items()}

    x = torch.stack(
        [sequence_dict[e] for e in handler.dataloader_generator.features],
        dim=-1).long()
    return x

def ableton_to_tensor(ableton_note_list, selected_region=None):
    d = {
        'pitch': [],
        'time': [],
        'duration': [],
        'velocity': [],
        'muted': [],
    }
    mod = -1

    # Todo remove debug
    flag = False
    i = 0
    for n in ableton_note_list:
        if flag:
            flag = False
            # print(f'{n}', sep=' ')
        if n == 'note':
            flag = True
            i += 1
    print(f'total: {i}')

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
    # TODO remove debug
    print(f'Max pitch: {d["pitch"].max()}')

    # compute start_event, end_event
    epsilon = 1e-4
    if selected_region is not None:
        i = 0
        flag = True
        while flag:
            if d['time'][i].item() >= start_time - epsilon:
                flag = False
                event_start = i  # TODO check
            else:
                i = i + 1

        i = 0
        flag = True
        while flag:
            if i > d['time'].size(0):
                flag = False
                event_end = i
            elif d['time'][i].item() >= end_time - epsilon:
                flag = False
                event_end = i  # TODO check
            else:
                i = i + 1

        # _, min_indices = torch.min((d['time'] > start_time).int(), dim=0)
        # event_start = min_indices.max().item()

        # _, max_indices = torch.max((d['time'] < end_time).int(), dim=0)
        # event_end = max_indices.min().item()
    else:
        event_start, event_end = None, None

    # multiply by tempo
    tempo = 0.5  # 120 bpm
    # tempo = 1  # absolute timing?
    d['time'] = d['time'] * tempo
    d['duration'] = d['duration'] * tempo

    # compute time_shift
    d['time_shift'] = torch.cat(
        [d['time'][1:] - d['time'][:-1],
         torch.zeros(1, )], dim=0)

    global handler

    # to numpy :(
    d = {k: t.numpy() for k, t in d.items()}
    sequence_dict = handler.dataloader_generator.dataset.tokenize(d)
    # to pytorch :)
    sequence_dict = {k: torch.LongTensor(t) for k, t in sequence_dict.items()}

    x = torch.stack(
        [sequence_dict[e] for e in handler.dataloader_generator.features],
        dim=-1).long()
    return x, (event_start, event_end)


def tensor_to_ableton(tensor, clip_start, start_event, end_event):
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
    tempo = 2  # 120 bpm
    # tempo = 1 # absolute time
    tensor = tensor.detach().cpu()
    global handler
    index2value = handler.dataloader_generator.dataset.index2value
    num_events, num_channels = tensor.size()
    timeshifts = torch.FloatTensor(
        [index2value['time_shift'][ts.item()] for ts in tensor[:, 3]])
    time = torch.cumsum(timeshifts, dim=0)
    time = (torch.cat([torch.zeros(
        (1, )), time[:-1]], dim=0) * tempo + clip_start)
    for i in range(num_events):
        note = dict(pitch=index2value['pitch'][tensor[i, 0].item()],
                    time=time[i].item(),
                    duration=index2value['duration'][tensor[i, 2].item()] *
                    tempo,
                    velocity=index2value['velocity'][tensor[i, 1].item()],
                    muted=0)
        notes.append(note)
        if start_event <= i < end_event:
            notes_region.append(note)

    track_duration = time[-1].item() + notes[-1]['duration'].item()
    return notes, notes_region, track_duration


def debug_test():
    {
        'id':
        '45',
        'notes': [
            'notes', 8, 'note', 62, 1.25, 0.25, 95, 0, 'note', 64, 0, 0.25,
            100, 0, 'note', 66, 0.5, 0.25, 95, 0, 'note', 67, 1, 0.25, 95, 0,
            'note', 68, 0.75, 0.25, 95, 0, 'note', 70, 0.25, 0.25, 95, 0,
            'note', 70, 1.5, 0.25, 95, 0, 'note', 72, 0.5, 0.25, 95, 0, 'done'
        ],
        'selected_region': {
            'start': 0.5,
            'end': 1.25
        }
    }
    notes = d['notes']
    selected_region = d['selected_region']
    x, (event_start, event_end) = ableton_to_tensor(notes, selected_region)

    print(x)
    print(event_start, event_end)


if __name__ == "__main__":
    launcher()

# Response format
# {'id': '14', 'notes': ['notes', 10, 'note', 64, 0.5, 0.25, 100, 0, 'note', 64, 0.75, 0.25, 100, 0, 'note', 64, 1, 0.25, 100, 0, 'note', 65, 0.25, 0.25, 100, 0, 'note', 68, 1, 0.25, 100, 0, 'note', 69, 0, 0.25, 100, 0, 'note', 69, 0.75, 0.25, 100, 0, 'note', 69, 1.25, 2, 100, 0, 'note', 70, 0.5, 0.25, 100, 0, 'note', 71, 0.25, 0.25, 100, 0, 'done'], 'duration': 4}