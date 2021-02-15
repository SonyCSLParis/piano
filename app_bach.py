import json
from flask import Flask
from flask import request
from flask.helpers import make_response
from flask.json import JSONDecoder, jsonify
from torch.utils import data
from flask_cors import CORS
from music21 import musicxml, converter
import subprocess
from DatasetManager.helpers import START_SYMBOL, PAD_SYMBOL, END_SYMBOL
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
from BFT.utils import cuda_variable
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
    global dataloader_generator
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
    global data_processor
    global dataloader_generator

    # === Parse request ===
    NUM_MIDI_TICKS_IN_SIXTEENTH_NOTE = 120
    start_tick_selection = int(
        float(request.form['start_tick']) / NUM_MIDI_TICKS_IN_SIXTEENTH_NOTE)
    end_tick_selection = int(
        float(request.form['end_tick']) / NUM_MIDI_TICKS_IN_SIXTEENTH_NOTE)
    file_path = request.form['file_path']
    root, ext = os.path.splitext(file_path)
    dir = os.path.dirname(file_path)
    assert ext == '.mxl'
    xml_file = f'{root}.xml'

    # if no selection REGENERATE and set chorale length
    if start_tick_selection == 0 and end_tick_selection == 0:
        generated_sheet = compose_from_scratch()
        generated_sheet.write('xml', xml_file)
        return sheet_to_response(generated_sheet)
    else:
        # --- Parse request---
        # Old method: does not work because the MuseScore plugin does not export to xml but only to compressed .mxl
        # with tempfile.NamedTemporaryFile(mode='wb', suffix='.xml') as file:
        #     print(file.name)
        #     xml_string = request.form['xml_string']
        #     file.write(xml_string)
        #     music21_parsed_chorale = converter.parse(file.name)

        # file_path points to an mxl file: we extract it
        subprocess.run(f'unzip -o {file_path} -d  {dir}', shell=True)
        music21_parsed_chorale = converter.parse(xml_file)

        _tensor_sheet, _tensor_metadata = dataloader_generator.dataset.transposed_score_and_metadata_tensors(
            music21_parsed_chorale, semi_tone=0)

        start_voice_index = int(request.form['start_staff'])
        end_voice_index = int(request.form['end_staff']) + 1

        time_index_range_ticks = [start_tick_selection, end_tick_selection]

        region_length = end_tick_selection - start_tick_selection

    original_size = _tensor_sheet.size(1)
    (x_beginning, x, x_end), masked_positions, offset, padding_length = preprocess_input(
        _tensor_sheet, start_tick_selection, end_tick_selection, start_voice_index,
        end_voice_index)

    global handler

    output = handler.inpaint_region_optimized(
        x=x,
        masked_positions=masked_positions,
        start_event=start_tick_selection - offset,
        end_event=end_tick_selection - offset,
        temperature=1.,
        top_p=0.95,
        top_k=0)

    # join (x_beginning, x, x_end)
    new_x = torch.cat(
        [
            x_beginning[0].detach().cpu(),
            output[0].detach().cpu(),
            x_end[0].detach().cpu()
        ],
        dim=0)[padding_length: padding_length + original_size]  # removes padding and end symbols
    
    new_x = new_x.transpose(1, 0)
    output_sheet = dataloader_generator.dataset.tensor_to_score(tensor_score=new_x)
    
    output_sheet.write('xml', '/tmp/deepbach.xml')
    return 'ok'

 


def sheet_to_response(sheet):
    # convert sheet to xml
    goe = musicxml.m21ToXml.GeneralObjectExporter(sheet)
    xml_chorale_string = goe.parse()

    response = make_response((xml_chorale_string, response_headers))
    return response


def preprocess_input(x, start_tick_selection, end_tick_selection,
                     start_voice_index, end_voice_index):
    """

    Args:
        x ([type]): original sequence (num_voices, num_events )
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
    x = torch.transpose(x, 0, 1)
    x = x.unsqueeze(0).repeat(num_examples, 1, 1)
    _, x, _ = data_processor.preprocess(x)

    total_length = x.size(1)
    
    num_events_model = handler.dataloader_generator.sequences_size
    
    # we overpad
    padding_length = num_events_model // 2 
    x = add_start_end_pad_symbols(x, event_start=-padding_length,
                              num_events_after_padding=total_length + 2 * padding_length
                              )
    
    # slice
    slice_begin = start_tick_selection # i.e. (start_tick_selection + padding_length) - padding_length
    slice_end = slice_begin + num_events_model

    x_beginning = x[:, :slice_begin]
    x_end = x[:, slice_end:]
    x = x[:, slice_begin:slice_end]

    
    masked_positions = torch.zeros_like(x).long()
    masked_positions[:, start_tick_selection + padding_length - slice_begin:
        end_tick_selection + padding_length - slice_begin,
        start_voice_index: end_voice_index
        ] = 1

    offset = slice_begin - padding_length
    return (x_beginning, x, x_end), masked_positions, offset, padding_length

def add_start_end_pad_symbols(x, event_start, num_events_after_padding):
    """[summary]

    Args:
        x ([batch_size, num_events, num_channels]): [description]
        event_start ([type]): [description]
        event_end ([type]): [description]
    """
    global dataloader_generator
    batch_size, num_events, num_channels = x.size()
    start_tokens = cuda_variable(torch.LongTensor([
            dataloader_generator.dataset.note2index_dicts[feature]
            [START_SYMBOL] for feature in dataloader_generator.features
        ]))
    end_tokens = cuda_variable(torch.LongTensor([
            dataloader_generator.dataset.note2index_dicts[feature]
            [END_SYMBOL] for feature in dataloader_generator.features
        ]))
    pad_tokens = cuda_variable(torch.LongTensor([
            dataloader_generator.dataset.note2index_dicts[feature]
            [PAD_SYMBOL] for feature in dataloader_generator.features
        ]))
        
    if event_start < 0:
        before_padding_length = - event_start
        x = torch.cat(
            [ pad_tokens.unsqueeze(0).unsqueeze(0).repeat(
                batch_size,
                before_padding_length - 1,
                1),
            start_tokens.unsqueeze(0).unsqueeze(0).repeat(
                batch_size,
                1,
                1),
            x
             ],
            dim=1
        )
        
    end_padding_length = num_events_after_padding - x.size(1)
    if end_padding_length > 0:
        x = torch.cat(
            [ 
             x,
             end_tokens.unsqueeze(0).unsqueeze(0).repeat(
                batch_size,
                1,
                1),
            pad_tokens.unsqueeze(0).unsqueeze(0).repeat(
                batch_size,
                end_padding_length - 1,
                1),
             ],
            dim=1
        )
    return x
        
    
    
if __name__ == "__main__":
    launcher()

# Response format
# {'id': '14', 'notes': ['notes', 10, 'note', 64, 0.5, 0.25, 100, 0, 'note', 64, 0.75, 0.25, 100, 0, 'note', 64, 1, 0.25, 100, 0, 'note', 65, 0.25, 0.25, 100, 0, 'note', 68, 1, 0.25, 100, 0, 'note', 69, 0, 0.25, 100, 0, 'note', 69, 0.75, 0.25, 100, 0, 'note', 69, 1.25, 2, 100, 0, 'note', 70, 0.5, 0.25, 100, 0, 'note', 71, 0.25, 0.25, 100, 0, 'done'], 'duration': 4}