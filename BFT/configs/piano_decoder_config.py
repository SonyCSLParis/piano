from pathlib import Path

config = {
    'training_method':             'decoder',
    'dataset':                     'piano',

    # --- Dataloader ---
    'dataloader_generator_kwargs': dict(
        # sequences_size=240,
        sequences_size=1024,
        transformations={
            'time_dilation':  True,
            'velocity_shift': True,
            'transposition':  True
        }
    ),  # Can be different from the encoder's data loader

    # --- DataProcessor ---
    'data_processor_type':         'piano',  # can be used to filter out some channels
    'data_processor_kwargs':       dict(
        embedding_size=32,
    ),  # Can be different from the encoder's data processor

    # --- Positional Embedding ---
    'positional_embedding_dict': dict(
        # 'channel_embedding': dict()
        sinusoidal_embedding= dict(
            positional_embedding_size=128,
            num_channels=4,
            dropout=0.2
        ),
        sinusoidal_elapsed_time_embedding=dict(
            positional_embedding_size=128,
            num_channels=4,
            dropout=0.2
        ),
        channel_embedding=dict(
            positional_embedding_size=12,
            num_channels=4
        )
    ),
        
    # --- Decoder ---
    'decoder_type':                'linear_transformer',
    'decoder_kwargs':              dict(
        d_model=512,
        n_head=8,
        num_decoder_layers=8,
        dim_feedforward=1024,
        dropout=0.1,
        label_smoothing=True
    ),
    # ======== Training ========
    'lr':                          1e-4,
    'batch_size':                  4,
    'num_batches':                 256,
    'num_epochs':                  2000,

    # ======== model ID ========
    'timestamp':                   None,
    'savename':                    Path(__file__).stem,
}
