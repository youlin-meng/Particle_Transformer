# Default configuration for the particle classifier
DEFAULT_CONFIG = {
    # Data paths
    'signal_path': 'new_df_signal_2lss_full.h5',
    'background_path': 'new_df_background_2lss_full.h5',
    
    # Model architecture
    'model_type': 'transformer',
    'max_particles': 23,
    'n_channels': 4,
    'embed_dim': 64,
    'num_heads': 8,
    'num_transformers': 6,
    'mlp_units': [256, 64],
    'mlp_head_units': [256, 128],
    'dropout_rate': 0.1688683657070016,
    
    # Training
    'max_events': 0,
    'test_ratio': 0.1,
    'val_ratio': 0.1,
    'batch_size': 2048,
    'epochs': 600,
    'learning_rate': 4.36906301650741e-05,
    'patience': 30,
    'use_amp': True,
    
    # Data augmentation
    'augmented_data': False,
    'noise_scale': 0.01,
    
    # System
    'num_workers': 8,
    'prefetch_factor': 4,
    'log_dir': 'runs',
    'log_freq': 5
}

# Feature configuration
FEATURE_MAP = {
    'log(px)': {'index': 0, 'scale_type': 'log'},
    'log(py)': {'index': 1, 'scale_type': 'log'},
    'cos(pz)': {'index': 2, 'scale_type': 'log'},
    'log(e)': {'index': 3, 'scale_type': 'log'}
}