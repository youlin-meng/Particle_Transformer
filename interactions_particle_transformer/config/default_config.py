DEFAULT_CONFIG = {
    # Data paths
    'signal_path': 'diego_VBF_sig.h5',
    'background_path': 'diego_QCD_bkg.h5',
    
    # Model architecture
    'model_type': 'transformer',
    'max_particles': 10,
    'n_channels': 6,
    'pairwise_channels': 4,
    'embed_dim': 128,
    'num_heads': 8,
    'num_particle_blocks': 8,
    'num_class_blocks': 2,
    'mlp_ratio': 4,
    'dropout_rate': 0.1,
    'interaction_mode': 'concat', # 'sum' or 'concat'
    
    # Training
    'max_events': 0,
    'test_ratio': 0.1,
    'batch_size': 2048,
    'epochs': 600,
    'learning_rate': 4.36906301650741e-05,
    'patience': 30,
    'use_amp': True,
    
    # System
    'num_workers': 8,
    'prefetch_factor': 4,
    'log_dir': 'runs',
    'log_freq': 5
}

PARTICLE_FEATURES = {
    'log(px)': {'index': 0, 'scale_type': 'log'},
    'log(py)': {'index': 1, 'scale_type': 'log'},
    'cos(pz)': {'index': 2, 'scale_type': 'log'},
    'log(e)': {'index': 3, 'scale_type': 'log'},
    'btag': {'index': 4, 'scale_type': 'none'},
    'particle_type': {'index': 5, 'scale_type': 'none'}
}

INTERACTION_FEATURES = {
    'delta_eta': {'index': 0, 'scale_type': 'none'}, 
    'delta_phi': {'index': 1, 'scale_type': 'none'}, 
    'delta_R': {'index': 2, 'scale_type': 'none'},
    'invariant_mass': {'index': 3, 'scale_type': 'none'}
}