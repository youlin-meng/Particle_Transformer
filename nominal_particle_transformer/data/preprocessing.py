import numpy as np
import pandas as pd
import torch
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from config.default_config import FEATURE_MAP

def get_feature_index(name):
    return FEATURE_MAP[name]['index']

def get_features_to_scale(scale_type):
    return [name for name, props in FEATURE_MAP.items() if props['scale_type'] == scale_type]

def convert_to_cartesian(pt, eta, phi, mass, particle_type):
    px = np.abs(pt * np.cos(phi))
    py = np.abs(pt * np.sin(phi))
    pz = np.abs(pt * np.sinh(eta))
    energy = np.sqrt(px**2 + py**2 + pz**2 + mass**2)
    
    return np.array([
        px, py, pz, energy
    ])

def select_particles(data, max_particles, n_channels):
    n_events = len(data['jet_pt_0'])
            
    jet_columns = [col for col in data.columns if col.startswith('jet_pt_')]
    max_jets = len(jet_columns)
    print(f'Max jets: {max_jets}')
    
    particles = np.full((n_events, max_particles, n_channels), -99, dtype=np.float32)
    
    for i in range(n_events):
        features = []
        
        # Leptons (electrons/muons)
        for prefix in ['el', 'mu']:
            _particle_type = 2 if prefix == 'el' else 3
            for j in [0, 1]:
                pt = data[f'{prefix}_pt_{j}'][i]
                if not (pt == -99 or np.isnan(pt)):
                    try:
                        features.append(convert_to_cartesian(
                            pt, 
                            data[f'{prefix}_eta_{j}'][i],
                            data[f'{prefix}_phi_{j}'][i],
                            data[f'{prefix}_mass_{j}'][i],
                            particle_type=_particle_type
                        ))
                    except:
                        features.append(np.array([-99] * n_channels, dtype=np.float32))
                else:
                    features.append(np.array([-99] * n_channels, dtype=np.float32))
        
        # Jets
        for j in range(max_jets):
            pt = data[f'jet_pt_{j}'][i]
            if not (pt == -99 or np.isnan(pt)):
                try:
                    _particle_type = 1 if int(data[f'jet_btag_{j}'][i]) == 1 else 0
                    features.append(convert_to_cartesian(
                        pt,
                        data[f'jet_eta_{j}'][i],
                        data[f'jet_phi_{j}'][i],
                        data[f'jet_mass_{j}'][i],
                        particle_type=_particle_type
                    ))
                except:
                    features.append(np.array([-99] * n_channels, dtype=np.float32))
            else:
                features.append(np.array([-99] * n_channels, dtype=np.float32))
    
        for j, feature in enumerate(features[:max_particles]):
            try:
                particles[i, j] = feature
            except:
                particles[i, j] = np.array([-99] * n_channels, dtype=np.float32)
    
    return particles

def build_input(particles, max_particles, n_channels):
    n_events = particles.shape[0]
    x = np.full((n_events, max_particles, n_channels), -99.0)
    src_mask = np.zeros((n_events, max_particles), dtype=bool)
    
    for i in range(max_particles):
        real_particles = particles[:, i, 0] != -99
        src_mask[:, i] = real_particles
        x[real_particles, i] = particles[real_particles, i, :]
    
    return x, src_mask

def prepare_particle_data(signal_path, background_path, max_particles, test_ratio, max_events, n_channels):
    df_signal = pd.read_hdf(signal_path, key='data')
    df_background = pd.read_hdf(background_path, key='data')
    
    if max_events > 0:
        df_signal = df_signal.sample(n=min(max_events, len(df_signal)), random_state=42)
        df_background = df_background.sample(n=min(max_events, len(df_background)), random_state=42)
        df_signal = df_signal.reset_index(drop=True)
        df_background = df_background.reset_index(drop=True)
    
    signal_particles = select_particles(df_signal, max_particles, n_channels)
    background_particles = select_particles(df_background, max_particles, n_channels)
    
    signal_x, signal_mask = build_input(signal_particles, max_particles, n_channels)
    background_x, background_mask = build_input(background_particles, max_particles, n_channels)
    
    x = np.concatenate([signal_x, background_x])
    mask = np.concatenate([signal_mask, background_mask])
    y = np.concatenate([np.ones(len(signal_x)), np.zeros(len(background_x))])
    x, mask, y = shuffle(x, mask, y, random_state=42)

    x_trainval, x_test, mask_trainval, mask_test, y_trainval, y_test = train_test_split(
        x, mask, y, test_size=test_ratio, random_state=42)
    
    x_train, x_val, mask_train, mask_val, y_train, y_val = train_test_split(
        x_trainval, mask_trainval, y_trainval, 
        test_size=test_ratio, random_state=42)

    def reshape_for_scaling(x, mask):
        x_flat = x.reshape(-1, n_channels)
        mask_flat = mask.reshape(-1)
        return x_flat[mask_flat]
    
    train_real = reshape_for_scaling(x_train, mask_train)
    
    if len(train_real) > 0:
        standard_scaler = StandardScaler()
        standard_features = get_features_to_scale('standard')
        standard_indices = [get_feature_index(name) for name in standard_features]
        
        if standard_indices:
            standard_scaler.fit(train_real[:, standard_indices])
        
        def scale_dataset(x, mask):
            original_shape = x.shape
            x_flat = x.reshape(-1, n_channels)
            mask_flat = mask.reshape(-1)
            real_particles = x_flat[mask_flat]
            
            for name in get_features_to_scale('log'):
                idx = get_feature_index(name)
                real_particles[:, idx] = np.log(real_particles[:, idx] + 1e-8)
            
            if standard_indices:
                real_particles[:, standard_indices] = standard_scaler.transform(
                    real_particles[:, standard_indices])
            
            x_flat[mask_flat] = real_particles
            x_flat[~mask_flat] = -99.0
            return x_flat.reshape(original_shape)
        
        x_train = scale_dataset(x_train, mask_train)
        x_val = scale_dataset(x_val, mask_val)
        x_test = scale_dataset(x_test, mask_test)
    
    def to_tensor(array, dtype):
        return torch.as_tensor(array, dtype=dtype)
    
    return (
        to_tensor(x_train, torch.float32),
        to_tensor(x_val, torch.float32),
        to_tensor(x_test, torch.float32),
        to_tensor(mask_train, torch.bool),
        to_tensor(mask_val, torch.bool),
        to_tensor(mask_test, torch.bool),
        to_tensor(y_train, torch.float32).unsqueeze(1),
        to_tensor(y_val, torch.float32).unsqueeze(1),
        to_tensor(y_test, torch.float32).unsqueeze(1)
    )