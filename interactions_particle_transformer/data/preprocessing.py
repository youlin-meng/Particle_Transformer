import numpy as np
import pandas as pd
import torch
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from config.default_config import PARTICLE_FEATURES, INTERACTION_FEATURES

def get_feature_index(name, feature_map):
    return feature_map[name]['index']

def get_features_to_scale(scale_type, feature_map):
    return [name for name, props in feature_map.items() if props['scale_type'] == scale_type]

def convert_to_features(e, eta, phi, pt, btag, charge, particle_type):
    """Convert particle features to our format"""
    if np.isnan(pt):
        return np.array([-99]*len(PARTICLE_FEATURES))
    
    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)
    
    return np.array([px, py, pz, e, float(btag), float(particle_type)])

def compute_interaction_features(particles, pairwise_channels):
    batch_size, n_particles, _ = particles.shape
    interactions = np.full((batch_size, n_particles, n_particles, pairwise_channels), -99.0)
    eps = 1e-12

    for i in range(batch_size):
        real_mask = ~(particles[i,:,0] == -99)
        real_indices = np.where(real_mask)[0]
        
        for j in real_indices:
            for k in real_indices:
                if j == k:
                    continue
                
                px1, py1, pz1, e1 = particles[i,j,:4]
                px2, py2, pz2, e2 = particles[i,k,:4]
                
                pt1 = np.sqrt(px1**2 + py1**2 + eps)
                pt2 = np.sqrt(px2**2 + py2**2 + eps)
                
                eta1 = np.arcsinh(pz1/pt1) if pt1 > 0 else 0
                eta2 = np.arcsinh(pz2/pt2) if pt2 > 0 else 0
                delta_eta = eta1 - eta2
                
                phi1 = np.arctan2(py1, px1)
                phi2 = np.arctan2(py2, px2)
                delta_phi = np.arctan2(np.sin(phi1 - phi2), np.cos(phi1 - phi2))
                delta_R = np.sqrt(delta_eta**2 + delta_phi**2)
                
                px_sum = px1 + px2
                py_sum = py1 + py2
                pz_sum = pz1 + pz2
                e_sum = e1 + e2
                m2 = max(0, e_sum**2 - (px_sum**2 + py_sum**2 + pz_sum**2))
                m = np.sqrt(m2) if m2 > 0 else 0

                interactions[i,j,k] = [delta_eta, delta_phi, delta_R, m]
    
    return interactions

def select_particles(data, max_particles, n_channels, pairwise_channels):
    n_events = len(data)
    particles = np.full((n_events, max_particles, n_channels), -99)
    
    for i in range(n_events):
        features = []
        for j in range(max_particles):
            pt = data[f'pt_{j}'][i]
            if not (pt == -99 or np.isnan(pt)):
                try:
                    features.append(convert_to_features(
                        data[f'e_{j}'][i],
                        data[f'eta_{j}'][i],
                        data[f'phi_{j}'][i],
                        pt,
                        data[f'btag_{j}'][i],
                        data[f'charge_{j}'][i],
                        data[f'type_{j}'][i]
                    ))
                except:
                    features.append(np.array([-99] * n_channels, dtype=np.float32))
        
        for j, feature in enumerate(features[:max_particles]):
            particles[i, j] = feature
            
    interaction_features = compute_interaction_features(particles, pairwise_channels)
    
    return particles, interaction_features

def build_input(particles, interaction_features, max_particles, n_channels, pairwise_channels):
    n_events = particles.shape[0]
    x = np.full((n_events, max_particles, n_channels), -99.0)
    u = np.full((n_events, max_particles, max_particles, pairwise_channels), -99.0)
    src_mask = np.zeros((n_events, max_particles), dtype=bool)
    
    for i in range(max_particles):
        real_particles = particles[:, i, 0] != -99
        src_mask[:, i] = real_particles
        x[real_particles, i] = particles[real_particles, i, :]
    
    for i in range(min(max_particles, interaction_features.shape[1])):
        for j in range(min(max_particles, interaction_features.shape[2])):
            real_interactions = interaction_features[:, i, j, 0] != -99
            u[real_interactions, i, j] = interaction_features[real_interactions, i, j]
    
    return x, u, src_mask

def reshape_for_scaling(data, mask):
    if data.ndim == 4:
        if torch.is_tensor(mask):
            mask = mask.cpu().numpy()
        interaction_mask = mask[:, :, np.newaxis] & mask[:, np.newaxis, :]
        data_flat = data.reshape(-1, data.shape[-1])
        mask_flat = interaction_mask.reshape(-1)
    else:
        data_flat = data.reshape(-1, data.shape[-1])
        if torch.is_tensor(mask):
            mask = mask.cpu().numpy()
        mask_flat = mask.reshape(-1)
    return data_flat[mask_flat]

def scale_dataset(data, mask, feature_map, scaler=None, is_particle=True):
    """Apply scaling (log + standard) in a single pass."""
    original_shape = data.shape
    data_flat = data.reshape(-1, data.shape[-1])
    
    if torch.is_tensor(mask):
        mask_np = mask.cpu().numpy()
    else:
        mask_np = mask
    
    if data.ndim == 4:
        interaction_mask = mask_np[:, :, np.newaxis] & mask_np[:, np.newaxis, :]
        mask_flat = interaction_mask.reshape(-1)
    else:
        mask_flat = mask_np.reshape(-1)
    
    real_data = data_flat[mask_flat].copy()
    
    # Log scaling
    log_indices = [get_feature_index(name, feature_map) for name in get_features_to_scale('log', feature_map)]
    for idx in log_indices:
        real_data[:, idx] = np.log(np.abs(real_data[:, idx]) + 1e-8)
    
    # Standard scaling
    standard_indices = [get_feature_index(name, feature_map) for name in get_features_to_scale('standard', feature_map)]
    if scaler and standard_indices:
        real_data[:, standard_indices] = scaler.transform(real_data[:, standard_indices])
    
    data_flat[mask_flat] = real_data
    data_flat[~mask_flat] = -99.0
    return data_flat.reshape(original_shape)

def prepare_particle_data(signal_path, background_path, max_particles, test_ratio, max_events, n_channels, pairwise_channels):
    df_signal = pd.read_hdf(signal_path, key='data')
    df_background = pd.read_hdf(background_path, key='data')
    
    if max_events > 0:
        df_signal = df_signal.sample(n=min(max_events, len(df_signal)), random_state=42)
        df_background = df_background.sample(n=min(max_events, len(df_background)), random_state=42)
    
    signal_particles, signal_interactions = select_particles(df_signal, max_particles, n_channels, pairwise_channels)
    background_particles, background_interactions = select_particles(df_background, max_particles, n_channels, pairwise_channels)
    
    signal_x, signal_u, signal_mask = build_input(signal_particles, signal_interactions, max_particles, n_channels, pairwise_channels)
    background_x, background_u, background_mask = build_input(background_particles, background_interactions, max_particles, n_channels, pairwise_channels)
    
    x = np.concatenate([signal_x, background_x])
    u = np.concatenate([signal_u, background_u])
    mask = np.concatenate([signal_mask, background_mask])
    y = np.concatenate([np.ones(len(signal_x)), np.zeros(len(background_x))])
    x, u, mask, y = shuffle(x, u, mask, y, random_state=42)

    x_trainval, x_test, u_trainval, u_test, mask_trainval, mask_test, y_trainval, y_test = train_test_split(
        x, u, mask, y, test_size=test_ratio, random_state=42)

    x_train, x_val, u_train, u_val, mask_train, mask_val, y_train, y_val = train_test_split(
        x_trainval, u_trainval, mask_trainval, y_trainval, test_size=0.2, random_state=42)

    scalers = {}
    
    train_real_particles = reshape_for_scaling(x_train, mask_train)
    standard_particle_indices = [get_feature_index(name, PARTICLE_FEATURES) for name in get_features_to_scale('standard', PARTICLE_FEATURES)]
    if standard_particle_indices and len(train_real_particles) > 0:
        particle_scaler = StandardScaler()
        particle_scaler.fit(train_real_particles[:, standard_particle_indices])
        scalers['particle_scaler'] = particle_scaler

    train_real_interactions = reshape_for_scaling(u_train, mask_train)
    standard_interaction_indices = [get_feature_index(name, INTERACTION_FEATURES) for name in get_features_to_scale('standard', INTERACTION_FEATURES)]
    if standard_interaction_indices and len(train_real_interactions) > 0:
        interaction_scaler = StandardScaler()
        interaction_scaler.fit(train_real_interactions[:, standard_interaction_indices])
        scalers['interaction_scaler'] = interaction_scaler

    x_train = scale_dataset(x_train, mask_train, PARTICLE_FEATURES, scalers.get('particle_scaler'))
    x_val = scale_dataset(x_val, mask_val, PARTICLE_FEATURES, scalers.get('particle_scaler'))
    x_test = scale_dataset(x_test, mask_test, PARTICLE_FEATURES, scalers.get('particle_scaler'))
    
    u_train = scale_dataset(u_train, mask_train, INTERACTION_FEATURES, scalers.get('interaction_scaler'), is_particle=False)
    u_val = scale_dataset(u_val, mask_val, INTERACTION_FEATURES, scalers.get('interaction_scaler'), is_particle=False)
    u_test = scale_dataset(u_test, mask_test, INTERACTION_FEATURES, scalers.get('interaction_scaler'), is_particle=False)
    
    def to_tensor(array, dtype):
        return torch.as_tensor(array, dtype=dtype)
    
    return (
        to_tensor(x_train, torch.float32),
        to_tensor(x_val, torch.float32),
        to_tensor(x_test, torch.float32),
        to_tensor(u_train, torch.float32),
        to_tensor(u_val, torch.float32),
        to_tensor(u_test, torch.float32),
        to_tensor(mask_train, torch.bool),
        to_tensor(mask_val, torch.bool),
        to_tensor(mask_test, torch.bool),
        to_tensor(y_train, torch.float32).unsqueeze(1),
        to_tensor(y_val, torch.float32).unsqueeze(1),
        to_tensor(y_test, torch.float32).unsqueeze(1)
    )
