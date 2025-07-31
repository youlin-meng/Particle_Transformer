import numpy as np
import pandas as pd
import torch
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
from sklearn.metrics import roc_curve, auc

def convert_to_cartesian(pt, eta, phi, mass, b_tag):
    
    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)
    energy = np.sqrt(px**2 + py**2 + pz**2 + mass**2)
    # return np.array([px, py, pz, energy, float(jet_tag), float(lep_tag)])
    # return np.array([px, py, pz, energy, float(b_tag)])
    
    # Add log transforms for better scaling
    log_pt = np.log(pt + 1e-8)
    log_energy = np.log(energy + 1e-8)
    
    return np.array([px, py, pz, log_energy])

def select_particles(data, max_particles, n_channels):
    n_events = len(data['jet_pt_0'])
            
    jet_columns = [col for col in data.columns if col.startswith('jet_pt_')]
    max_jets = len(jet_columns)
    
    particles = np.full((n_events, max_particles, n_channels), -99)
    
    for i in range(n_events):
        features = []
        
        # Leptons (electrons/muons)
        for prefix in ['el', 'mu']:
            for j in [0, 1]:
                if data[f'{prefix}_pt_{j}'][i] != -99:
                    features.append(convert_to_cartesian(
                        data[f'{prefix}_pt_{j}'][i], 
                        data[f'{prefix}_eta_{j}'][i],
                        data[f'{prefix}_phi_{j}'][i],
                        data[f'{prefix}_mass_{j}'][i],
                        # particle_type=1 #lepton
                        b_tag=0
                    ))
                else:
                    features.append(np.array([-99] * n_channels))
        
        # Jets
        for j in range(max_jets):
            if data[f'jet_pt_{j}'][i] != -99:
                if int(data[f'jet_btag_{j}'][i]) == 1:
                    _particle_type=3
                    _btag=1
                else:
                    _particle_type=2
                    _btag=0
                features.append(convert_to_cartesian(
                    data[f'jet_pt_{j}'][i],
                    data[f'jet_eta_{j}'][i],
                    data[f'jet_phi_{j}'][i],
                    data[f'jet_mass_{j}'][i],
                    b_tag = _btag
                    # particle_type=_particle_type  # 2 for non-btagged, 3 for btagged
                ))
            else:
                features.append(np.array([-99] * n_channels))
        
        # Sort by pt and store (leaving padding as -99)
        # features.sort(key=lambda x: np.sqrt(x[0]**2 + x[1]**2), reverse=True)
        for j, feature in enumerate(features[:max_particles]):
            particles[i, j] = feature
    
    return particles

def build_input(particles, max_particles, n_channels):
    """Build model input and mask from -99-padded data"""
    n_events = particles.shape[0]
    x = np.full((n_events, max_particles, n_channels), -99.0)  # Initialize with -99
    src_mask = np.zeros((n_events, max_particles), dtype=bool)
    
    for i in range(max_particles):
        # Create mask (True for real particles)
        real_particles = particles[:, i, 0] != -99
        src_mask[:, i] = real_particles
        
        # Copy features for real particles, keep -99 for padding
        x[real_particles, i] = particles[real_particles, i, :]
    
    print(f"Data stats - Real particles: {src_mask.mean():.1%}")
    return x, src_mask

def prepare_particle_data(signal_path, background_path,  max_particles, test_ratio, max_events, n_channels):
    
    df_signal = pd.read_hdf(signal_path, key='data')
    df_background = pd.read_hdf(background_path, key='data')
    
    #------------------------------------------max events--------------------------------------------
    
    if max_events > 0:
        # Randomly sample events while maintaining the DataFrame structure
        df_signal = df_signal.sample(n=min(max_events, len(df_signal)), random_state=42, replace=False)
        df_background = df_background.sample(n=min(max_events, len(df_background)), random_state=42, replace=False)
        
        # Reset index to maintain continuous integer indexing
        df_signal = df_signal.reset_index(drop=True)
        df_background = df_background.reset_index(drop=True)
    
    #------------------------------------------------------------------------------------------------
    
    # Process particles (preserving -99 padding)
    signal_particles = select_particles(df_signal, max_particles, n_channels)
    background_particles = select_particles(df_background, max_particles, n_channels)
    
    # Build inputs and masks
    signal_x, signal_mask = build_input(signal_particles, max_particles, n_channels)
    background_x, background_mask = build_input(background_particles, max_particles, n_channels)
    
    # Combine and shuffle
    x = np.concatenate([signal_x, background_x])
    mask = np.concatenate([signal_mask, background_mask])
    y = np.concatenate([np.ones(len(signal_x)), np.zeros(len(background_x))])
    x, mask, y = shuffle(x, mask, y, random_state=42)
    
    # Scale only real particles (not -99 padding)
    # Create numpy array copies explicitly
    x_np = np.array(x) if not isinstance(x, np.ndarray) else x.copy()
    mask_np = np.array(mask) if not isinstance(mask, np.ndarray) else mask.copy()
    
    # # Get indices of real particles
    # real_indices = np.where(mask_np)
    
    # # Extract real particles for scaling
    # real_particles = x_np[real_indices]
    
    # if len(real_particles) > 0:
    #     print("yes, scaling now")
    #     # Initialize and fit scaler
    #     scaler = StandardScaler()
    #     scaled_particles = scaler.fit_transform(real_particles)
    #     x_np[real_indices] = scaled_particles
    
    # Train-test split
    x_train, x_test, mask_train, mask_test, y_train, y_test = train_test_split(
        x, mask, y, test_size=test_ratio, random_state=42)
    
    # Convert to tensors
    def to_tensor(array, dtype):
        return torch.as_tensor(array, dtype=dtype)
    
    return  (
                to_tensor(x_train, torch.float32),
                to_tensor(x_test, torch.float32),
                to_tensor(mask_train, torch.bool),
                to_tensor(mask_test, torch.bool),
                to_tensor(y_train, torch.float32).unsqueeze(1),
                to_tensor(y_test, torch.float32).unsqueeze(1)
            )

def plot_training_loss_curve(train_losses, val_losses):
    
    plt.figure(figsize=(8, 6))
    
    plt.plot(train_losses, label='Training Loss', linewidth=2, color='r')
    plt.plot(val_losses, label='Validation Loss', linewidth=2, color='b')
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()

    plt.savefig("results/loss_vs_epoch.png", dpi=300)
    plt.close()
    print(f"Loss curve plot saved to results")

def plot_score_distribution(y_true, y_pred):
    """Plot distribution of predicted scores for each class"""
    plt.figure(figsize=(10, 6))
    
    if torch.is_tensor(y_true):
        y_true = y_true.cpu().numpy().flatten()
    if torch.is_tensor(y_pred):
        y_pred = y_pred.cpu().numpy().flatten()
        
    # Save score distribution data
    score_df = pd.DataFrame({
        'y_true': y_true,
        'y_pred': y_pred
    })
    score_df.to_csv("results/score_distribution_data.csv", index=False)
    
    plt.hist(y_pred[y_true == 1], bins=50, alpha=0.5, label='Signal', density=True, histtype="step", color='r')
    plt.hist(y_pred[y_true == 0], bins=50, alpha=0.5, label='Background', density=True, histtype="step", color='b')
    
    plt.xlabel('Predicted Score')
    plt.ylabel('Normalised counts')
    plt.legend()
    plt.savefig("results/normalised_counts_vs_predicted_score.png", dpi=300)
    plt.close()

def plot_roc_curve(y_true, y_pred):
    """Plot ROC curve"""
    
    # Convert tensors to numpy if needed
    if torch.is_tensor(y_true):
        y_true = y_true.cpu().numpy().flatten()
    if torch.is_tensor(y_pred):
        y_pred = y_pred.cpu().numpy().flatten()
    
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    
    # Save ROC data to CSV
    roc_df = pd.DataFrame({
        'fpr': fpr,
        'tpr': tpr,
    })
    roc_df.to_csv("results/roc_curve_data.csv", index=False)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.3f})', color='b')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.savefig("results/ROC_curve.png", dpi=300)
    plt.close()

def plot_feature_attention(x_input, attention_weights, save_path=None):
    
    feature_names = ["log_pt", "eta", "phi", "elog_nergy"]
    
    # Convert to numpy
    x_input = x_input.cpu().numpy() if torch.is_tensor(x_input) else x_input
    attention_weights = attention_weights.cpu().numpy() if torch.is_tensor(attention_weights) else attention_weights
    
    # Mask out padded particles (where px = -99)
    mask = (x_input[:, :, 0] != -99)  # Assuming px is the first feature
    x_input_masked = x_input[mask]
    attention_masked = attention_weights[mask]

    # # Normalize attention weights per event
    # attention_masked = attention_masked / (attention_masked.sum(axis=1, keepdims=True) + 1e-8)
    
    # Compute weighted feature importance
    feature_importance = np.abs(x_input_masked) * attention_masked[:, np.newaxis]
    avg_feature_importance = feature_importance.mean(axis=0)
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.bar(feature_names, avg_feature_importance)
    plt.xlabel("Feature")
    plt.ylabel("Average Attention Contribution")
    plt.xticks(rotation=45)
    
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()

# attention weights for particles
def plot_attention_weights(particles, weights, save_path=None):
    
    # If weights are 3D (batch × particles × heads), average over heads
    if weights.ndim == 3:
        weights = weights.mean(axis=1)  # Now shape: batch × particles
    
    plt.figure(figsize=(10, 5))
    for i in range(min(5, len(particles))):
        plt.subplot(2, 3, i+1)
        plt.bar(range(len(weights[i])), weights[i])
        plt.title(f"Event {i}")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()

def analyze_kinematics_after_cuts(results, n_channels, score_cut=0.05):
    """Analyze kinematics for events passing score cut"""
    
    # Extract data
    y_true = results['y_true']
    y_pred = results['y_pred']
    x_input = results['x_input']  # Shape: [n_events, max_particles, 6]
    
    # Create DataFrame for analysis
    analysis_df = pd.DataFrame({
        'is_signal': y_true,
        'pred_score': y_pred,
        'passes_cut': y_pred < score_cut # now less than 0.05
    })
    
    # Convert Cartesian back to polar coordinates for analysis
    def cartesian_to_polar(features):
        
        log_pt, eta, phi, log_energy = features
        
        # if pt == -99:  # Padding
        #     return np.nan, np.nan, np.nan, np.nan
        
        # pt = np.sqrt(px**2 + py**2)
        # eta = np.arcsinh(pz/pt) if pt > 0 else 0
        # phi = np.arctan2(py, px)
        # mass = np.sqrt(np.abs(energy**2 - (px**2 + py**2 + pz**2))) 
        return log_pt, eta, phi, log_energy
    
    # Analyze leading particle (highest pT) in each event
    leading_particles = []
    for event in x_input:
        # Find leading particle (first non-padded particle)
        for particle in event:
            if particle[0] != -99:  # Not padded
                log_pt, eta, phi, log_energy = cartesian_to_polar(particle)
                leading_particles.append([log_pt, eta, phi, log_energy])
                break
        else:
            leading_particles.append([np.nan]*n_channels)  # All padded
    
    # Add leading particle info to DataFrame
    analysis_df[['lead_log_pt', 'lead_eta', 'lead_phi', 'lead__logenergy']] = leading_particles
    
    # Plot kinematics distributions
    os.makedirs("results/kinematics", exist_ok=True)
    
    for var in ['lead_log_pt', 'lead_eta', 'lead_phi', 'lead_log_energy']:
        plt.figure(figsize=(10, 6))
        
        # Plot signal vs background for events passing cut
        for label, mask in [('Signal', analysis_df['is_signal'] == 1),
                            ('Background', analysis_df['is_signal'] == 0)]:
            data = analysis_df.loc[mask & (analysis_df['passes_cut']), var].dropna()  # Drop NaN values
            if len(data) > 0:  # Only plot if we have data
                plt.hist(data, bins=50, alpha=0.5, label=label, density=True, color='r' if label == 'Signal' else 'b')
        
        if len(plt.gca().patches) > 0:  # Only save if we plotted something
            plt.xlabel(var)
            plt.ylabel('Normalised counts')
            plt.title(f'{var} distribution after score cut < {score_cut}')
            plt.legend()
            plt.savefig(f"results/kinematics/{var}_after_cut.png")
            plt.close()
        else:
            print(f"No data to plot for {var} with cut {score_cut}")
    
    # Save analysis results
    analysis_df.to_csv("results/kinematics/kinematic_analysis.csv", index=False)
    return analysis_df
