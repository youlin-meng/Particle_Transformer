import numpy as np
import pandas as pd
import torch
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os

def convert_to_cartesian(pt, eta, phi, mass, is_bjet=0, is_jet=0):
    if pt == -99:  # Pad with -99
        return np.array([-99, -99, -99, -99, -99, -99])
    
    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)
    energy = np.sqrt(px**2 + py**2 + pz**2 + mass**2)
    return np.array([px, py, pz, energy, float(is_bjet), float(is_jet)])

def select_particles(data, max_particles=13):
    """Select particles while preserving original -99 padding"""
    n_events = len(data['el_pt_0'])
    particles = np.full((n_events, max_particles, 6), -99)  # Initialize with -99
    
    for i in range(n_events):
        features = []
        
        # Leptons (electrons/muons)
        for prefix in ['el', 'mu']:
            for j in [0, 1]:
                pt_key = f'{prefix}_pt_{j}'
                if data[pt_key][i] != -99:
                    mass = 0.000511 if prefix == 'el' else 0.105658
                    features.append(convert_to_cartesian(
                        data[pt_key][i], 
                        data[f'{prefix}_eta_{j}'][i],
                        data[f'{prefix}_phi_{j}'][i],
                        mass,
                        is_bjet=0,
                        is_jet=0
                    ))
        
        # Jets (with b-tagging)
        for j in range(18):
            pt_key = f'jet_pt_{j}'
            if data[pt_key][i] != -99:
                features.append(convert_to_cartesian(
                    data[pt_key][i],
                    data[f'jet_eta_{j}'][i],
                    data[f'jet_phi_{j}'][i],
                    data[f'jet_mass_{j}'][i],
                    is_bjet=data.get(f'jet_btag_{j}', [0]*n_events)[i],
                    is_jet=1
                ))
        
        # Sort by pt and store (leaving padding as -99)
        features.sort(key=lambda x: np.sqrt(x[0]**2 + x[1]**2), reverse=True)
        for j, feat in enumerate(features[:max_particles]):
            particles[i, j] = feat
    
    return particles

def build_input(particles, max_particles=13):
    """Build model input and mask from -99-padded data"""
    n_events = particles.shape[0]
    x = np.full((n_events, max_particles, 6), -99.0)  # Initialize with -99
    src_mask = np.zeros((n_events, max_particles), dtype=bool)
    
    for i in range(max_particles):
        # Create mask (True for real particles)
        real_particles = particles[:, i, 0] != -99
        src_mask[:, i] = real_particles
        
        # Copy features for real particles, keep -99 for padding
        x[real_particles, i] = particles[real_particles, i, :]
    
    print(f"Data stats - Real particles: {src_mask.mean():.1%}")
    return x, src_mask

def prepare_particle_data(signal_pkl_path, background_pkl_path, test_ratio=0.2, max_particles=13):
    """Main data preparation pipeline"""
    # Load data
    df_signal = pd.read_pickle(signal_pkl_path)
    df_background = pd.read_pickle(background_pkl_path)
    
    # Process particles (preserving -99 padding)
    signal_particles = select_particles(df_signal, max_particles)
    background_particles = select_particles(df_background, max_particles)
    
    # Build inputs and masks
    signal_x, signal_mask = build_input(signal_particles, max_particles)
    background_x, background_mask = build_input(background_particles, max_particles)
    
    # Combine and shuffle
    x = np.concatenate([signal_x, background_x])
    mask = np.concatenate([signal_mask, background_mask])
    y = np.concatenate([np.ones(len(signal_x)), np.zeros(len(background_x))])
    x, mask, y = shuffle(x, mask, y, random_state=42)
    
    # Train-test split
    x_train, x_test, mask_train, mask_test, y_train, y_test = train_test_split(
        x, mask, y, test_size=test_ratio, random_state=42)
    
    # Convert to tensors
    def to_tensor(array, dtype):
        return torch.as_tensor(array, dtype=dtype)
    
    return (
        to_tensor(x_train, torch.float32),
        to_tensor(x_test, torch.float32),
        to_tensor(mask_train, torch.bool),
        to_tensor(mask_test, torch.bool),
        to_tensor(y_train, torch.float32).unsqueeze(1),
        to_tensor(y_test, torch.float32).unsqueeze(1)
    )

def plot_training_loss_curve(train_losses, val_losses):
    plt.figure(figsize=(8, 6))
    
    plt.plot(train_losses, label='Training Loss', linewidth=2)
    plt.plot(val_losses, label='Validation Loss', linewidth=2)
    
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
    
    plt.hist(y_pred[y_true == 1], bins=50, alpha=0.5, label='Signal', density=False, histtype="step")
    plt.hist(y_pred[y_true == 0], bins=50, alpha=0.5, label='Background', density=False, histtype="step")
    
    plt.xlabel('Predicted Score')
    plt.ylabel('Normalized Counts')
    plt.legend()
    plt.savefig("results/counts_vs_predicted_score.png", dpi=300)
    plt.close()

def plot_roc_curve(y_true, y_pred):
    """Plot ROC curve"""
    from sklearn.metrics import roc_curve, auc
    
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
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.savefig("results/ROC_curve.png", dpi=300)
    plt.close()
    
def plot_attention_weights(particles, weights, save_path=None):
    """Simple attention visualization"""
    import matplotlib.pyplot as plt
    
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

def analyze_kinematics_after_cuts(results, score_cut=0.5):
    """Analyze kinematics for events passing score cut"""
    import matplotlib.pyplot as plt
    import pandas as pd
    
    # Extract data
    y_true = results['y_true']
    y_pred = results['y_pred']
    x_input = results['x_input']  # Shape: [n_events, max_particles, 6]
    
    # Create DataFrame for analysis
    analysis_df = pd.DataFrame({
        'is_signal': y_true,
        'pred_score': y_pred,
        'passes_cut': y_pred > score_cut
    })
    
    # Convert Cartesian back to polar coordinates for analysis
    def cartesian_to_polar(features):
        px, py, pz, energy, is_bjet, is_jet = features
        if px == -99:  # Padding
            return np.nan, np.nan, np.nan, np.nan
        
        pt = np.sqrt(px**2 + py**2)
        eta = np.arcsinh(pz/pt) if pt > 0 else 0
        phi = np.arctan2(py, px)
        mass = np.sqrt(energy**2 - (px**2 + py**2 + pz**2)) if energy**2 > (px**2 + py**2 + pz**2) else 0
        return pt, eta, phi, mass
    
    # Analyze leading particle (highest pT) in each event
    leading_particles = []
    for event in x_input:
        # Find leading particle (first non-padded particle)
        for particle in event:
            if particle[0] != -99:  # Not padded
                pt, eta, phi, mass = cartesian_to_polar(particle)
                leading_particles.append([pt, eta, phi, mass])
                break
        else:
            leading_particles.append([np.nan]*4)  # All padded
    
    # Add leading particle info to DataFrame
    analysis_df[['lead_pt', 'lead_eta', 'lead_phi', 'lead_mass']] = leading_particles
    
    # Plot kinematics distributions
    os.makedirs("results/kinematics", exist_ok=True)
    
    for var in ['lead_pt', 'lead_eta', 'lead_phi', 'lead_mass']:
        plt.figure(figsize=(10, 6))
        
        # Plot signal vs background for events passing cut
        for label, mask in [('Signal', analysis_df['is_signal'] == 1),
                            ('Background', analysis_df['is_signal'] == 0)]:
            data = analysis_df.loc[mask & (analysis_df['passes_cut']), var]
            plt.hist(data, bins=50, alpha=0.5, label=label, density=True)
        
        plt.xlabel(var)
        plt.ylabel('Normalized Counts')
        plt.title(f'{var} distribution after score cut > {score_cut}')
        plt.legend()
        plt.savefig(f"results/kinematics/{var}_after_cut.png")
        plt.close()
    
    # Save analysis results
    analysis_df.to_csv("results/kinematics/kinematic_analysis.csv", index=False)
    return analysis_df