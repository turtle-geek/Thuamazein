import torch
import torch.nn as nn
import yaml
import os
import importlib
import gc

from models.ensemble_manager import EnsembleManager
from models.architectures import SimpleCNN  # Standardized to ResNet-18
from logic.judge import SimilarityAudit
from data.data_processing import get_clean_loaders

# Custom sharpening logic
import sharpen_ensemble 

# Logic for the specific training phases
train_initial = importlib.import_module("scripts.01_train_initial")
innovation_loop = importlib.import_module("scripts.02_innovation_loop")
test_robustness = importlib.import_module("scripts.03_test_robustness")

def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

def main():
    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dir = config['paths']['model_save_path']
    
    # Epoch settings pulled from Config with hard-coded fallbacks
    PHASE1_EPOCHS = config['training'].get('epochs_phase1', 15)
    PHASE3_EPOCHS = config['training'].get('innovation_epochs', 20)
    SHARPEN_EPOCHS = config['training'].get('sharpen_epochs', 25)

    print(f"--- THAUMAZEIN SYSTEM ONLINE [Device: {device}] ---")

    # PHASE 1: Build the base agents if they don't exist
    if not os.path.exists(os.path.join(model_dir, "agent_2.pth")):
        print(f"Starting Phase 1: Training initial SSL agents ({PHASE1_EPOCHS} epochs)")
        train_initial.run_training(epochs=PHASE1_EPOCHS) 
    else:
        print("Phase 1: Existing foundation found, skipping pre-training.")
    
    # PHASE 1.5: Smart Loading (Fixes size mismatch errors)
    agents = []
    for i in range(3):
        path = os.path.join(model_dir, f"agent_{i}.pth")
        
        # Load raw state dict to check output layer size
        checkpoint = torch.load(path, map_location=device)
        
        # Determine if this is a 4-class (Rotation) or 10-class (CIFAR) model
        # Check both potential layer names for ResNet and SimpleCNN
        if "fc_final.bias" in checkpoint:
            out_features = checkpoint["fc_final.bias"].shape[0]
        elif "fc2.bias" in checkpoint:
            out_features = checkpoint["fc2.bias"].shape[0]
        else:
            out_features = 4  # Fallback to default
            
        print(f"Loading Agent {i}: Detected {out_features} output classes.")
        
        # Initialize model with the DETECTED number of classes to avoid RuntimeError
        model = SimpleCNN(num_classes=out_features).to(device)
        model.load_state_dict(checkpoint)
        agents.append(model)
    
    ensemble = EnsembleManager(agents).to(device)
    
    # PHASE 2: Similarity Audit
    print("Phase 2: Running Representational Similarity Audit (RSA)...")
    _, val_loader = get_clean_loaders(batch_size=config['training']['batch_size'])
    accuracies, feature_maps = ensemble.get_stats(val_loader, device)
    
    auditor = SimilarityAudit()
    sim_matrix = auditor.evaluate_diversity(feature_maps)
    cut_idx = auditor.the_cut(accuracies, sim_matrix)
    print(f"Audit Result: Agent {cut_idx} is redundant. Initiating Dissonance Loop.")
    
    # PHASE 3: Diversity Training
    # VRAM Cleanup before the heavy ResNet Dissonance training
    torch.cuda.empty_cache()
    gc.collect()

    print(f"Phase 3: Training Successor with Lambda={config['training'].get('lambda_dissonance', 0.8)}")
    innovation_loop.run_successor_training(
        ensemble, 
        cut_idx, 
        lmbda=config['training'].get('lambda_dissonance', 0.8),
        epochs=PHASE3_EPOCHS 
    ) 

    # Standardization: Map all diversity-trained heads to CIFAR-10 classes (10)
    print("Standardizing architectures for CIFAR-10 task mapping...")
    for agent in ensemble.agents:
        # Check ResNet-18 structure
        if hasattr(agent, 'fc_final'): 
            if agent.fc_final.out_features != 10:
                agent.fc_final = nn.Linear(agent.feature_dim, 10).to(device)
        # Check Legacy SimpleCNN structure
        elif hasattr(agent, 'fc2'): 
            if agent.fc2.out_features != 10:
                agent.fc2 = nn.Linear(512, 10).to(device)
    
    # PHASE 3.5: High-Fidelity Sharpening (Alignment)
    print(f"Phase 3.5: Running High-Fidelity Sharpening ({SHARPEN_EPOCHS} epochs)")
    sharpen_ensemble.sharpen(ensemble, SHARPEN_EPOCHS, device)

    # PHASE 4: Final Stress Test & Logging
    print("Phase 4: Evaluating Robustness on Environmental Corruptions...")
    test_robustness.run_stress_test(ensemble, device)
    
    print("\n[SYSTEM NOTIFICATION]: Evolution Complete. Please run generate_proof.py for final metrics.")

if __name__ == "__main__":
    main()