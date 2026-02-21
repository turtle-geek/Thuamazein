import torch
import yaml
import os
import importlib

from models.ensemble_manager import EnsembleManager
from models.architectures import SimpleCNN
from logic.judge import SimilarityAudit
from data.data_processing import get_clean_loaders

# Using importlib to handle scripts starting with numbers
train_initial = importlib.import_module("scripts.01_train_initial")
innovation_loop = importlib.import_module("scripts.02_innovation_loop")
test_robustness = importlib.import_module("scripts.03_test_robustness")

def load_config():
    """Loads hyperparameters and paths from the config file."""
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

def main():
    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dir = config['paths']['model_save_path']
    
    # PHASE 1: Initialization / Smart Check
    # Skip training if agents already exist to save time.
    if not os.path.exists(os.path.join(model_dir, "agent_2.pth")):
        print("PHASE 1: No existing agents found. Training initial agents...")
        train_initial.run_training(epochs=config['training']['epochs_phase1']) 
    else:
        print("PHASE 1: Existing agents detected. Skipping to Audit phase.")
    
    # Assembly: Load trained agents into the Ensemble Manager
    agents = []
    for i in range(3):
        model = SimpleCNN().to(device)
        path = os.path.join(model_dir, f"agent_{i}.pth")
        model.load_state_dict(torch.load(path, map_location=device))
        agents.append(model)
    
    ensemble = EnsembleManager(agents).to(device)
    
    # PHASE 2: Representational Audit
    # The Judge identifies logical mimicry via RSA.
    print("PHASE 2: Running Similarity Audit to detect redundancy...")
    _, val_loader = get_clean_loaders(batch_size=config['training']['batch_size'])
    accuracies, feature_maps = ensemble.get_stats(val_loader, device)
    
    auditor = SimilarityAudit()
    sim_matrix = auditor.evaluate_diversity(feature_maps)
    cut_idx = auditor.the_cut(accuracies, sim_matrix)
    print(f"Result: Agent {cut_idx} identified as redundant and slated for innovation.")
    
    # PHASE 3: Adversarial Innovation Loop
    # Trains a Successor with a Dissonance Penalty to create unique logic.
    print("PHASE 3: Training diverse Successor with Dissonance Penalty...")
    innovation_loop.run_successor_training(
        ensemble, 
        cut_idx, 
        lmbda=config['training']['lambda_dissonance']
    ) 
    
    # PHASE 4: Robustness Stress Test
    # Final benchmark against environmental noise in CIFAR-10-C.
    print("PHASE 4: Evaluating ensemble robustness against corruptions...")
    test_robustness.run_stress_test(ensemble, device)

if __name__ == "__main__":
    main()