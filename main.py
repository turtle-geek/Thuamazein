# import torch
# import torch.nn as nn
# import yaml
# import os
# import importlib
# import gc
# import json

# from models.ensemble_manager import EnsembleManager
# from models.architectures import SimpleCNN 
# from data.data_processing import get_clean_loaders
# import sharpen_ensemble 

# # Phase scripts
# train_initial = importlib.import_module("scripts.01_train_initial")
# innovation_loop = importlib.import_module("scripts.02_innovation_loop")
# test_robustness = importlib.import_module("scripts.03_test_robustness")

# def load_config():
#     with open("config.yaml", "r") as f:
#         return yaml.safe_load(f)

# def main():
#     config = load_config()
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model_dir = config['paths']['model_save_path']
    
#     # Epoch settings
#     PHASE1_EPOCHS = config['training'].get('epochs_phase1', 20)
#     PHASE3_EPOCHS = config['training'].get('innovation_epochs', 25)
#     SHARPEN_EPOCHS = config['training'].get('sharpen_epochs', 40)

#     print(f"Thaumazein evolution started on {device}")

#     # Phase 1: SSL Foundation
#     # Trains initial models on rotation task to build visual features
#     if not os.path.exists(os.path.join(model_dir, "agent_2.pth")):
#         print(f"phase 1: training 3 initial agents ({PHASE1_EPOCHS} epochs)")
#         train_initial.run_training(epochs=PHASE1_EPOCHS) 
#     else:
#         print("phase 1: skipping, weights found")
    
#     agents = []
#     for i in range(3):
#         path = os.path.join(model_dir, f"agent_{i}.pth")
#         model = SimpleCNN(num_classes=4).to(device)
#         model.load_state_dict(torch.load(path, map_location=device))
#         agents.append(model)
    
#     ensemble = EnsembleManager(agents).to(device)
    
#     # Phase 2: Similarity Audit
#     # Identifies the most redundant agent for replacement
#     print("phase 2: running rsa similarity audit")
#     _, val_loader = get_clean_loaders(batch_size=config['training']['batch_size'])
#     accuracies, feature_maps = ensemble.get_stats(val_loader, device)
    
#     from logic.judge import SimilarityAudit
#     auditor = SimilarityAudit()
#     sim_matrix = auditor.evaluate_diversity(feature_maps)
#     cut_idx = auditor.the_cut(accuracies, sim_matrix)
#     print(f"audit result: replacing agent {cut_idx}")
    
#     # Phase 3: Dissonance Loop
#     # Trains a successor to be accurate but mathematically diverse
#     torch.cuda.empty_cache()
#     gc.collect()

#     print(f"phase 3: training diverse successor ({PHASE3_EPOCHS} epochs)")
#     innovation_loop.run_successor_training(
#         ensemble, 
#         cut_idx, 
#         lmbda=config['training'].get('lambda_dissonance', 0.8),
#         epochs=PHASE3_EPOCHS 
#     ) 

#     # Transition: Standardization
#     # Convert all model heads to 10-class CIFAR-10 output
#     print("standardizing architecture for final task")
#     for agent in ensemble.agents:
#         if hasattr(agent, 'fc_final') and agent.fc_final.out_features != 10:
#             agent.fc_final = nn.Linear(agent.feature_dim, 10).to(device)

#     # Phase 3.5: Accuracy Sharpening
#     # Final training with data augmentation to hit 90%
#     print(f"phase 3.5: starting accuracy sharpen ({SHARPEN_EPOCHS} epochs)")
#     sharpen_ensemble.sharpen(ensemble, SHARPEN_EPOCHS, device)

#     # Phase 4: Robustness Test
#     # Evaluating performance on corrupted datasets (Fog, Noise, Blur)
#     print("phase 4: evaluating robustness benchmarks")
#     test_robustness.run_stress_test(ensemble, device)
    
#     print("evolution complete. use visualize_results.py for graphs.")

# if __name__ == "__main__":
#     main()

import torch
import torch.nn as nn
import yaml
import os
import importlib
import gc

from models.ensemble_manager import EnsembleManager
from models.architectures import SimpleCNN 
from data.data_processing import get_clean_loaders

# Phase scripts
test_robustness = importlib.import_module("scripts.03_test_robustness")

def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

def main():
    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dir = config['paths']['model_save_path']
    
    print(f"thaumazein diagnostic mode on {device}")

    # phase 1, 2, 3, 3.5: skipped to preserve 92.68% accuracy weights
    
    agents = []
    for i in range(3):
        path = os.path.join(model_dir, f"agent_{i}.pth")
        if not os.path.exists(path):
            print(f"error: agent_{i}.pth not found. cannot run robustness test.")
            return

        # smart loading: detect if weights are 4-class or 10-class
        checkpoint = torch.load(path, map_location=device)
        if "fc_final.bias" in checkpoint:
            num_classes = checkpoint["fc_final.bias"].shape[0]
        else:
            num_classes = 4 # fallback for phase 1 weights
            
        print(f"loading agent {i} ({num_classes} classes)")
        model = SimpleCNN(num_classes=num_classes).to(device)
        model.load_state_dict(checkpoint)
        agents.append(model)
    
    ensemble = EnsembleManager(agents).to(device)
    
    # ensure all models are 10-class before testing on cifar-10-c
    for agent in ensemble.agents:
        if hasattr(agent, 'fc_final') and agent.fc_final.out_features != 10:
            agent.fc_final = nn.Linear(agent.feature_dim, 10).to(device)

    # phase 4: robustness test
    # make sure you have updated data_processing.py with the 'severity' argument fix
    print("phase 4: evaluating robustness benchmarks (fog, noise, blur)")
    torch.cuda.empty_cache()
    gc.collect()
    
    test_robustness.run_stress_test(ensemble, device)
    
    print("evaluation complete. run visualize_results.py to see your accuracy curves.")

if __name__ == "__main__":
    main()