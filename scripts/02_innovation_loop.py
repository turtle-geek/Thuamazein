import torch
import torch.nn as nn
from logic.diversity_loss import repulsion_penalty
from models.architectures import SimpleCNN
from data.data_processing import get_clean_loaders

def run_successor_training(ensemble, cut_idx, lmbda=0.5, epochs=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, _ = get_clean_loaders()
    
    # Keep the agents that passed the Similarity Audit
    survivors = [agent for i, agent in enumerate(ensemble.agents) if i != cut_idx]
    for s in survivors:
        s.eval() # Freeze survivors during innovation
    
    # Successor trains on 10-class CIFAR task while being 'repulsed' from survivors
    successor = SimpleCNN(num_classes=10).to(device)
    optimizer = torch.optim.Adam(successor.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    successor.train()
    for epoch in range(epochs):
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            
            output, succ_features = successor(images)
            task_loss = criterion(output, labels)
            
            # Extract features from survivors to calculate Dissonance Penalty
            with torch.no_grad():
                # We only need the latent features index [1] from SimpleCNN
                survivor_features = [s(images)[1] for s in survivors]
            
            diss_loss = repulsion_penalty(succ_features, survivor_features, lambda_val=lmbda)
            
            (task_loss + diss_loss).backward()
            optimizer.step()
            
    # Integrate the diverse logic back into the ensemble
    ensemble.agents[cut_idx] = successor
    torch.save(successor.state_dict(), f"models/agent_{cut_idx}_diverse.pth")
    print(f"Phase 3 Complete: Diverse Successor integrated at index {cut_idx}.")