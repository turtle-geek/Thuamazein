import torch
import torch.nn as nn
import os # Required for checkpointing
from logic.diversity_loss import repulsion_penalty
from models.architectures import SimpleCNN
from data.data_processing import get_clean_loaders

def run_successor_training(ensemble, cut_idx, lmbda=0.5, epochs=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, _ = get_clean_loaders()
    
    survivors = [agent for i, agent in enumerate(ensemble.agents) if i != cut_idx]
    for s in survivors:
        s.eval()
    
    successor = SimpleCNN(num_classes=10).to(device)
    optimizer = torch.optim.Adam(successor.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # Resume Logic
    checkpoint_path = f"models/successor_checkpoint.pth"
    start_epoch = 0
    if os.path.exists(checkpoint_path):
        print("Checkpoint found! Resuming from last saved state...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        successor.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
    
    successor.train()
    for epoch in range(start_epoch, epochs):
        print(f"Epoch [{epoch+1}/{epochs}]")
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            
            output, succ_features = successor(images)
            task_loss = criterion(output, labels)
            
            with torch.no_grad():
                survivor_features = [s(images)[1] for s in survivors]
            
            diss_loss = repulsion_penalty(succ_features, survivor_features, lambda_val=lmbda)
            
            (task_loss + diss_loss).backward()
            optimizer.step()

            # Progress heartbeat
            if batch_idx % 100 == 0:
                print(f"  Batch {batch_idx}/{len(train_loader)} | Loss: {task_loss.item():.4f}")

        # Save checkpoint after each epoch
        torch.save({
            'epoch': epoch,
            'model_state_dict': successor.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, checkpoint_path)
            
    ensemble.agents[cut_idx] = successor
    torch.save(successor.state_dict(), f"models/agent_{cut_idx}_diverse.pth")
    if os.path.exists(checkpoint_path): os.remove(checkpoint_path)
    print(f"Phase 3 Complete: Diverse Successor integrated at index {cut_idx}.")