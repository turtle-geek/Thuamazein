import torch
import torch.nn as nn
import json
from torch.optim.lr_scheduler import CosineAnnealingLR
from data.data_processing import get_clean_loaders

def sharpen(ensemble, epochs, device):
    train_loader, test_loader = get_clean_loaders(batch_size=128)
    agents = ensemble.agents
    
    # High-Performance Settings: SGD + No Smoothing
    optimizer = torch.optim.SGD([{'params': a.parameters()} for a in agents], 
                                lr=0.01, momentum=0.9, weight_decay=5e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    # Diagnostic History
    history = {'train_loss': [], 'test_acc': [], 'lr': []}

    print(f"--- High-Precision Training: {epochs} Epochs ---")
    for epoch in range(epochs):
        ensemble.train()
        running_loss = 0.0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            
            total_loss = 0
            for agent in agents:
                out, _ = agent(images)
                total_loss += criterion(out, labels)
            
            total_loss.backward()
            optimizer.step()
            running_loss += total_loss.item()
        
        # Validation Pass (For Graphing)
        ensemble.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs, _ = ensemble(images)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        acc = 100. * correct / total
        avg_loss = running_loss / len(train_loader)
        current_lr = scheduler.get_last_lr()[0]
        
        # Log metrics
        history['train_loss'].append(avg_loss)
        history['test_acc'].append(acc)
        history['lr'].append(current_lr)
        
        print(f"  Epoch [{epoch+1}/{epochs}] | Loss: {avg_loss:.4f} | Acc: {acc:.2f}%")
        scheduler.step()

    # Save Log to File
    with open('training_log.json', 'w') as f:
        json.dump(history, f)

    for i, agent in enumerate(agents):
        torch.save(agent.state_dict(), f"models/agent_{i}.pth")