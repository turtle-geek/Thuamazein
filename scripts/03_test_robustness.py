import torch
from data.data_processing import CorruptedCIFAR
from models.ensemble_manager import EnsembleManager

def run_stress_test(ensemble, device):
    corruptions = ['fog', 'gaussian_noise', 'glass_blur']
    for c in corruptions:
        test_set = CorruptedCIFAR(corruption_type=c, severity=3)
        loader = torch.utils.data.DataLoader(test_set, batch_size=64)
        
        correct = 0
        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                outputs, _ = ensemble(images)
                correct += (outputs.argmax(1) == labels).sum().item()
        
        print(f"Robustness on {c}: {100 * correct / len(test_set):.2f}%")