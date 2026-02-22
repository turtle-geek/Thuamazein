import torch
from data.data_processing import CorruptedCIFAR

def run_stress_test(ensemble, device):
    """Benchmarks the ensemble against environmental noise."""
    corruptions = ['fog', 'gaussian_noise', 'glass_blur']
    
    print("Beginning Phase 4: Robustness Stress Test")
    ensemble.eval()
    
    for c in corruptions:
        # Load the specific corrupted dataset
        try:
            test_set = CorruptedCIFAR(corruption_type=c, severity=3)
            loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False)
            
            correct = 0
            with torch.no_grad():
                for images, labels in loader:
                    images, labels = images.to(device), labels.to(device)
                    # Get the consensus prediction from the ensemble
                    outputs, _ = ensemble(images)
                    correct += (outputs.argmax(1) == labels).sum().item()
            
            accuracy = 100 * correct / len(test_set)
            print(f"  > Robustness on {c}: {accuracy:.2f}%")
            
        except FileNotFoundError:
            print(f"  ! Error: Could not find data for {c}. Check your data/CIFAR-10-C/ folder.")

    print("Stress Test Complete.")