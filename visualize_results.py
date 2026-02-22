import matplotlib.pyplot as plt
import json
import os

def plot_metrics():
    if not os.path.exists('training_log.json'):
        print("No training log found. Run main.py first.")
        return

    with open('training_log.json', 'r') as f:
        data = json.load(f)

    epochs = range(1, len(data['train_loss']) + 1)

    plt.figure(figsize=(12, 5))

    # Subplot 1: Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, data['train_loss'], 'r-', label='Training Loss')
    plt.title('Training Loss Decay')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()

    # Subplot 2: Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, data['test_acc'], 'g-', label='Validation Accuracy')
    plt.axhline(y=90, color='b', linestyle='--', label='90% Target')
    plt.title('Accuracy Growth')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig('performance_graphs.png')
    print("Graphs saved as 'performance_graphs.png'")
    plt.show()

if __name__ == "__main__":
    plot_metrics()