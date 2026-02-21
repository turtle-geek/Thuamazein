import torch
import torch.nn as nn

class EnsembleManager(nn.Module):
    def __init__(self, agents_list):
        super().__init__()
        self.agents = nn.ModuleList(agents_list)

    def forward(self, x):
        """Standard forward pass: Returns averaged predictions and all latent features."""
        results = [agent(x) for agent in self.agents]
        all_outputs = [r[0] for r in results]
        all_features = [r[1] for r in results]
        
        # Weighted Consensus: Average the predictions of all agents
        avg_output = torch.stack(all_outputs).mean(0)
        return avg_output, all_features

    def get_stats(self, loader, device):
        """
        Gathers accuracy and feature maps for the Judge to analyze.
        Crucial for the 'Audit' phase of the innovation loop.
        """
        self.eval()
        all_acc = [0.0] * len(self.agents)
        collected_features = [[] for _ in range(len(self.agents))]
        
        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                _, features = self.forward(images)
                
                for i, agent in enumerate(self.agents):
                    output, _ = agent(images)
                    pred = output.argmax(dim=1)
                    all_acc[i] += (pred == labels).sum().item()
                    collected_features[i].append(features[i].cpu())

        # Normalize accuracy and concatenate latent features for analysis
        accuracies = [a / len(loader.dataset) for a in all_acc]
        final_features = [torch.cat(f) for f in collected_features]
        return accuracies, final_features