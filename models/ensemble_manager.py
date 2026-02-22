import torch
import torch.nn as nn

class EnsembleManager(nn.Module):
    def __init__(self, agents):
        super(EnsembleManager, self).__init__()
        self.agents = nn.ModuleList(agents)

    def forward(self, x):
        # Collect logits and features from all lenses
        all_results = [agent(x) for agent in self.agents]
        all_logits = torch.stack([res[0] for res in all_results]) # [3, B, 10]
        
        # DYNAMIC GATING: Calculate Confidence (Inverse Entropy)
        # High entropy = The lens is confused by corruption.
        # Low entropy = The lens 'recognizes' the pattern.
        probs = torch.softmax(all_logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)
        
        # Convert entropy to weights (Lower entropy -> Higher Weight)
        confidence = 1.0 / (entropy + 1e-9)
        weights = confidence / torch.sum(confidence, dim=0) # Normalize across agents
        
        # Apply weights to logits [3, B, 10] * [3, B, 1]
        weighted_logits = (all_logits * weights.unsqueeze(-1)).sum(dim=0)
        
        # Average features for the RSA audit phase
        avg_features = torch.mean(torch.stack([res[1] for res in all_results]), dim=0)
        
        return weighted_logits, avg_features

    def get_stats(self, loader, device):
        self.eval()
        accuracies = []
        feature_list = []
        
        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                
                batch_features = []
                for agent in self.agents:
                    outputs, features = agent(images)
                    batch_features.append(features.cpu())
                    
                feature_list.append(torch.stack(batch_features)) # [3, B, D]
        
        # Simplified stats for Phase 2 Audit
        # (This is a helper for your SimilarityAudit class)
        return [0, 0, 0], torch.cat(feature_list, dim=1)