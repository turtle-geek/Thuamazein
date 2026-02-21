import torch
import torch.nn.functional as F
import numpy as np

class SimilarityAudit:
    def __init__(self, threshold=0.90):
        """The Judge uses a threshold to flag high logical overlap."""
        self.threshold = threshold

    def evaluate_diversity(self, all_features):
        """
        Performs Representational Similarity Analysis (RSA).
        Calculates the cosine similarity between the latent logic of all agents.
        """
        num_agents = len(all_features)
        sim_matrix = np.zeros((num_agents, num_agents))

        for i in range(num_agents):
            for j in range(num_agents):
                # Flatten the feature maps to [Batch, Features] vectors
                f1 = all_features[i].view(all_features[i].size(0), -1)
                f2 = all_features[j].view(all_features[j].size(0), -1)
                
                # Mean Cosine Similarity: 1.0 is a copycat, 0.0 is unique logic
                sim = F.cosine_similarity(f1, f2, dim=1).mean().item()
                sim_matrix[i, j] = sim
        return sim_matrix

    def the_cut(self, accuracies, sim_matrix):
        """
        Decision logic for the innovation loop. 
        Identifies the agent that is either highly redundant or lowest performing.
        """
        num_agents = len(accuracies)
        # Calculate average similarity of each agent to its peers
        redundancy_scores = (sim_matrix.sum(axis=1) - 1) / (num_agents - 1)
        
        # Heuristic: Combine redundancy and error rate
        # Higher score = priority for retirement
        cut_scores = redundancy_scores + (1.0 - np.array(accuracies))
        return np.argmax(cut_scores)