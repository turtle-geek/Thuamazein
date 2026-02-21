import torch
import torch.nn.functional as F

def repulsion_penalty(succ_feat, survivor_feats, lambda_val=0.5):
    """Calculates the dissonance/repulsion between models."""
    if not survivor_feats:
        return torch.tensor(0.0, device=succ_feat.device)
    
    succ_vec = succ_feat.view(succ_feat.size(0), -1)
    total_sim = 0
    
    for surv_feat in survivor_feats:
        surv_vec = surv_feat.view(surv_feat.size(0), -1)
        # Cosine Similarity: 1.0 is identical logic, 0.0 is orthogonal
        sim = F.cosine_similarity(succ_vec, surv_vec, dim=1).mean()
        total_sim += sim
        
    return lambda_val * (total_sim / len(survivor_feats))