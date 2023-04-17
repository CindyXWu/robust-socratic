import torch
import torch.nn as nn
import torch.nn.functional as F

scores = torch.tensor([[1.0, 2.0], [2.0, 1.0]])
targets = torch.tensor([[7.5, 2.5], [2.5, 7.5]])
# Apply softmax to convert the tensors into probability distributions
scores_softmax = F.softmax(scores, dim=1)
targets_softmax = F.softmax(targets, dim=1)

# Calculate KL divergence using torch.nn.KLDivLoss
kldiv_loss = nn.KLDivLoss()
kl_divergence = kldiv_loss(scores_softmax.log(), targets_softmax)

print(kl_divergence.item())