import torch
import torch.nn as nn
import torch.nn.functional as F

kl_loss = nn.KLDivLoss(reduction='batchmean')
temp = 1

def compute_kl_div_loss(scores, targets):
        return kl_loss(F.log_softmax(scores / temp, dim=1), F.softmax(targets / temp, dim=1))

def test_kl_div_loss():
    temp = 1  # Set your temperature value here

    # # Test case 1: Same scores and targets should result in KL divergence of 0
    # scores = torch.tensor([[1.0, 2.0], [2.0, 1.0]])
    # targets = torch.tensor([[1.0, 2.0], [2.0, 1.0]])
    # KL_diff = compute_kl_div_loss(scores, targets)
    # assert torch.isclose(KL_diff, torch.tensor(0.0), atol=1e-6), f"Expected 0, but got {KL_diff}"

    # Test case 2: Different scores and targets with known KL divergence
    scores = torch.tensor([[1.0, 2.0], [2.0, 1.0]])
    targets = torch.tensor([[10, 2.5], [2.5, 10]])
    expected_KL_diff = 0.05423  # The expected KL divergence value for this case
    KL_diff = compute_kl_div_loss(scores, targets)
    assert torch.isclose(KL_diff, torch.tensor(expected_KL_diff), atol=1e-5), f"Expected {expected_KL_diff}, but got {KL_diff}"

    print("All test cases passed!")

kl_loss = nn.KLDivLoss(reduction='mean')
temp = 1

def compute_kl_div_loss(scores, targets):
        return kl_loss(F.log_softmax(scores / temp, dim=1), F.softmax(targets / temp, dim=1))

if __name__ == "__main__":
    scores = torch.tensor([[1.0, 2.0], [2.0, 1.0]])
    targets = torch.tensor([[10, 2.5], [2.5, 10]])
    print(compute_kl_div_loss(scores, targets))

    scores = torch.tensor([[2, 2.0], [2.0, 1.0]])
    targets = torch.tensor([[2, 2], [2, 1]])
    print(compute_kl_div_loss(scores, targets))