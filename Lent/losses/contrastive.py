"""'Contrastive Representation Distillation' (Tian et al. 2019)"""
import torch
from torch import nn
import logging

from losses.loss_common import *


class CRDLoss(nn.Module):
    """CRD Loss function for student model
    Args:
        s_dim: Dimension of student's feature.
        t_dim: Dimension of teacher's feature.
        feat_dim: Dimension of projection space.
        T: Temperature to divide dot product of student and teacher features by.
    """
    def __init__(self, 
                 s_dim: int, 
                 t_dim: int, 
                 T: int, 
                 feat_dim: int = 128, 
                 device: torch.device = torch.device("cuda")):
        super(CRDLoss, self).__init__()
        self.embed_s = Embed(s_dim, feat_dim, device).to(device)
        self.embed_t = Embed(t_dim, feat_dim, device).to(device)
        self.T = T
        self.criterion_s = ContrastLoss()
        self.device = device

    def forward(self, 
                f_s: torch.Tensor, 
                f_t: torch.Tensor, 
                y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            f_s: Feature of student network, size [batch_size, s_dim].
            f_t: Feature of teacher network, size [batch_size, t_dim].
            y: Labels of samples, size [batch_size].
            
        Returns:
            s_loss: Contrastive loss for student model.
        """
        f_s = self.embed_s(f_s)
        f_t = self.embed_t(f_t)
        # Numerator of equation 19
        f_s = torch.div(f_s,torch.norm(f_s, dim=1).unsqueeze(1))
        f_t = torch.div(f_t,torch.norm(f_t, dim=1).unsqueeze(1))
        dot_product = torch.mm(f_s, f_t.t())
        out_s = torch.exp(torch.div(dot_product, self.T))
        s_loss = self.criterion_s(out_s, y)
        return s_loss

    
class ContrastLoss(nn.Module):
    """Contrastive loss with in-batch negatives."""
    def __init__(self, eps=0.97):
        super(ContrastLoss, self).__init__()
        self.eps = eps # Approximately N/M

    def forward(self, x: float, y: int) -> torch.Tensor:
        """
        Args:
            x: Exponentiated dot product similarity, size [batch_size, batch_size].
            y: Labels, size [batch_size].
        """
        bsz = x.shape[0]
        y_expand = y.unsqueeze(0).repeat(bsz, 1)
        
        # Matrix of comparisons
        pos_mask = (y_expand == y_expand.t())
        neg_mask = ~pos_mask
        
        # Exclude diagonal elements
        diag_mask = torch.eye(bsz, dtype=torch.bool, device=x.device)
        pos_mask = pos_mask & ~diag_mask
        neg_mask = neg_mask & ~diag_mask

        # Loss for positive pairs
        P_pos = x[pos_mask].view(-1, 1)
        log_D1 = torch.log(torch.div(P_pos, P_pos + self.eps))

         # Loss for negative pairs using in-batch negatives
        P_neg = x[neg_mask].view(-1, 1)
        log_D0 = torch.log(torch.div(self.eps, P_neg + self.eps))

        # Batch average loss (don't take into account variation in number of neg samples per image, is small overall)
        loss = - (log_D1.sum(0) + log_D0.sum(0)) / bsz
        
        return loss


class Embed(nn.Module):
    def __init__(self, 
                 dim_in: int = 1024, 
                 dim_out: int = 128,
                 device: torch.device = torch.device("cuda")):
        super(Embed, self).__init__()
        self.linear = nn.Linear(dim_in, dim_out)
        self.l2norm = Normalize(2)
        self.device = device

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.shape[0], -1).to(self.device)
        x = self.linear(x)
        x = self.l2norm(x)
        return x


class Normalize(nn.Module):
    """Normalization layer."""
    def __init__(self, power: int = 2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out


if __name__ == "__main__":
    s_dim = t_dim = 128
    T = 0.1
    feat_dim = 128
    bsz = 32
    criterion = CRDLoss(s_dim, t_dim, T, feat_dim)

    # Loss for random features
    f_s = torch.randn(bsz, s_dim)
    f_t = torch.randn(bsz, t_dim)
    y = torch.randint(0, 10, (bsz,))
    loss = criterion(f_s, f_t, y)
    # Loss for same features - should be much higher
    f_s1 = f_t1 = torch.cat((torch.ones(bsz, s_dim//2), 0.5*torch.ones(bsz, t_dim//2)), dim=1)
    y1 = torch.cat((torch.ones(bsz//2), 2*torch.ones(bsz//2)), dim=0)
    loss1 = criterion(f_s1, f_t1, y1)

    print(loss)
    print(loss1)