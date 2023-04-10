"""'Contrastive Representation Distillation' (Tian et al. 2019)"""
import torch
from torch import nn

# Approximately N/M
eps = 0.7

class CRDLoss(nn.Module):
    """CRD Loss function for student model
    Args:
        s_dim: the dimension of student's feature
        t_dim: the dimension of teacher's feature
        feat_dim: the dimension of the projection space
        T: temperature
    """
    def __init__(self, s_dim, t_dim, T, feat_dim=128):
        super(CRDLoss, self).__init__()
        self.embed_s = Embed(s_dim, feat_dim)
        self.embed_t = Embed(t_dim, feat_dim)
        self.T = T
        self.embed = s_dim != t_dim
        self.criterion_s = ContrastLoss()

    def forward(self, f_s, f_t, y):
        """
        Args:
            f_s: the feature of student network, size [batch_size, s_dim]
            f_t: the feature of teacher network, size [batch_size, t_dim]
            y: the labels of the samples, size [batch_size]
        Returns:
            The contrastive loss for the student model
        """
        # Only embed if the dimensions are different
        if self.embed:
            f_s = self.embed_s(f_s)
            f_t = self.embed_t(f_t)
        # Numerator of equation 19
        dot_product = torch.mm(torch.div(f_s,torch.norm(f_s, dim=1)), torch.div(f_t.t(),torch.norm(f_t, dim=1)))
        out_s = torch.exp(torch.div(dot_product, self.T))
        s_loss = self.criterion_s(out_s, y)
        return s_loss

class ContrastLoss(nn.Module):
    """
    contrastive loss with in-batch negatives
    """
    def __init__(self):
        super(ContrastLoss, self).__init__()

    def forward(self, x: float, y: int):
        """
        Args:
            x: exponentiated dot product similarity, size [batch_size, batch_size]
            y: labels, size [batch_size]
        """
        bsz = x.shape[0]
        c = 1e-7 # Small constant for stability
        x = x - torch.diag(torch.diag(x))   # Ignore the image itself
        y_expand = y.unsqueeze(0).repeat(bsz, 1)         # Repeat y along second dimension
        pos_mask = torch.eq(y_expand, y_expand.t()).detach()         # Matrix of comparisons
        neg_mask = torch.logical_not(pos_mask).detach()
        # Loss for positive pairs - flatten for ease
        P_pos = x[pos_mask].view(-1, 1)
        log_D1 = torch.log(torch.div(P_pos.add(c), P_pos.add(eps)))
        # Loss for negative pairs using in-batch negatives
        P_neg = x[neg_mask].view(-1, 1)
        log_D0 = torch.log(torch.div(eps, P_neg.add(eps+c)))
        # Batch average loss (don't take into account variation in number of neg samples per image, is small overall)
        loss = - (log_D1.sum(0) + log_D0.sum(0)) / bsz
        return loss

class Embed(nn.Module):
    """Embedding module"""
    def __init__(self, dim_in=1024, dim_out=128):
        super(Embed, self).__init__()
        self.linear = nn.Linear(dim_in, dim_out)
        self.l2norm = Normalize(2)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.linear(x)
        x = self.l2norm(x)
        return x

class Normalize(nn.Module):
    """normalization layer"""
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out