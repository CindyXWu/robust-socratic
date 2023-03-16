""">>>'Contrastive Representation Distillation (Tiang et al 2022)'. 
Code adapted from https://github.com/HobbitLong/RepDistiller/blob/master/crd/criterion.py to be used without memory buffer.
"""
import torch.nn as nn
import torch

eps = 1e-7

class ContrastiveRep(nn.Module):
    """CRD Loss function
    Using teacher as anchor, choose positive and negative samples over the student side.
    This means the teacher data distribution is fixed
    Note original implementation uses memory buffer to allow large N (hence tighter bounds on MI) but this is not implemented here.
    Args:
        s_dim: dimension of student's feature
        t_dim: dimension of teacher's feature
        e_dim: dimension of projection space
        k: number of negatives paired with each positive
        T: temperature
        n_data: number of samples in the training set, therefore the memory buffer is: n_data * e_dim
    """
    def __init__(self, t_dim, s_dim, n_data, e_dim=50, k=10, T=1):
        super(ContrastiveRep, self).__init__()
        self.embed_s = Embed(dim_in=s_dim, dim_out=e_dim)
        self.embed_t = Embed(dim_in=t_dim, dim_out=e_dim)
        self.criterion_t = ContrastLoss(n_data)
        self.criterion_s = ContrastLoss(n_data)
        self.T = T
    
    def forward(self, f_s, f_t, idx, contrast_idx=None):
        """
        Args:
            f_s: the feature of student network, size [batch_size, s_dim]
            f_t: the feature of teacher network, size [batch_size, t_dim]
            idx: the indices of these positive samples in the dataset, size [batch_size]
            contrast_idx: the indices of negative samples, size [batch_size, nce_k]
        Returns:
            The contrastive loss
        """
        # N.B. if I fix teacher side, then I will assume teacher samples are fixed and I am optimising 
        # Embed module forward function
        f_s = self.embed_s(f_s)
        f_t = self.embed_t(f_t)
        # Compute dot product between embedded student and teacher features
        out = torch.bmm(f_s.unsqueeze(1), f_t.unsqueeze(2)).squeeze(2)
        out = torch.exp(torch.div(out, self.T))
        # Links to ContrastMemory class (fron original implementation)
        # out_s, out_t = self.contrast(f_s, f_t, idx, contrast_idx)
        # s_loss = self.criterion_s(out_s)
        # t_loss = self.criterion_t(out_t)
        # loss = s_loss + t_loss
        loss = self.criterion_s(out)
        return loss
    
class Embed(nn.Module):
    """Embedding module (Eq 19). 
    Embeds the feature to a lower and consistent dimension space between student and teacher, and normalises result.
    """
    def __init__(self, dim_in=1024, dim_out=128):
        super(Embed, self).__init__()
        self.linear = nn.Linear(dim_in, dim_out)
        self.l2norm = Normalize(2)

    def forward(self, x):
        # Size -1 is inferred from other dimensions
        # x.shape[0] is batch size
        x = x.view(x.shape[0], -1)
        x = self.linear(x)
        x = self.l2norm(x)
        # batch_size * dim_out
        return x

class ContrastLoss(nn.Module):
    """Contrastive loss, corresponding to Eq 18."""
    def __init__(self, n_data):
        super(ContrastLoss, self).__init__()
        self.n_data = n_data

    def forward(self, x):
        """x: some tensor that looks like the exponential of dot product of embedded vectors part of h(T,S)"""
        bsz = x.shape[0] # batch size
        m = x.size(1) - 1  # N, number of negative samples
        # noise distribution 1/M
        Pn = 1 / float(self.n_data)
        
        # Loss for positive pair
        # First value: axis, second value: index
        P_pos = x.select(1, 0) # first column is positive sample
        # m * Pn = N/M 
        log_D1 = torch.div(P_pos, P_pos.add(m * Pn + eps)).log_() # eps is some noise for stability? Not sure

        # loss for K negative pair
        # Only draw data from negative samples (expectation over q(T,S|C=0))
        P_neg = x.narrow(1, 1, m) # last m columns (batch dimension) are negative samples
        log_D0 = torch.div(P_neg.clone().fill_(m * Pn), P_neg.add(m * Pn + eps)).log_()

        loss = - (log_D1.sum(0) + log_D0.view(-1, 1).sum(0)) / bsz

        return loss
    
class Normalize(nn.Module):
    """Normalization layer."""
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out

if __name__ == '__main__':
    x = ContrastLoss(1000)
    x.forward(torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4]]))
    