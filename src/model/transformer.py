import torch
import torch.nn as nn


class Transformer(nn.Module):
    def __init__(self, feat_dim, pe_dim):
        super(Transformer, self).__init__()
        self.feat_dim = feat_dim
        self.pe_dim = pe_dim

        self.lin_q = nn.Linear(pe_dim, pe_dim, bias=False)
        self.lin_k = nn.Linear(pe_dim, pe_dim, bias=False)
        self.lin_v = nn.Linear(feat_dim, feat_dim, bias=False)

        self.lin_q_self = nn.Linear(pe_dim, pe_dim, bias=False)
        self.lin_k_self = nn.Linear(pe_dim, pe_dim, bias=False)
        self.lin_v_self = nn.Linear(feat_dim, feat_dim, bias=False)

    def reset_parameters(self):
        self.lin_q.reset_parameters()
        self.lin_k.reset_parameters()
        self.lin_v.reset_parameters()

        self.lin_q_self.reset_parameters()
        self.lin_k_self.reset_parameters()
        self.lin_v_self.reset_parameters()

    def forward(self, pe_src, pe_dst, x_src, x_dst):
        # pe_src: [N_src, D_pe]
        # pe_dst: [N_dst, D_pe]
        # x_src: [N_src, D_feat]
        # x_dst: [N_dst, D_feat]
        # out: [N_src, D_feat]
        q, k, v = self.lin_q(pe_src), self.lin_k(pe_dst), self.lin_v(x_dst)
        q = q / torch.norm(q, p=2, dim=-1, keepdim=True)
        k = k / torch.norm(k, p=2, dim=-1, keepdim=True)

        q_self, k_self, v_self = self.lin_q_self(pe_src), self.lin_k_self(pe_src), self.lin_v_self(x_src)
        q_self = q_self / torch.norm(q_self, p=2, dim=-1, keepdim=True)
        k_self = k_self / torch.norm(k_self, p=2, dim=-1, keepdim=True)

        up = torch.einsum('ij,ik->jk', k, v)    # [N_dst, D_pe] [N_dst, D_feat] -> [D_pe, D_feat]
        up = torch.einsum('ij,jk->ik', q, up)   # [N_src, D_pe] [D_pe, D_feat] -> [N_src, D_feat]
        up += (q_self * k_self).sum(dim=-1)[..., None] * v_self

        down = k.sum(dim=0)                         # [N_dst, D_pe] -> [D_pe]
        down = torch.einsum('ij,j->i', q, down)     # [N_src, D_pe] [D_pe] -> [N_src]
        down += (q_self * k_self).sum(dim=-1)
        down.unsqueeze_(dim=-1)

        attn = up / (down + 1e-5)
        return attn
