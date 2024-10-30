import torch
import torch.nn.functional as F
from torch.nn import Module
from torch import nn
from torch.nn.parameter import Parameter
import numpy as np
import math
# from .MS_HGNN_batch import MS_HGNN_oridinary, MS_HGNN_hyper, MLP

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        try:
            ret = super().forward(x.type(torch.float32))
        except Exception as e:
            print(e)
        return ret.type(orig_type)

def make_fc(dim_in, hidden_dim, a=1):
    '''
        Caffe2 implementation uses XavierFill, which in fact
        corresponds to kaiming_uniform_ in PyTorch
        a: negative slope
    '''
    fc = nn.Linear(dim_in, hidden_dim)
    nn.init.kaiming_uniform_(fc.weight, a=a)
    nn.init.constant_(fc.bias, 0)
    return fc

class HyperGraphHead(Module):
    def __init__(self, hidden_dim, layer=1):
        super().__init__()
        self.layer = layer
        self.hgnn1 = HGNN(hidden_dim, hidden_dim)
        fusion_dim = hidden_dim*(1+self.layer)
        self.fc = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    def init_adj_attention(self, feat, feat_corr, scale_factor=2):
        batch = feat.shape[0]
        actor_number = feat.shape[1]
        if scale_factor == actor_number:
            H_matrix = torch.ones(batch, 1, actor_number).type_as(feat)
            return H_matrix
        group_size = scale_factor
        if group_size < 1:
            group_size = 1

        _, indice = torch.topk(feat_corr, dim=2, k=group_size, largest=True, sorted=False)
        H_matrix = torch.zeros(batch, actor_number, actor_number).type_as(feat)
        H_matrix = H_matrix.scatter(2, indice, 1)

        return H_matrix

    def _generate_G_from_H(self, H):
        n_edge = H.size(-2)
        # the weight of the hyperedge
        W = torch.ones(H.size()[:-1]).to(H.device)
        # the degree of the node
        DV = torch.sum(H * W.unsqueeze(-1), axis=-1)
        # the degree of the hyperedge
        DE = torch.sum(H, axis=-2)

        # invDE = torch.diag_embed(torch.pow(DE, -1), dim1=-2, dim2=-1)
        # DV2 = torch.diag_embed(torch.pow(DV, -0.5), dim1=-2, dim2=-1)
        invDE = torch.diag_embed(torch.pow(DE+1e-4, -1)*(DE>0), dim1=-2, dim2=-1)
        DV2 = torch.diag_embed(torch.pow(DV+1e-4, -0.5)*(DV>0), dim1=-2, dim2=-1)
        W = torch.diag_embed(W, dim1=-2, dim2=-1)
        HT = torch.transpose(H, -2, -1)

        G = DV2 * H * W * invDE * HT * DV2
        return G

    def forward(self, embed):
        query_input = F.normalize(embed, p=2, dim=2)
        feat_corr = torch.matmul(query_input, query_input.permute(0, 2, 1))

        embeds = [embed]
        for scale_factor in range(1, self.layer+1):
            H = self.init_adj_attention(embed, feat_corr, scale_factor=scale_factor)
            G = self._generate_G_from_H(H)
            _embed = self.hgnn1(embed, G)
            embeds.append(_embed)
        embed = self.fc(torch.cat(embeds, dim=-1))

        return embed

class HGNN_conv(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super(HGNN_conv, self).__init__()

        self.weight = Parameter(torch.Tensor(in_ft, out_ft))
        if bias:
            self.bias = Parameter(torch.Tensor(out_ft))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x: torch.Tensor, G: torch.Tensor):
        x = x.matmul(self.weight)  # x_l * theta_l
        if self.bias is not None:
            x = x + self.bias
        x = G.matmul(x)  # G * x_l
        return x

class HGNN(nn.Module):
    def __init__(self, in_ch, n_hid, dropout=0.5):
        super(HGNN, self).__init__()
        self.dropout = dropout
        self.hgc1 = HGNN_conv(in_ch, n_hid)
        self.hgc2 = HGNN_conv(n_hid, n_hid)

    def forward(self, x, G):
        x = F.relu(self.hgc1(x, G))
        x = F.dropout(x, self.dropout)
        x = self.hgc2(x, G)

        return x



