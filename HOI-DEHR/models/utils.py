import torch
import torch.nn as nn
from torch.nn.functional import linear, pad, softmax, dropout # containing original multi_head_attention_forward implementation
from torch.nn.init import xavier_normal_, xavier_uniform_, constant_
from torch.nn.modules.activation import Parameter # containing original MultiheadAttention implementation
# from torch.nn.modules.linear import _LinearWithBias
import torch.nn.functional as F

import numpy as np
import torchvision
from .box_ops import box_cxcywh_to_xyxy

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        try:
            ret = super().forward(x.type(torch.float32))
        except Exception as e:
            print(e)
        return ret.type(orig_type)

# 使用XavierFill初始化的FC
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

class RelationFeatureExtractor(nn.Module):
    def __init__(self, args, in_channels, out_dim, num_objs):
        super(RelationFeatureExtractor, self).__init__()
        self.args = args

        # head & tail feature (base feature)
        # fusion_dim = self.args.hidden_dim
        sub_dim = self.args.hidden_dim
        obj_dim = self.args.hidden_dim

        # spatial feature
        if args.use_spatial_feature:
            spatial_in_dim, spatial_out_dim = 8, 64
            # spatial_in_dim, spatial_out_dim = 16, 64
            self.spatial_proj = make_fc(spatial_in_dim, spatial_out_dim)
            # fusion_dim += spatial_out_dim
            sub_dim += spatial_out_dim
            obj_dim += spatial_out_dim

        # tail semantic feature
        if args.use_tail_semantic_feature:
            semantic_dim = 300
            self.label_embedding = nn.Embedding(num_objs, semantic_dim)
            # fusion_dim += semantic_dim
            obj_dim += semantic_dim

        # spatial relation
        if self.args.use_spatial_relation:
            semantic_dim = 300
            self.relation_embedding = nn.Embedding(5, semantic_dim)
            # fusion_dim += semantic_dim
            sub_dim += semantic_dim
            obj_dim += semantic_dim

        self.sub_fc = nn.Sequential(
            make_fc(sub_dim, out_dim), nn.ReLU(),
            make_fc(out_dim, out_dim), nn.ReLU()
        )

        self.obj_fc = nn.Sequential(
            make_fc(obj_dim, out_dim), nn.ReLU(),
            make_fc(out_dim, out_dim), nn.ReLU()
        )


    # 需要加上bs
    # head_boxes: bs * num_queries * boxes
    def forward(self, head_boxes, tail_boxes, head_feats, tail_feats, obj_label_logits=None):
        """pool feature for boxes on one image
            features: dxhxw
            boxes: Nx4 (cx_cy_wh, nomalized to 0-1)
            rel_pairs: Nx2
        """

        # head & tail features
        # relation_feats = (head_feats + tail_feats) / 2.0
        head_boxes = box_cxcywh_to_xyxy(head_boxes).clamp(0, 1)
        tail_boxes = box_cxcywh_to_xyxy(tail_boxes).clamp(0, 1)

        # 增加更多的spatial
        # spatial layout feats
        if self.args.use_spatial_feature:
            bs, num_queries, _ = head_boxes.size()

            index = torch.arange(0, num_queries)
            box_layout_feats = self.extract_spatial_layout_feats(torch.cat([head_boxes, tail_boxes], dim=1))
            rel_spatial_feats = self.spatial_proj(box_layout_feats[:, index, num_queries+index, :])
            head_feats = torch.cat([head_feats, rel_spatial_feats], dim=-1)
            tail_feats = torch.cat([tail_feats, rel_spatial_feats], dim=-1)

        # 使用pre-trained CLIP和word2vec可以获得更好的泛华性能？
        # semantic feature
        if self.args.use_tail_semantic_feature:
            semantic_feats = obj_label_logits.softmax(-1) @ self.label_embedding.weight
            tail_feats = torch.cat([tail_feats, semantic_feats], dim=-1)

        if self.args.use_spatial_relation:
            spatial_relation = self.generate_spatial_relation(head_boxes, tail_boxes)
            spatial_relation = torch.zeros((spatial_relation.size(0), spatial_relation.size(1), 5)).to(spatial_relation.device).scatter_(2, spatial_relation.unsqueeze(-1), 1.0)
            spatial_relation = spatial_relation @ self.relation_embedding.weight
            head_feats = torch.cat([head_feats, spatial_relation], dim=-1)
            tail_feats = torch.cat([tail_feats, spatial_relation], dim=-1)

        head_feats = self.sub_fc(head_feats)
        tail_feats = self.obj_fc(tail_feats)
        relation_feats = (head_feats + tail_feats) / 2.0

        return relation_feats

        # x = self.fusion_fc(relation_feats)
        # return x

    # 都是相对指标
    def extract_spatial_layout_feats(self, xyxy_boxes):
        # 距离
        # bs * num_queries * boxes//2
        box_center = torch.stack([(xyxy_boxes[:, :, 0] + xyxy_boxes[:, :, 2]) / 2, (xyxy_boxes[:, :, 1] + xyxy_boxes[:, :, 3]) / 2], dim=2)
        dxdy = box_center.unsqueeze(2) - box_center.unsqueeze(1) # distances
        theta = (torch.atan2(dxdy[...,1], dxdy[...,0]) / np.pi).unsqueeze(-1)
        dis = dxdy.norm(dim=-1, keepdim=True)

        # 面积
        box_area = (xyxy_boxes[:, :, 2:] - xyxy_boxes[:, :, :2]).prod(dim=2) # areas
        intersec_lt = torch.max(xyxy_boxes.unsqueeze(2)[...,:2], xyxy_boxes.unsqueeze(1)[...,:2]) # 两两配对，获取lt
        intersec_rb = torch.min(xyxy_boxes.unsqueeze(2)[...,2:], xyxy_boxes.unsqueeze(1)[...,2:]) # 两两配对，获取rb
        overlap = (intersec_rb - intersec_lt).clamp(min=0).prod(dim=-1, keepdim=True)  # 算面积
        union_lt = torch.min(xyxy_boxes.unsqueeze(2)[...,:2], xyxy_boxes.unsqueeze(1)[...,:2])
        union_rb = torch.max(xyxy_boxes.unsqueeze(2)[...,2:], xyxy_boxes.unsqueeze(1)[...,2:])
        union = (union_rb - union_lt).clamp(min=1).prod(dim=-1, keepdim=True) # 100 * 100

        spatial_feats = torch.cat([
            dxdy, dis, theta, # dx, dy, distance, theta
            overlap, union, box_area[:,:,None,None].expand(*union.shape), box_area[:,None,:,None].expand(*union.shape) # overlap, union, subj, obj
        ], dim=-1)
        return spatial_feats

    # cxcywh
    def generate_spatial_relation(self, head_boxes, tail_boxes):
        alpha = self.args.spatial_alpha
        # print(head_boxes.size(), tail_boxes.size())
        # print((tail_boxes[:, : 1] + 0.8 * tail_boxes[:, : 3]).size(), head_boxes[:, :, 1].size())
        # exit()
        # above
        above = head_boxes[:, :, 1] > (tail_boxes[:, :, 1] + alpha * tail_boxes[:, :, 3])
        # below
        below = head_boxes[:, :, 1] < (tail_boxes[:, :, 1] - alpha * tail_boxes[:, :, 3])

        # around
        # around = ((tail_boxes[:, :, 1] - 0.8 * tail_boxes[:, :, 3]) < head_boxes[:, :, 1] < (tail_boxes[:, :, 1] + 0.8 * tail_boxes[:, :, 3])) and \
        #         ((head_boxes[:, :, 0] < tail_boxes[:, :, 0] - 0.8 * tail_boxes[:, :, 2]) or (head_boxes[:, :, 0] > tail_boxes[:, :, 0] + 0.8 * tail_boxes[:, :, 2]))
        around = ((tail_boxes[:, :, 1] - alpha * tail_boxes[:, :, 3]) < head_boxes[:, :, 1]) * (head_boxes[:, :, 1] < (tail_boxes[:, :, 1] + alpha * tail_boxes[:, :, 3])) *\
                ((head_boxes[:, :, 0] < (tail_boxes[:, :, 0] - alpha * tail_boxes[:, :, 2])) + (head_boxes[:, :, 0] > (tail_boxes[:, :, 0] + alpha * tail_boxes[:, :, 2])))
        around = around > 0

        # within
        # within = ((tail_boxes[:, :, 1] - 0.8 * tail_boxes[:, :, 3]) < head_boxes[:, :, 1] < (tail_boxes[:, :, 1] + 0.8 * tail_boxes[:, :, 3])) and \
        #         ((tail_boxes[:, :, 0] - 0.8 * tail_boxes[:, :, 2]) < head_boxes[:, :, 0] < (tail_boxes[:, :, 0] + 0.8 * tail_boxes[:, :, 2]))
        within = ((tail_boxes[:, :, 1] - alpha * tail_boxes[:, :, 3]) < head_boxes[:, :, 1]) * (head_boxes[:, :, 1] < (tail_boxes[:, :, 1] + alpha * tail_boxes[:, :, 3])) *\
                ((head_boxes[:, :, 0] > (tail_boxes[:, :, 0] - alpha * tail_boxes[:, :, 2])) * (head_boxes[:, :, 0] < (tail_boxes[:, :, 0] + alpha * tail_boxes[:, :, 2])))
        within = within > 0

        # contain
        contain = (tail_boxes[:, :, 2] * tail_boxes[:, :, 3]) < 1e-5

        relation = above * 1 + below * 2 + around * 3 + within * 4

        relation = relation * (1 - contain*1)

        return relation






