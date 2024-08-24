import torch
from torch import nn
import torch.nn.functional as F

from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou, box_eou, generalized_box_eou
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size,
                       is_dist_avail_and_initialized)
import numpy as np

from .backbone import build_backbone
from .matcher import build_matcher
from .transformer import build_transformer

import json
from lavis.models import load_model_and_preprocess
# from fairseq.models.roberta import RobertaModel

from .ASL import *

def _sigmoid(x):
    y = torch.clamp(x.sigmoid(), min=1e-4, max=1 - 1e-4)
    return y

class GEN_VLKT(nn.Module):
    def __init__(self, backbone, transformer, num_queries, aux_loss=False, args=None):
        super().__init__()

        self.args = args
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.query_embed_h = nn.Embedding(num_queries, hidden_dim)
        self.query_embed_o = nn.Embedding(num_queries, hidden_dim)

        self.pos_guided_embedd = nn.Embedding(num_queries, hidden_dim)
        self.inst_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.group_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)

        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone
        self.aux_loss = aux_loss
        self.dec_layers = self.args.dec_layers

        with open(args.attribute_config_path, 'r') as f:
            num_classes = json.loads(f.read())['num_classes']
        self.label_names = list(num_classes.keys())

        # self.logit_scales = nn.Parameter(torch.ones([5]) * np.log(1 / 0.07))
        self.logit_scales = nn.Parameter(torch.ones([22]) * np.log(1 / 0.07))
        self.obj_logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.attr_class_fc = nn.Sequential(
            nn.Linear(hidden_dim, args.clip_embed_dim),
            nn.LayerNorm(args.clip_embed_dim),
        )

        self.nvi_class_embeddings = nn.Linear(args.clip_embed_dim, 22) 
        self._obj_class_embed = nn.Linear(hidden_dim, 1)

        self.hidden_dim = hidden_dim
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.pos_guided_embedd.weight)

    def forward(self, samples: NestedTensor, is_training=True):
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)

        src, mask = features[-1].decompose()
        assert mask is not None
        h_hs, o_hs, attr_hs = self.transformer(self.input_proj(src), mask,
                                                self.query_embed_h.weight,
                                                self.query_embed_o.weight,
                                                self.pos_guided_embedd.weight,
                                                pos[-1])[:3]

        outputs_inst_coord = self.inst_bbox_embed(h_hs).sigmoid()
        outputs_group_coord = self.group_bbox_embed(o_hs).sigmoid()

        outputs_obj_class = self._obj_class_embed(o_hs)
        attr_hs = self.attr_class_fc(attr_hs)  # 可以尝试去掉
        outputs_attr_class = self.nvi_class_embeddings(attr_hs)

        out = {'pred_nvi_logits': outputs_attr_class[-1], 'pred_obj_logits': outputs_obj_class[-1],
               'pred_inst_boxes': outputs_inst_coord[-1], 'pred_group_boxes': outputs_group_coord[-1],}

        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss_triplet(outputs_attr_class, outputs_obj_class, outputs_inst_coord, outputs_group_coord)

        return out

    @torch.jit.unused
    def _set_aux_loss_triplet(self, outputs_attr_class, outputs_obj_class,
                               outputs_inst_coord, outputs_group_coord):

        aux_outputs = {'pred_nvi_logits': outputs_attr_class[-self.dec_layers: -1],
                       'pred_obj_logits': outputs_obj_class[-self.dec_layers: -1],
                       'pred_inst_boxes': outputs_inst_coord[-self.dec_layers: -1],
                       'pred_group_boxes': outputs_group_coord[-self.dec_layers: -1]}

        outputs_auxes = []
        for i in range(self.dec_layers - 1):
            output_aux = {}
            for aux_key in aux_outputs.keys():
                output_aux[aux_key] = aux_outputs[aux_key][i]
            outputs_auxes.append(output_aux)
        return outputs_auxes

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class SetCriterionHOI(nn.Module):

    def __init__(self, num_obj_classes, num_queries, num_verb_classes, matcher, weight_dict, eos_coef, losses, args):
        super().__init__()

        self.num_obj_classes = num_obj_classes
        self.num_queries = num_queries
        self.num_verb_classes = num_verb_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.alpha = args.alpha

        with open(args.attribute_config_path, 'r') as f:
            attribute_config = json.loads(f.read())
            self.num_classes = attribute_config['num_classes']
            self.empty_weight = attribute_config['empty_weight']

        self.args = args
        self.empty_weight_dict = {}
        for label_name in self.num_classes.keys():
            empty_weight = torch.ones(self.num_classes[label_name] + 1)
            empty_weight[0] = self.empty_weight[label_name]
            self.empty_weight_dict[label_name] = empty_weight

        self.asl_action_loss = AsymmetricLossOptimized(gamma_neg=4, gamma_pos=0, clip=0.05, alpha=0.5)
  
    def loss_instances(self, outputs, targets, indices, num_boxes, log=True):
        src_logits = outputs['pred_obj_logits'].squeeze(-1).sigmoid()
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([torch.ones(len(t['inst_boxes'])) for t in targets]).to(src_logits.device)
        target_classes = torch.zeros(src_logits.shape).to(src_logits.device)
        target_classes[idx] = target_classes_o

        loss_human_exist = self._neg_loss(src_logits, target_classes)
        losses = {'loss_instance': loss_human_exist}

        return losses

    def loss_union_constrain(self, outputs, targets, indices, num_boxes):
        assert 'pred_inst_boxes' in outputs and 'pred_group_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_sub_boxes = outputs['pred_inst_boxes'][idx]
        src_obj_boxes = outputs['pred_group_boxes'][idx]
        target_sub_boxes = torch.cat([t['inst_boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        target_obj_boxes = torch.cat([t['group_boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        exist_obj_boxes = (target_obj_boxes != 0).any(dim=1)

        losses = {}
        if src_sub_boxes.shape[0] == 0:
            losses['loss_ciou'] = src_sub_boxes.sum() + src_obj_boxes.sum()
        else:
            loss_sub_ciou = 1 - torch.diag(generalized_box_eou(box_cxcywh_to_xyxy(src_sub_boxes),
                                                            box_cxcywh_to_xyxy(target_obj_boxes)))
            loss_obj_ciou = 1 - torch.diag(generalized_box_eou(box_cxcywh_to_xyxy(target_sub_boxes),
                                                            box_cxcywh_to_xyxy(src_obj_boxes)))
            losses['loss_sub_ciou'] = (loss_sub_ciou * exist_obj_boxes).sum() / (exist_obj_boxes.sum() + 1e-4)
            losses['loss_obj_ciou'] = loss_obj_ciou.sum() / num_boxes

        return losses

    @torch.no_grad()
    def loss_obj_cardinality(self, outputs, targets, indices, num_boxes):
        pred_logits = outputs['pred_obj_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v['obj_labels']) for v in targets], device=device)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'obj_cardinality_error': card_err}
        return losses

    def loss_attributes(self, outputs, targets, indices, num_boxes):
        assert 'pred_nvi_logits' in outputs
        src_logits = outputs['pred_nvi_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t['attributes'][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.zeros_like(src_logits)
        target_classes[idx] = target_classes_o
        src_logits = _sigmoid(src_logits)
        loss_hoi_ce = self.asl_action_loss(src_logits, target_classes)
        losses = {'loss_nvi_labels': loss_hoi_ce}

        return losses

    def loss_inst_group_boxes(self, outputs, targets, indices, num_boxes):
        assert 'pred_inst_boxes' in outputs and 'pred_group_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_sub_boxes = outputs['pred_inst_boxes'][idx]
        src_obj_boxes = outputs['pred_group_boxes'][idx]
        target_sub_boxes = torch.cat([t['inst_boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        target_obj_boxes = torch.cat([t['group_boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        exist_obj_boxes = (target_obj_boxes != 0).any(dim=1)

        losses = {}
        if src_sub_boxes.shape[0] == 0:
            losses['loss_inst_bbox'] = src_sub_boxes.sum()
            losses['loss_group_bbox'] = src_obj_boxes.sum()
            losses['loss_inst_giou'] = src_sub_boxes.sum()
            losses['loss_group_giou'] = src_obj_boxes.sum()
        else:
            loss_sub_bbox = F.l1_loss(src_sub_boxes, target_sub_boxes, reduction='none')
            loss_obj_bbox = F.l1_loss(src_obj_boxes, target_obj_boxes, reduction='none')
            losses['loss_inst_bbox'] = loss_sub_bbox.sum() / num_boxes
            losses['loss_group_bbox'] = (loss_obj_bbox * exist_obj_boxes.unsqueeze(1)).sum() / (exist_obj_boxes.sum() + 1e-4)
            loss_sub_giou = 1 - torch.diag(generalized_box_iou(box_cxcywh_to_xyxy(src_sub_boxes),
                                                               box_cxcywh_to_xyxy(target_sub_boxes)))
            loss_obj_giou = 1 - torch.diag(generalized_box_iou(box_cxcywh_to_xyxy(src_obj_boxes),
                                                               box_cxcywh_to_xyxy(target_obj_boxes)))
            losses['loss_inst_giou'] = loss_sub_giou.sum() / num_boxes
            losses['loss_group_giou'] = (loss_obj_giou * exist_obj_boxes).sum() / (exist_obj_boxes.sum() + 1e-4)
        return losses

    def _neg_loss(self, pred, gt, weights=None, alpha=0.25):
        ''' Modified focal loss. Exactly the same as CornerNet.
          Runs faster and costs a little bit more memory
        '''
        pos_inds = gt.eq(1).float()
        neg_inds = gt.lt(1).float()

        loss = 0

        pos_loss = alpha * torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
        if weights is not None:
            pos_loss = pos_loss * weights[:-1]

        neg_loss = (1 - alpha) * torch.log(1 - pred) * torch.pow(pred, 2) * neg_inds

        num_pos = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if num_pos == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos
        return loss

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def get_loss(self, loss, outputs, targets, indices, num, **kwargs):
        loss_map = {
            'instances': self.loss_instances,
            'inst_group_boxes': self.loss_inst_group_boxes,
            'attributes': self.loss_attributes,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num, **kwargs)

    def forward(self, outputs, targets):
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        num_boxes = sum(len(t["inst_boxes"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=outputs['pred_inst_boxes'].device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses


class PostProcessHOITriplet(nn.Module):

    def __init__(self, args):
        super().__init__()
        with open(args.attribute_config_path, 'r') as f:
            num_classes = json.loads(f.read())['num_classes']
        self.label_names = list(num_classes.keys())

    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        out_nvi_logits, out_obj_logits, out_inst_boxes, out_group_boxes = outputs['pred_nvi_logits'], \
                                                                           outputs['pred_obj_logits'], \
                                                                           outputs['pred_inst_boxes'], \
                                                                           outputs['pred_group_boxes']

        assert len(out_obj_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        obj_scores = out_obj_logits.sigmoid()

        # 这里把bbox缩放回去了
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1).to(obj_scores.device)
        inst_boxes = box_cxcywh_to_xyxy(out_inst_boxes)
        inst_boxes = inst_boxes * scale_fct[:, None, :]
        group_boxes = box_cxcywh_to_xyxy(out_group_boxes)
        group_boxes = group_boxes * scale_fct[:, None, :]

        nvi_scores = out_nvi_logits.sigmoid()
        results = []
        for index in range(len(nvi_scores)):
            results.append({'inst_boxes': inst_boxes[index].to('cpu'), 'group_boxes': group_boxes[index].to('cpu')})
            ids = torch.arange(obj_scores.shape[1])
            results[-1].update({'nvi_scores': nvi_scores[index].to('cpu'), 'obj_scores': obj_scores[index].to('cpu'), 'inst_ids': ids, 'group_ids': ids})

        return results


def build(args):
    device = torch.device(args.device)

    backbone = build_backbone(args)

    transformer = build_transformer(args)

    model = GEN_VLKT(
        backbone,
        transformer,
        num_queries=args.num_queries,
        aux_loss=args.aux_loss,
        args=args
    )

    matcher = build_matcher(args)
    weight_dict = {}
    weight_dict['loss_inst_bbox'] = args.bbox_loss_coef
    weight_dict['loss_group_bbox'] = args.bbox_loss_coef
    weight_dict['loss_inst_giou'] = args.giou_loss_coef
    weight_dict['loss_group_giou'] = args.giou_loss_coef
    weight_dict['loss_instance'] = args.obj_loss_coef
    weight_dict['loss_sub_ciou'] = 1.0
    weight_dict['loss_obj_ciou'] = 1.0
    weight_dict['loss_nvi_labels'] = 2.0

    if args.with_mimic:
        weight_dict['loss_feat_mimic'] = args.mimic_loss_coef

    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)
    losses = ['instances', 'inst_group_boxes', 'attributes']
    if args.with_mimic:
        losses.append('feats_mimic')

    criterion = SetCriterionHOI(args.num_obj_classes, args.num_queries, args.num_verb_classes, matcher=matcher,
                                weight_dict=weight_dict, eos_coef=args.eos_coef, losses=losses,
                                args=args)
    criterion.to(device)
    postprocessors = {'hoi': PostProcessHOITriplet(args)}

    return model, criterion, postprocessors

