# ------------------------------------------------------------------------
# Copyright (c) Hitachi, Ltd. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
import numpy as np
from collections import defaultdict

import util.misc as utils

class NVIEvaluator():

    def __init__(self, preds, gts, LOGGER, args, overlap_iou=0.5, topK=100):
        self.overlap_iou = overlap_iou
        self.overlap_iou = [0.25, 0.50, 0.75]
        self.topK = [25, 50, 100]
        self.LOGGER = LOGGER
        self.args = args

        self.fp = defaultdict(list)
        self.tp = defaultdict(list)
        self.score = defaultdict(list)
        self.sum_gts = defaultdict(lambda: 0)

        self.num_classes = {
            "expression": 7,
            "gesture": 4,
            "posture": 5,
            "gaze": 3,
            "touch": 3, 
        }

        self.preds = []
        for img_preds in preds:
            img_preds = {k: v.to('cpu').numpy() for k, v in img_preds.items()}

            inst_ids = img_preds['inst_ids']
            group_ids = img_preds['group_ids']
            nvi_scores = img_preds['nvi_scores']
            # nvi_scores = img_preds['nvi_scores'] * img_preds['obj_scores']

            nvi_labels = np.tile(np.arange(nvi_scores.shape[1]), (nvi_scores.shape[0], 1))
            inst_ids = np.tile(img_preds['inst_ids'], (nvi_scores.shape[1], 1)).T
            group_ids = np.tile(img_preds['group_ids'], (nvi_scores.shape[1], 1)).T

            nvi_scores = nvi_scores.ravel()
            nvi_labels = nvi_labels.ravel()
            inst_ids = inst_ids.ravel()
            group_ids = group_ids.ravel()

            if len(inst_ids) > 0:
                nvis = [{'inst_id': subject_id, 'group_id': object_id, 'category_id': category_id, 'score': score} for
                        subject_id, object_id, category_id, score in zip(inst_ids, group_ids, nvi_labels, nvi_scores)]
                nvis.sort(key=lambda k: (k.get('score', 0)), reverse=True)
            else:
                nvis = []

            self.preds.append({
                'inst_predictions': [{'bbox': inst_box} for inst_box in img_preds['inst_boxes']],
                'group_predictions': [{'bbox': group_box} for group_box in img_preds['group_boxes']],
                'nvi_prediction': nvis,
            })

        self.gts = []
        for img_gts in gts:
            self.gts.append({
                'inst_annotations': [{'bbox': inst_box} for inst_box in img_gts['boxes'].to('cpu').numpy()],
                'group_annotations': [{'bbox': group_box} for group_box in img_gts['group_boxes'].to('cpu').numpy()],
                'nvi_annotations': [{'inst_id': pair[0], 'group_id': pair[1], 'category_id': pair[2]} for pair in img_gts['group_pairs']],
            })
            for rel in self.gts[-1]['nvi_annotations']:
                self.sum_gts[rel['category_id']] += 1

    def evaluate(self):
        mRecall_ind, mRecall_gro, mRecall_all = 0, 0, 0
        map_all = None
        for topK in self.topK:
            map = None
            for overlap_iou in self.overlap_iou:
                for img_preds, img_gts in zip(self.preds, self.gts):
                    pred_inst_bboxes = img_preds['inst_predictions']
                    gt_inst_bboxes = img_gts['inst_annotations']
                    pred_group_bboxes = img_preds['group_predictions']
                    gt_group_bboxes = img_gts['group_annotations']
                    pred_attrs = img_preds['nvi_prediction'][:topK]
                    gt_attrs = img_gts['nvi_annotations']

                    if len(gt_inst_bboxes) != 0:
                        bbox_inst_pairs, bbox_inst_overlaps = self.compute_iou_mat(gt_inst_bboxes, pred_inst_bboxes, overlap_iou)
                        bbox_group_pairs, bbox_group_overlaps = self.compute_iou_mat(gt_group_bboxes, pred_group_bboxes, overlap_iou)
                        self.compute_fptp(pred_attrs, gt_attrs, bbox_inst_pairs, bbox_inst_overlaps, bbox_group_pairs, bbox_group_overlaps)
                    else:
                        for pred_attr in pred_attrs:
                            self.tp[pred_attr['category_id']].append(0)
                            self.fp[pred_attr['category_id']].append(1)
                            self.score[pred_attr['category_id']].append(pred_attr['score'])

                if map is None:
                    map = self.compute_map()
                else:
                    _map = self.compute_map()
                    for key in _map.keys():
                        map[key] += _map[key]

                # reset
                self.fp = defaultdict(list)
                self.tp = defaultdict(list)
                self.score = defaultdict(list)

            # if self.args.output_dir and utils.is_main_process():
            print(topK, 'mRecall_ind', map['mRecall_ind'] / len(self.overlap_iou))
            print(topK, 'mRecall_gro', map['mRecall_gro'] / len(self.overlap_iou))
            print(topK, 'mRecall_all', map['mRecall_all'] / len(self.overlap_iou))
            mRecall_ind += map['mRecall_ind'] / len(self.overlap_iou)
            mRecall_gro += map['mRecall_gro'] / len(self.overlap_iou)
            mRecall_all += map['mRecall_all'] / len(self.overlap_iou)

            if map_all is None:
                map_all = map
            else:
                for key in map.keys():
                    map_all[key] += map[key]

        mRecall_ind /= len(self.topK)
        mRecall_gro /= len(self.topK)
        mRecall_all /= len(self.topK)

        print('Final: ')
        print('mRecall_ind', map_all['mRecall_ind'] / (len(self.overlap_iou) * len(self.topK)))
        print('mRecall_gro', map_all['mRecall_gro'] / (len(self.overlap_iou) * len(self.topK)))
        print('mRecall_all', map_all['mRecall_all'] / (len(self.overlap_iou) * len(self.topK)))

        result = {}
        result['mRecall_ind'] = map_all['mRecall_ind'] / (len(self.overlap_iou) * len(self.topK))
        result['mRecall_gro'] = map_all['mRecall_gro'] / (len(self.overlap_iou) * len(self.topK))
        result['mRecall_all'] = map_all['mRecall_all'] / (len(self.overlap_iou) * len(self.topK))
        for category_name in ["expression", "gesture", "posture", "gaze", "touch"]:
            result[category_name] = map_all[category_name] / (len(self.overlap_iou) * len(self.topK))


        return result

    def compute_map(self):
        recall = defaultdict(lambda: 0)
        recalls = {}
        for category_id in sorted(list(self.sum_gts.keys())):
            sum_gts = self.sum_gts[category_id]
            if sum_gts == 0:
                continue

            tp = np.array((self.tp[category_id]))
            fp = np.array((self.fp[category_id]))

            if len(tp) == 0:
                recall[category_id] = 0
            else:
                recall[category_id] = np.sum(tp) / sum_gts

            recalls['Recall_{}'.format(category_id)] = recall[category_id]

        m_recall_ind = np.mean([recall[category_id] for category_id in range(16)])
        m_recall_gro = np.mean([recall[category_id] for category_id in range(16, 22)])
        m_recall_all = np.mean(list(recall.values()))

        num_acc = 0
        for category_name in ["expression", "gesture", "posture", "gaze", "touch"]:
            num = self.num_classes[category_name]
            recalls.update({category_name: np.mean([recall[category_id] for category_id in range(num_acc, num_acc+num)])})
            num_acc += num

        recalls.update({'mRecall_ind': m_recall_ind})
        recalls.update({'mRecall_gro': m_recall_gro})
        recalls.update({'mRecall_all': m_recall_all})

        return recalls


    def compute_fptp(self, pred_attrs, gt_attrs, inst_match_pairs, inst_bbox_overlaps, group_match_pairs, group_bbox_overlaps):
        inst_pos_pred_ids = inst_match_pairs.keys()
        group_pos_pred_ids = group_match_pairs.keys()

        vis_tag = np.zeros(len(gt_attrs))
        pred_attrs.sort(key=lambda k: (k.get('score', 0)), reverse=True)

        if len(pred_attrs) != 0:
            for pred_attr in pred_attrs:
                is_match = 0
                max_overlap = 0
                max_gt_attr = 0
                for gt_attr in gt_attrs:
                    if len(inst_match_pairs) != 0 and pred_attr['inst_id'] in inst_pos_pred_ids and \
                       gt_attr['group_id'] == -1:
                        pred_inst_ids = inst_match_pairs[pred_attr['inst_id']]
                        pred_inst_overlaps = inst_bbox_overlaps[pred_attr['inst_id']]
                        pred_category_id = pred_attr['category_id']
                        if gt_attr['inst_id'] in pred_inst_ids and pred_category_id == gt_attr['category_id']:
                            is_match = 1
                            min_overlap_gt = pred_inst_overlaps[pred_inst_ids.index(gt_attr['inst_id'])]
                            if min_overlap_gt > max_overlap:
                                max_overlap = min_overlap_gt
                                max_gt_attr = gt_attr
                    elif len(inst_match_pairs) != 0 and len(group_match_pairs) != 0 and pred_attr['inst_id'] in inst_pos_pred_ids and \
                        pred_attr['group_id'] in group_pos_pred_ids:
                        pred_inst_ids = inst_match_pairs[pred_attr['inst_id']]
                        pred_group_ids = group_match_pairs[pred_attr['group_id']]
                        pred_inst_overlaps = inst_bbox_overlaps[pred_attr['inst_id']]
                        pred_group_overlaps = group_bbox_overlaps[pred_attr['group_id']]
                        pred_category_id = pred_attr['category_id']
                        if gt_attr['inst_id'] in pred_inst_ids and gt_attr['group_id'] in pred_group_ids and pred_category_id == gt_attr['category_id']:
                            is_match = 1
                            min_overlap_gt = min(pred_inst_overlaps[pred_inst_ids.index(gt_attr['inst_id'])],
                                                 pred_group_overlaps[pred_group_ids.index(gt_attr['group_id'])])
                            if min_overlap_gt > max_overlap:
                                max_overlap = min_overlap_gt
                                max_gt_attr = gt_attr
                if is_match == 1 and vis_tag[gt_attrs.index(max_gt_attr)] == 0:
                    self.fp[pred_attr['category_id']].append(0)
                    self.tp[pred_attr['category_id']].append(1)
                    vis_tag[gt_attrs.index(max_gt_attr)] = 1
                else:
                    self.fp[pred_attr['category_id']].append(1)
                    self.tp[pred_attr['category_id']].append(0)
                self.score[pred_attr['category_id']].append(pred_attr['score'])

    def compute_iou_mat(self, bbox_list1, bbox_list2, overlap_iou):
        iou_mat = np.zeros((len(bbox_list1), len(bbox_list2)))
        if len(bbox_list1) == 0 or len(bbox_list2) == 0:
            return {}, {}

        for i, bbox1 in enumerate(bbox_list1):
            for j, bbox2 in enumerate(bbox_list2):
                iou_i = self.compute_IOU(bbox1, bbox2)
                iou_mat[i, j] = iou_i

        iou_mat_ov=iou_mat.copy()
        iou_mat[iou_mat>=overlap_iou] = 1
        iou_mat[iou_mat<overlap_iou] = 0

        match_pairs = np.nonzero(iou_mat)
        match_pairs_dict = {}
        match_pair_overlaps = {}
        if iou_mat.max() > 0:
            for i, pred_id in enumerate(match_pairs[1]):
                if pred_id not in match_pairs_dict.keys():
                    match_pairs_dict[pred_id] = []
                    match_pair_overlaps[pred_id]=[]
                match_pairs_dict[pred_id].append(match_pairs[0][i])
                match_pair_overlaps[pred_id].append(iou_mat_ov[match_pairs[0][i], pred_id])
        return match_pairs_dict, match_pair_overlaps

    def compute_IOU(self, bbox1, bbox2):
        rec1 = bbox1['bbox']
        rec2 = bbox2['bbox']
        # computing area of each rectangles
        S_rec1 = (rec1[2] - rec1[0]+1) * (rec1[3] - rec1[1]+1)
        S_rec2 = (rec2[2] - rec2[0]+1) * (rec2[3] - rec2[1]+1)

        # computing the sum_area
        sum_area = S_rec1 + S_rec2

        # find the each edge of intersect rectangle
        left_line = max(rec1[1], rec2[1])
        right_line = min(rec1[3], rec2[3])
        top_line = max(rec1[0], rec2[0])
        bottom_line = min(rec1[2], rec2[2])
        # judge if there is an intersect
        if left_line >= right_line or top_line >= bottom_line:
            return 0
        else:
            intersect = (right_line - left_line+1) * (bottom_line - top_line+1)
            return intersect / (sum_area - intersect)


