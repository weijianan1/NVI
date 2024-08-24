from pathlib import Path
from PIL import Image
import json
import numpy as np

import os
import torch
import torch.utils.data
import torchvision

import datasets.transforms as T

class VPIC(torch.utils.data.Dataset):

    def __init__(self, img_set, img_folder, anno_file, transforms, args):
        self.img_set = img_set
        self.img_folder = img_folder
        with open(anno_file, 'r') as f:
            self.annotations = json.load(f)
        self.filenames = list(self.annotations['_via_img_metadata'].keys())
        self._transforms = transforms
        self.individual_labels = ['expression', 'gesture', 'posture']
        self.group_labels = ['gaze', 'touch']
        self.labels = ['neutral', 'anger', 'smile', 'surprise', 'sadness', 'fear', 'disgust', 
                       'wave', 'point', 'beckon', 'palmout',
                       'arm-crossing', 'leg-crossing', 'slouching', 'arms-akimbo', 'bowing', 
                       'gaze-aversion', 'mutual-gaze', 'gaze-following', 
                       'hug', 'handshake', 'hit']
        self.labels_dict = {
            'expression': ['neutral', 'anger', 'smile', 'surprise', 'sadness', 'fear', 'disgust'],
            'gesture': ['wave', 'point', 'beckon', 'palmout'],
            'posture': ['arm-crossing', 'leg-crossing', 'slouching', 'arms-akimbo', 'bowing'],
            'gaze': ['gaze-aversion', 'mutual-gaze', 'gaze-following'],
            'touch': ['hug', 'handshake', 'hit'],
        }

        device = "cuda" if torch.cuda.is_available() else "cpu"

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        img_anno = self.annotations['_via_img_metadata'][filename]

        img = Image.open(self.img_folder / img_anno['filename']).convert('RGB')
        w, h = img.size

        boxes = []
        for region in img_anno['regions']:
            shape_attributes = region['shape_attributes']

            boxes.append([shape_attributes['x'], shape_attributes['y'], 
                          shape_attributes['x'] + shape_attributes['width'], 
                          shape_attributes['y'] + shape_attributes['height']])

        target = {}
        target['orig_size'] = torch.as_tensor([int(h), int(w)])
        target['size'] = torch.as_tensor([int(h), int(w)])
        target['boxes'] = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)

        if self._transforms is not None:
            if self.img_set == 'train':
                img, target = self._transforms[0](img, target)
            else:
                img, _ = self._transforms(img, None)
        boxes = target['boxes']

        inst_boxes, group_boxes = [], []
        inst_group_pairs = []
        attribute_labels = []
        for inst_id, region in enumerate(img_anno['regions']):
            region_attributes = region['region_attributes']

            for category in region_attributes.keys():
                if len(region_attributes[category].keys()) == 0:
                    continue

                if category in self.individual_labels:
                    inst_group = [inst_id]
                    if inst_group in inst_group_pairs:
                        attribute_label = attribute_labels[inst_group_pairs.index(inst_group)]
                        for atomic_index in region_attributes[category].keys():
                            atomic_class = self.labels_dict[category][int(atomic_index)-1]
                            attribute_label[self.labels.index(atomic_class)] = 1
                        attribute_labels[inst_group_pairs.index(inst_group)] = attribute_label
                    else:
                        inst_group_pairs.append(inst_group)
                        attribute_label = [0] * len(self.labels)
                        for atomic_index in region_attributes[category].keys():
                            atomic_class = self.labels_dict[category][int(atomic_index)-1]
                            attribute_label[self.labels.index(atomic_class)] = 1
                        attribute_labels.append(attribute_label)
                        inst_boxes.append(torch.as_tensor(boxes[inst_id], dtype=torch.float32))
                        group_boxes.append(torch.as_tensor(boxes[inst_id], dtype=torch.float32))
                else:
                    inst_group = [inst_id]
                    g_bbox = boxes[inst_id]
                    if "{}-group".format(category) in region_attributes.keys():
                        for key in region_attributes["{}-group".format(category)].keys():
                            if int(key) <= len(boxes):
                                inst_group.append(int(key)-1)
                                g_bbox = [min(g_bbox[0], boxes[int(key)-1][0]), 
                                        min(g_bbox[1], boxes[int(key)-1][1]), 
                                        max(g_bbox[2], boxes[int(key)-1][2]), 
                                        max(g_bbox[3], boxes[int(key)-1][3])]
                        if inst_group in inst_group_pairs:
                            attribute_label = attribute_labels[inst_group_pairs.index(inst_group)]
                            for atomic_index in region_attributes[category].keys():
                                atomic_class = self.labels_dict[category][int(atomic_index)-1]
                                attribute_label[self.labels.index(atomic_class)] = 1
                            attribute_labels[inst_group_pairs.index(inst_group)] = attribute_label
                        else:
                            inst_group_pairs.append(inst_group)
                            attribute_label = [0] * len(self.labels)
                            for atomic_index in region_attributes[category].keys():
                                atomic_class = self.labels_dict[category][int(atomic_index)-1]
                                attribute_label[self.labels.index(atomic_class)] = 1
                            attribute_labels.append(attribute_label)
                            inst_boxes.append(torch.as_tensor(boxes[inst_id], dtype=torch.float32))
                            group_boxes.append(torch.as_tensor(g_bbox, dtype=torch.float32))

        if self.img_set == 'train':
            if len(inst_group_pairs) == 0:
                target['attributes'] = torch.zeros((0, 22), dtype=torch.float32)
                target['inst_boxes'] = torch.zeros((0, 4), dtype=torch.float32)
                target['group_boxes'] = torch.zeros((0, 4), dtype=torch.float32)
            else:
                target['attributes'] = torch.as_tensor(attribute_labels, dtype=torch.float32)
                target['inst_boxes'] = torch.stack(inst_boxes)
                target['group_boxes'] = torch.stack(group_boxes)

            if self._transforms is not None:
                img, target = self._transforms[1](img, target)
            target['filename'] = img_anno['filename']

            return img, target
        else:
            target['id'] = idx
            target['filename'] = img_anno['filename']

            unique_insts = []
            unique_boxes = []
            for group_id, (inst_ids, boxes) in enumerate(zip(inst_group_pairs, group_boxes)):
                _inst_ids = inst_ids.copy()
                _inst_ids.sort(reverse=False)
                if inst_ids not in unique_insts:
                    unique_insts.append(_inst_ids)
                    unique_boxes.append(boxes)

            if len(group_boxes) > 0:
                target['group_boxes'] = torch.stack(unique_boxes)
            else:
                target['group_boxes'] = torch.zeros((0, 4), dtype=torch.float32)

            group_pairs = []
            for group_id, (inst_ids, attribute_label) in enumerate(zip(inst_group_pairs, attribute_labels)):
                non_zero_indexs = np.where(np.array(attribute_label) != 0)[0]
                for index in non_zero_indexs:
                    if index < 16:
                        group_pairs.append([inst_ids[0], -1, index])
                    else:
                        _inst_ids = inst_ids.copy()
                        _inst_ids.sort(reverse=False)
                        group_pairs.append([inst_ids[0], unique_insts.index(_inst_ids), index])
            target['group_pairs'] = group_pairs

            return img, target

def make_vcoco_transforms(image_set):
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == 'train':
        return [
            T.Compose([
                T.RandomHorizontalFlip(),
                T.ColorJitter(.4, .4, .4),
                T.RandomResize(scales, max_size=1333),
                # T.RandomSelect(
                #     T.RandomResize(scales, max_size=1333),
                #     T.Compose([
                #         T.RandomResize([400, 500, 600]),
                #         T.RandomSizeCrop(384, 600),
                #         T.RandomResize(scales, max_size=1333),
                #     ])
                # ),
            ]),
            T.Compose([normalize,]),
        ]

    if image_set == 'val' or image_set == 'test':
        return T.Compose([
                T.RandomResize([800], max_size=1333),
                normalize,
            ])

    raise ValueError(f'unknown {image_set}')

def build(image_set, args):
    root = Path(args.data_path)
    assert root.exists(), f'provided data path {root} does not exist'
    PATHS = {
        'train': (root / 'images' / 'train', root / 'annotations' / 'train.json'),
        'val': (root / 'images' / 'val', root / 'annotations' / 'val.json'),
        'test': (root / 'images' / 'test', root / 'annotations' / 'test.json')
    }

    img_folder, anno_file = PATHS[image_set]
    dataset = VPIC(image_set, img_folder, anno_file, transforms=make_vcoco_transforms(image_set), args=args)

    return dataset






