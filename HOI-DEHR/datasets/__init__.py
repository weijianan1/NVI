import torch.utils.data
import torchvision

from .hico import build as build_hico
from .vcoco import build as build_vcoco

def build_dataset(vis_processors, image_set, args):
    if args.dataset_file == 'hico':
        return build_hico(vis_processors, image_set, args)
    if args.dataset_file == 'vcoco':
        return build_vcoco(vis_processors, image_set, args)
    raise ValueError(f'dataset {args.dataset_file} not supported')
