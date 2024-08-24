import torch.utils.data
import torchvision

from .nvi import build as build_nvi

def build_dataset(image_set, args):
    if args.dataset_file == 'nvi':
        return build_nvi(image_set, args)
    raise ValueError(f'dataset {args.dataset_file} not supported')


