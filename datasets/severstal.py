import pickle
from types import new_class
import torch
import numpy as np
import os
import json
from pycocotools.coco import COCO
from os.path import join, dirname, isdir, isfile, expanduser, realpath, basename
from random import shuffle, seed as set_seed
from PIL import Image

from itertools import combinations
from torchvision import transforms
from torchvision.transforms.transforms import Resize

from datasets.utils import blend_image_segmentation
from general_utils import get_from_repository

COCO_CLASSES = {1: 'class_1', 2: 'class_2', 3: 'class_3', 4: 'class_4'}


class COCOWrapper(object):

    def __init__(self, split, image_size=256, aug=None, mask='text_and_blur3_highlight01', negative_prob=0,
                 with_class_label=False):
        super().__init__()

        self.mask = mask
        self.with_class_label = with_class_label
        self.negative_prob = negative_prob
        self.split = split

        from third_party.Severstal.severstal_coco import DatasetCOCO



        metadatapath = f'/home/eas/Enol/pycharm_projects/clipseg/third_party/Severstal/annotations_COCO_{self.split}.json'
        datapath = '/home/eas/Enol/pycharm_projects/clipseg/third_party/Severstal/train_subimages'

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        self.coco = DatasetCOCO(datapath, transform, split,3,True)

        self.all_classes = [self.coco.class_ids]
        self.coco.base_path = datapath

    def __len__(self):
        return len(self.coco)

    def __getitem__(self, i):
        sample = self.coco[i]

        label_name = COCO_CLASSES[int(sample['class_id'])]

        img_s, seg_s = sample['support_imgs'][0], sample['support_masks'][0]

        if self.negative_prob > 0 and torch.rand(1).item() < self.negative_prob:
            new_class_id = sample['class_id']
            while new_class_id == sample['class_id']:
                sample2 = self.coco[torch.randint(0, len(self), (1,)).item()]
                new_class_id = sample2['class_id']
            img_s = sample2['support_imgs'][0]
            seg_s = torch.zeros_like(seg_s)

        mask = self.mask
        if mask == 'separate':
            supp = (img_s, seg_s)
        elif mask == 'text_label':
            # DEPRECATED
            supp = [int(sample['class_id'])]
        elif mask == 'text':
            supp = [label_name]
        else:
            if mask.startswith('text_and_'):
                mask = mask[9:]
                label_add = [label_name]
            else:
                label_add = []

            supp = label_add + blend_image_segmentation(img_s, seg_s, mode=mask)

        if self.with_class_label:
            label = (torch.zeros(0), sample['class_id'],)
        else:
            label = (torch.zeros(0),)

        return (sample['query_img'],) + tuple(supp), (sample['query_mask'].unsqueeze(0),) + label