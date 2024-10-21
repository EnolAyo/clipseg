r""" COCO-20i few-shot semantic segmentation dataset """
import os
from pycocotools.coco import COCO
import pycocotools.mask as mask_util
from torch.utils.data import Dataset
import torch.nn.functional as F
import torch
import PIL.Image as Image
import numpy as np
import random


class DatasetCOCO(Dataset):
    def __init__(self, datapath, transform, split, use_original_imgsize):
        self.split = split
        self.nclass = 4
        self.benchmark = 'coco_severstal'
        self.base_path = datapath
        self.transform = transform
        self.use_original_imgsize = use_original_imgsize

        self.class_ids = [1, 2, 3, 4]
        self.img_metadata_classwise = self.build_img_metadata_classwise()
        self.len = self.__len__()

    def __len__(self):
        return len(self.img_metadata_classwise.anns)

    def __getitem__(self, idx):
        # ignores idx during training & testing and perform uniform sampling over object classes to form an episode
        # (due to the large size of the COCO dataset)
        query_img, query_mask, support_imgs, support_masks, query_name, support_names, class_sample, org_qry_imsize = self.load_frame(idx)

        query_img = self.transform(query_img)
        query_mask = query_mask.float()
        if not self.use_original_imgsize:
            query_mask = F.interpolate(query_mask.unsqueeze(0).unsqueeze(0).float(), query_img.size()[-2:], mode='nearest').squeeze()

        support_imgs = torch.stack([self.transform(support_img) for support_img in support_imgs])
        for midx, smask in enumerate(support_masks):
            support_masks[midx] = F.interpolate(smask.unsqueeze(0).unsqueeze(0).float(), support_imgs.size()[-2:], mode='nearest').squeeze()
        support_masks = torch.stack(support_masks)

        batch = {'query_img': query_img,
                 'query_mask': query_mask,
                 'query_name': query_name,

                 'org_query_imsize': org_qry_imsize,

                 'support_imgs': support_imgs,
                 'support_masks': support_masks,
                 'support_names': support_names,
                 'class_id': torch.tensor(class_sample)}

        return batch


    def build_img_metadata_classwise(self):
        coco = COCO(f"/home/eas/Enol/pycharm_projects/clipseg/third_party/Severstal/annotations_COCO_{self.split}.json")
        return coco

    def read_mask(self, rle_code):
        binary_mask = mask_util.decode(rle_code)
        binary_mask[binary_mask != 0] = 1
        mask = torch.tensor(binary_mask)
        return mask

    def load_frame(self):
        metadata = self.img_metadata_classwise
        query = random.choice(metadata)
        class_sample = query['category_id']
        query_name = query['image_id']

        query_img = Image.open(os.path.join(self.base_path, query_name)).convert('RGB')
        rle_mask = query['segmentation']
        query_mask = self.read_mask(rle_mask)

        org_qry_imsize = query_img.size
        n_samples = 0

        for i, ann in enumerate(metadata.anns):
            if ann['category_id'] == class_sample:
                n_samples += 1

        support_samples = []
        while True:  # keep sampling support set if query == support
            support = random.choice(metadata)
            support_name = support['image_id']
            if query_name != support_name:
                support_samples.append(support)

            if len(support_samples) == self.shot or len(support_samples) == n_samples - 1:
                break

        support_imgs = []
        support_masks = []
        support_names = []
        for support in support_samples:
            support_name = support['image_id']
            support_names.append(support_name)
            support_imgs.append(Image.open(os.path.join(self.base_path, support_name)).convert('RGB'))
            support_mask_rle = support['segmentation']
            support_mask = self.read_mask(support_mask_rle)
            support_masks.append(support_mask)

        return query_img, query_mask, support_imgs, support_masks, query_name, support_names, class_sample, org_qry_imsize

