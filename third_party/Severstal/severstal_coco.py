r""" COCO-20i few-shot semantic segmentation dataset """
import os
from pycocotools.coco import COCO
import pycocotools.mask as mask_util
from torch.utils.data import Dataset
import torch.nn.functional as F
import torch
import PIL.Image as Image
from collections import Counter
import random


class DatasetCOCO(Dataset):
    def __init__(self, datapath, transform, split, use_original_imgsize, random_seed=33):
        self.split = split
        self.nclass = 5
        self.benchmark = 'coco_severstal'
        self.base_path = datapath
        self.transform = transform
        self.use_original_imgsize = use_original_imgsize
        self.random_seed = random_seed

        self.class_ids = [1, 2, 3, 4, 5]
        self.img_metadata, self.train_ids, self.val_ids = self.build_img_metadata()
        self.len = self.__len__()
        self.duplicates = self.find_duplicates()

    def __len__(self):
        if self.split == 'train':
            return len(self.train_ids)
        else:
            return len(self.val_ids)

    def __getitem__(self, idx):
        # ignores idx during training & testing and perform uniform sampling over object classes to form an episode
        # (due to the large size of the COCO dataset)
        query_img, query_mask, support_imgs, support_masks, query_name, support_names, class_sample, org_qry_imsize = self.load_frame()

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


    def build_img_metadata(self):
        coco = COCO(f"/home/eas/Enol/pycharm_projects/clipseg/third_party/Severstal/annotations_COCO.json")
        random.seed(self.random_seed)
        split_ratio = 0.7
        keys = list(coco.anns.keys())
        random.shuffle(keys)
        split_point = int(len(keys) * split_ratio)
        keys_train = keys[:split_point]
        train_ids = [coco.anns[i]['id'] for i in keys_train]
        keys_val = keys[split_point:]
        val_ids = [coco.anns[i]['id'] for i in keys_val]
        random.seed(None)
        return coco, train_ids, val_ids


    def find_duplicates(self):
        id_list = []
        for i in range(len(self.img_metadata.anns)):
            id_list.append(self.img_metadata.anns[i]['image_id'])

        counts = Counter(id_list)
        duplicates = [item for item, frequency in counts.items() if frequency > 1]
        return duplicates


    def read_mask(self, rle_code):
        binary_mask = mask_util.decode(rle_code)
        binary_mask[binary_mask != 0] = 1
        mask = torch.tensor(binary_mask)
        return mask

    def load_frame(self):
        metadata = self.img_metadata
        train_ids = self.train_ids
        val_ids = self.val_ids
        if self.split == 'train':
            metadata = metadata.loadAnns(ids=train_ids)
        else:
            metadata = metadata.loadAnns(val_ids)

        query = random.choice(metadata)
        class_sample = query['category_id']
        query_name = query['image_id']

        query_img = Image.open(os.path.join(self.base_path, query_name)).convert('RGB')
        rle_mask = query['segmentation']
        query_mask = self.read_mask(rle_mask)

        org_qry_imsize = query_img.size
        n_samples = 0

        for i, ann in enumerate(metadata.values()):
            if ann['category_id'] == class_sample:
                n_samples += 1

        while True:
            support_samples = []  # keep sampling support set if query == support
            support = random.choice(metadata)
            support_name = support['image_id']
            support_class = support['category_id']
            if class_sample in [1, 2, 3, 4]:
                if query_name != support_name and class_sample == support_class:
                    support_samples.append(support)
                    break
            else: # query with no defect
                if query_name != support_name and class_sample in [1, 2, 3, 4]:
                    support_samples.append(support)
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

