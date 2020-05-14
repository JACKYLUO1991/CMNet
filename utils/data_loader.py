# -*- coding: utf-8 -*-
# @Author: luoling
# @Date:   2019-09-19 17:19:32
# @Last Modified by:   luoling
# @Last Modified time: 2019-09-19 20:58:33

import os
import os.path as osp
from PIL import Image, ImageOps
import numpy as np
import random
# import cv2 as cv

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF


# def randomShiftScaleRotate(image, label,
#                            shift_limit=(-0.0, 0.0),
#                            scale_limit=(-0.0, 0.0),
#                            rotate_limit=(-0.0, 0.0),
#                            aspect_limit=(-0.0, 0.0),
#                            borderMode=cv.BORDER_CONSTANT):
#     # https://github.com/Guzaiwang/CE-Net/blob/master/data.py
#     if random.random() > 0.5:
#         image = cv.cvtColor(np.array(image), cv.COLOR_RGB2BGR)
#         label = np.array(label)
#         height, width, channel = image.shape

#         angle = np.random.uniform(rotate_limit[0], rotate_limit[1])
#         scale = np.random.uniform(1 + scale_limit[0], 1 + scale_limit[1])
#         aspect = np.random.uniform(1 + aspect_limit[0], 1 + aspect_limit[1])
#         sx = scale * aspect / (aspect ** 0.5)
#         sy = scale / (aspect ** 0.5)
#         dx = round(np.random.uniform(shift_limit[0], shift_limit[1]) * width)
#         dy = round(np.random.uniform(shift_limit[0], shift_limit[1]) * height)

#         cc = np.math.cos(angle / 180 * np.math.pi) * sx
#         ss = np.math.sin(angle / 180 * np.math.pi) * sy
#         rotate_matrix = np.array([[cc, -ss], [ss, cc]])

#         box0 = np.array([[0, 0], [width, 0], [width, height], [0, height], ])
#         box1 = box0 - np.array([width / 2, height / 2])
#         box1 = np.dot(box1, rotate_matrix.T) + \
#                np.array([width / 2 + dx, height / 2 + dy])

#         box0 = box0.astype(np.float32)
#         box1 = box1.astype(np.float32)
#         mat = cv.getPerspectiveTransform(box0, box1)
#         image = cv.warpPerspective(image, mat, (width, height), flags=cv.INTER_LINEAR, borderMode=borderMode,
#                                    borderValue=(
#                                        0, 0,
#                                        0,))
#         label = cv.warpPerspective(label, mat, (width, height), flags=cv.INTER_LINEAR, borderMode=borderMode,
#                                    borderValue=(
#                                        0, 0,
#                                        0,))
#         image = Image.fromarray(cv.cvtColor(image, cv.COLOR_BGR2RGB))
#         label = Image.fromarray(label)

#     return image, label


def augmentation(img, mask):
    """Image and mask augmentation"""

    if random.random() > 0.5:
        img = transforms.ColorJitter(brightness=.3, contrast=.3,
                                     saturation=0.02, hue=0.02)(img)
    if random.random() > 0.5:
        img = TF.hflip(img)
        mask = TF.hflip(mask)
    if random.random() > 0.5:
        img = TF.vflip(img)
        mask = TF.vflip(mask)
    if random.random() > 0.5:
        angle = random.randrange(-180, 181, 5)
        img = TF.rotate(img, angle, resample=Image.BILINEAR)
        mask = TF.rotate(mask, angle, resample=Image.NEAREST, fill=(0,))

    return img, mask


class DRIVE_Dataset(Dataset):
    """
    DRIVE datset reading
    """

    def __init__(self, root_dir, resize=None, mode='train'):
        assert mode.lower() == 'train' or mode.lower() == 'test'

        if mode == 'train':
            image_root = osp.join(root_dir, "train/images")
            label_root = osp.join(root_dir, "train/masks")
        else:
            image_root = osp.join(root_dir, "test/images")
            label_root = osp.join(root_dir, "test/masks")

        image_list = []
        label_list = []

        for image_name in os.listdir(image_root):
            image_path = osp.join(image_root, image_name)
            label_path = osp.join(
                label_root, image_name.split('_')[0] + '_manual1.gif')
            image_list.append(image_path)
            label_list.append(label_path)

        self._image_list = image_list
        self._label_list = label_list
        # self._crop_size = crop_size
        self._resize = resize
        self.mode = mode

    def __len__(self):
        return len(self._image_list)

    def __getitem__(self, idx):
        img = Image.open(self._image_list[idx]).convert('RGB')
        label = Image.open(self._label_list[idx]).convert('L')

        # if self._crop_size is not None and isinstance(self._crop_size, tuple):
        #     img = TF.center_crop(img, self._crop_size)
        #     label = TF.center_crop(label, self._crop_size)

        if self._resize is not None and isinstance(self._resize, tuple):
            img = TF.resize(img, self._resize, Image.BILINEAR)
            label = TF.resize(label, self._resize, Image.NEAREST)  # Pay Attention...

        if self.mode.lower() == 'train':
            # # Color jitter
            # img = TF.ColorJitter(brightness=.3, contrast=.3,
            #                      saturation=0.02, hue=0.02)(img)

            # Scale translation transformation
            # img, label = randomShiftScaleRotate(img, label,
            #                                     rotate_limit=(-30, 30))
            img, label = augmentation(img, label)

        img = np.array(img, dtype=np.float32)
        label = np.expand_dims(label, axis=-1)
        label = np.array(label, dtype=np.float32).transpose(2, 0, 1) / 255.
        label[label >= 0.5] = 1
        label[label < 0.5] = 0

        # Convert to the tensor format required by pytorch
        img = TF.to_tensor(img)
        label = torch.Tensor(label)

        return img, label


class CHASEDB1_Dataset(Dataset):
    """
    CHASEBD1 datset reading
    """

    def __init__(self, root_dir, resize=None, mode='train'):
        assert mode.lower() == 'train' or mode.lower() == 'test'

        if mode == 'train':
            image_root = osp.join(root_dir, "train/images")
            label_root = osp.join(root_dir, "train/masks")
        else:
            image_root = osp.join(root_dir, "test/images")
            label_root = osp.join(root_dir, "test/masks")

        image_list = []
        label_list = []

        for image_name in os.listdir(image_root):
            image_path = osp.join(image_root, image_name)
            label_path = osp.join(
                label_root, image_name.split('.')[0] + '_1stHO.png')
            image_list.append(image_path)
            label_list.append(label_path)

        self._image_list = image_list
        self._label_list = label_list
        # self._crop_size = crop_size
        self._resize = resize
        self.mode = mode

    def __len__(self):
        return len(self._image_list)

    def __getitem__(self, idx):
        img = Image.open(self._image_list[idx]).convert('RGB')
        label = Image.open(self._label_list[idx]).convert('L')

        # # Image center crop
        # if self._crop_size is not None and isinstance(self._crop_size, tuple):
        #     img = TF.center_crop(img, self._crop_size)
        #     label = TF.center_crop(label, self._crop_size)

        if self._resize is not None and isinstance(self._resize, tuple):
            img = TF.resize(img, self._resize)
            label = TF.resize(label, self._resize)

        if self.mode.lower() == 'train':
            # # Color jitter
            # img = TF.ColorJitter(brightness=.3, contrast=.3,
            #                      saturation=0.02, hue=0.01)(img)

            # # Scale translation transformation
            # img, label = randomShiftScaleRotate(img, label, shift_limit=(-0.1, 0.1),
            #                                     scale_limit=(-0.1, 0.1),
            #                                     aspect_limit=(-0.1, 0.1),
            #                                     rotate_limit=(-30, 30)
            #                                     )
            # img, label = self.__random_hflip(img, label)
            # img, label = self.__random_vflip(img, label)
            img, label = augmentation(img, label)

        label = np.expand_dims(label, axis=-1)
        img = np.array(img, dtype=np.float32)
        label = np.array(label, dtype=np.float32).transpose(2, 0, 1) / 255.
        label[label > 0.5] = 1
        label[label <= 0.5] = 0

        # Convert to the tensor format required by pytorch
        img = TF.to_tensor(img)
        label = torch.Tensor(label)

        return img, label

    # def __random_hflip(self, img, label):
    #     if random.random() > 0.5:
    #         img = TF.hflip(img)
    #         label = TF.hflip(label)

    #     return img, label

    # def __random_vflip(self, img, label):
    #     if random.random() > 0.5:
    #         img = TF.vflip(img)
    #         label = TF.vflip(label)

    #     return img, label


class HRF_Dataset(Dataset):
    """
    HRF datset reading
    """

    def __init__(self, root_dir, resize=None, mode='train'):
        assert mode.lower() == 'train' or mode.lower() == 'test'

        if mode == 'train':
            image_root = osp.join(root_dir, "train/images")
            label_root = osp.join(root_dir, "train/masks")
        else:
            image_root = osp.join(root_dir, "test/images")
            label_root = osp.join(root_dir, "test/masks")

        image_list = []
        label_list = []

        for image_name in os.listdir(image_root):
            image_path = osp.join(image_root, image_name)
            label_path = osp.join(
                label_root, image_name.split('.')[0] + '.tif')
            image_list.append(image_path)
            label_list.append(label_path)

        self._image_list = image_list
        self._label_list = label_list
        # self._crop_size = crop_size
        self._resize = resize
        self.mode = mode

    def __len__(self):
        return len(self._image_list)

    def __getitem__(self, idx):
        img = Image.open(self._image_list[idx]).convert('RGB')
        label = Image.open(self._label_list[idx]).convert('L')

        # # Image center crop
        # if self._crop_size is not None and isinstance(self._crop_size, tuple):
        #     img = TF.center_crop(img, self._crop_size)
        #     label = TF.center_crop(label, self._crop_size)

        if self._resize is not None and isinstance(self._resize, tuple):
            img = TF.resize(img, self._resize)
            label = TF.resize(label, self._resize)

        if self.mode.lower() == 'train':
            # # Color jitter
            # img = TF.ColorJitter(brightness=.3, contrast=.3,
            #                      saturation=0.02, hue=0.01)(img)

            # # Scale translation transformation
            # img, label = randomShiftScaleRotate(img, label, shift_limit=(-0.1, 0.1),
            #                                     scale_limit=(-0.1, 0.1),
            #                                     aspect_limit=(-0.1, 0.1),
            #                                     rotate_limit=(-30, 30)
            #                                     )
            # img, label = self.__random_hflip(img, label)
            # img, label = self.__random_vflip(img, label)
            img, label = augmentation(img, label)

        label = np.expand_dims(label, axis=-1)
        img = np.array(img, dtype=np.float32)
        label = np.array(label, dtype=np.float32).transpose(2, 0, 1) / 255.
        label[label > 0.5] = 1
        label[label <= 0.5] = 0

        # Convert to the tensor format required by pytorch
        img = TF.to_tensor(img)
        label = torch.Tensor(label)

        return img, label

    # def __random_hflip(self, img, label):
    #     if random.random() > 0.5:
    #         img = TF.hflip(img)
    #         label = TF.hflip(label)

    #     return img, label

    # def __random_vflip(self, img, label):
    #     if random.random() > 0.5:
    #         img = TF.vflip(img)
    #         label = TF.vflip(label)

    #     return img, label


def get_loader(data_path, resize, batch_size, shuffle=False, dataset_name='DRIVE', num_workers=1,
               mode='train'):
    # Data generation iterator
    if dataset_name == 'DRIVE':
        dataset = DRIVE_Dataset(data_path, resize, mode)
    elif dataset_name == 'CHASEDB1':
        dataset = CHASEDB1_Dataset(data_path, resize, mode)
    elif dataset_name == 'HRF':
        dataset = HRF_Dataset(data_path, resize, mode)
    else:
        raise Exception("No this dataset...")

    data_loader = DataLoader(dataset=dataset, batch_size=batch_size,
                             shuffle=shuffle, num_workers=num_workers, pin_memory=True)

    return data_loader
