import torch
import cv2
from torch.utils.data import Dataset
import os
import numpy as np
import albumentations as A

def get_augmentations(augmentation, img_size):
    if augmentation == 'geometric':
        geometric_transform = A.Compose(
        [
            A.Resize(height=img_size, width=img_size),
            A.Transpose(p=0.5),
            A.OneOf([
                A.HorizontalFlip(),
                A.VerticalFlip(),
            ], p=0.5),
            A.RandomRotate90(p=0.5),

            A.ShiftScaleRotate(p=0.1),
            A.GridDistortion(p=0.1),
            A.ElasticTransform(p=0.1),  
        ])
        color_transform = None
    elif augmentation == 'color':
        geometric_transform = A.Resize(height=img_size, width=img_size)
        color_transform = A.Compose(
        [
            A.OneOf([
                A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20),
                A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
            ], p=0.5),
            A.RandomGamma(p=0.5),
        ])
    elif augmentation == 'full':
        geometric_transform = A.Compose(
        [
            A.Resize(height=img_size, width=img_size),
            A.Transpose(p=0.5),
            A.OneOf([
                A.HorizontalFlip(),
                A.VerticalFlip(),
            ], p=0.5),
            A.RandomRotate90(p=0.5),

            A.ShiftScaleRotate(p=0.1),
            A.GridDistortion(p=0.1),
            A.ElasticTransform(p=0.1),
        ])
        color_transform = A.Compose(
        [
            A.OneOf([
                A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20),
                A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
            ], p=0.5),
            A.RandomGamma(p=0.5),
        ])
    elif augmentation == 'none': 
        geometric_transform = A.Resize(height=img_size, width=img_size)
        color_transform = None
    else:
        raise ValueError("Invalid value for augment")
    return geometric_transform, color_transform
class ISICDataset(Dataset):
    def __init__(self, images_path, masks_path, ids, size, geometric_transform=None, color_transform=None):
        self.images_path = images_path
        self.masks_path = masks_path
        self.geometric_transform = geometric_transform
        self.color_transform = color_transform
        self.ids = ids
        self.size = size

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        image_path = os.path.join(self.images_path, self.ids[idx] + '.jpg')
        mask_path = os.path.join(self.masks_path, self.ids[idx] + '_segmentation.png')

        # Load image and mask
        img = cv2.imread(os.path.join(self.images_path, self.ids[idx] + '.jpg'), cv2.IMREAD_COLOR)
        mask = cv2.imread(os.path.join(self.masks_path, self.ids[idx] + '_segmentation.png'), cv2.IMREAD_GRAYSCALE)
        
        # Convert to RGB, And convert mask to binary
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ret, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        
        if self.geometric_transform is not None:
            geometric_augmentations = self.geometric_transform(image=img, mask=mask)
            img = geometric_augmentations['image']
            mask = geometric_augmentations['mask']
        if self.color_transform is not None:
            color_augmentation = self.color_transform(image=img)
            img = color_augmentation['image']
        # Convert numpy arrays to PyTorch tensors
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.
        mask = torch.from_numpy(mask).unsqueeze(0).float()
        mask[mask == 255.0] = 1.0

        return img, mask