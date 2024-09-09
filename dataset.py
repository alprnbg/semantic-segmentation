import os, shutil
from glob import glob

import cv2
import numpy as np
import torch
import torch.utils.data
from albumentations.augmentations import transforms, geometric
import albumentations.augmentations
from albumentations.core.composition import Compose
from sklearn.model_selection import train_test_split


def fetch_image(img_path, cached_path, resize_w, resize_h):
    if os.path.exists(cached_path):
        img = cv2.cvtColor(cv2.imread(cached_path), cv2.COLOR_BGR2RGB)
    else:
        img = cv2.imread(img_path)
        if resize_h is not None:
            img = cv2.resize(img, (resize_w, resize_h))
            os.makedirs((os.sep).join(cached_path.split(os.sep)[:-1]), exist_ok=True)
            cv2.imwrite(cached_path, img) # caching
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def fetch_mask(img_id, mask_dir, mask_ext, num_classes, resize_w, resize_h, img_shape):
    mask = []
    for i in range(num_classes):
        if mask_dir is not None:
            org_mask_path = os.path.join(mask_dir, str(i), img_id + mask_ext)
            cached_mask_path = os.path.join(".data_resized_cache", org_mask_path)
            if os.path.exists(cached_mask_path):
                mask.append(cv2.imread(cached_mask_path, cv2.IMREAD_GRAYSCALE)[..., None])
            else:
                mask_img = cv2.imread(org_mask_path, cv2.IMREAD_GRAYSCALE)
                if resize_h is not None:
                    mask_img = cv2.resize(mask_img, (resize_w, resize_h))
                    os.makedirs((os.sep).join(cached_mask_path.split(os.sep)[:-1]), exist_ok=True)
                    cv2.imwrite(cached_mask_path, mask_img) # caching
                mask.append(mask_img[..., None])
        else:
            if resize_h is not None:
                mask.append(np.zeros((resize_w, resize_h, 1)))
            else:
                mask.append(np.zeros((img_shape[0], img_shape[1], 1)))
    mask = np.dstack(mask)
    return mask


class Dataset(torch.utils.data.Dataset):
    def __init__(self, img_ids, img_dir, mask_dir, img_ext, mask_ext, num_classes,
                 input_w, input_h, transform=None):
        self.img_ids = img_ids
        self.img_dir = os.path.relpath(img_dir)
        if mask_dir is None:
            self.mask_dir = None
        else:
            self.mask_dir = os.path.relpath(mask_dir)
        self.img_ext = img_ext
        self.mask_ext = mask_ext
        self.num_classes = num_classes
        self.transform = transform
        self.input_w = input_w
        self.input_h = input_h
        if os.path.exists(".data_resized_cache"):
            shutil.rmtree(".data_resized_cache")

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        org_img_path = os.path.join(self.img_dir, img_id + self.img_ext)
        cached_img_path = os.path.join(".data_resized_cache", org_img_path)
        img = fetch_image(org_img_path, cached_img_path, self.input_w, self.input_h)
        mask = fetch_mask(img_id, self.mask_dir, self.mask_ext, self.num_classes, self.input_w,
                          self.input_h, img.shape)
        if self.transform is not None:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']
        img = img.transpose(2, 0, 1)
        mask = (mask > 50).astype('float32')
        mask = mask.transpose(2, 0, 1)
        return img, mask, {'img_id': img_id}


def get_dataloaders(config, is_testing_enabled):
    # Data
    train_ratio = config["train_val_ratio"]
    subset_samples = None
    if config["subset_file"] is not None:
        with open(config["subset_file"], "r") as f:
            subset_samples = [s.strip() for s in f.readlines()]
    user_defined_sets = os.path.exists(os.path.join('datasets', config['dataset'], 'images', 'train'))
    if subset_samples is not None:
        # hypersearch
        if user_defined_sets:
            all_img_dir = os.path.join('datasets', config['dataset'], 'images', 'train')
            all_mask_dir = os.path.join('datasets', config['dataset'], 'masks', 'train')
        else:
            all_img_dir = os.path.join('datasets', config['dataset'], 'images')
            all_mask_dir = os.path.join('datasets', config['dataset'], 'masks')
        train_img_dir, val_img_dir = all_img_dir, all_img_dir
        train_mask_dir, val_mask_dir = all_mask_dir, all_mask_dir
        train_val_img_ids = []
        for path in glob(os.path.join(all_img_dir, '*' + config['img_ext'])):
            if path.split(os.sep)[-1] in subset_samples:
                train_val_img_ids.append(path)
        train_val_img_ids = [os.path.splitext(os.path.basename(p))[0] for p in train_val_img_ids]
        train_img_ids, val_img_ids = train_test_split(train_val_img_ids, test_size=1-train_ratio,
                                                      random_state=41)
    else:
        # no hypersearch
        if user_defined_sets:
            # User-defined train and val sets
            train_img_dir = os.path.join('datasets', config['dataset'], 'images', 'train')
            val_img_dir = os.path.join('datasets', config['dataset'], 'images', 'val')
            train_mask_dir = os.path.join('datasets', config['dataset'], 'masks', 'train')
            val_mask_dir = os.path.join('datasets', config['dataset'], 'masks', 'val')
            train_img_ids = glob(os.path.join(train_img_dir, '*' + config['img_ext']))
            val_img_ids = glob(os.path.join(val_img_dir, '*' + config['img_ext']))
            train_img_ids = [os.path.splitext(os.path.basename(p))[0] for p in train_img_ids]
            val_img_ids = [os.path.splitext(os.path.basename(p))[0] for p in val_img_ids]
        else:
            # Random split
            all_img_dir = os.path.join('datasets', config['dataset'], 'images')
            all_mask_dir = os.path.join('datasets', config['dataset'], 'masks')
            train_img_dir, val_img_dir = all_img_dir, all_img_dir
            train_mask_dir, val_mask_dir = all_mask_dir, all_mask_dir
            train_val_img_ids = glob(os.path.join(all_img_dir, '*' + config['img_ext']))
            train_val_img_ids = [os.path.splitext(os.path.basename(p))[0] for p in train_val_img_ids]
            train_img_ids, val_img_ids = train_test_split(train_val_img_ids, test_size=1-train_ratio,
                                                          random_state=41)

    if is_testing_enabled:
        test_img_dir = os.path.join(config["test_dataset_path"], 'images')
        test_mask_dir = os.path.join(config["test_dataset_path"], 'masks')
        test_img_ids = glob(os.path.join(test_img_dir, '*' + config['img_ext']))
        test_img_ids = [os.path.splitext(os.path.basename(p))[0] for p in test_img_ids]

    train_transform_list = [
            geometric.Flip(p=0.2),
            geometric.rotate.Rotate(10, p=0.2),
            transforms.GaussNoise(p=0.2),
            transforms.RandomBrightnessContrast(p=0.1),
            albumentations.Normalize()
        ]
    val_transform_list = [
            albumentations.Normalize()
        ]

    loader_input_w, loader_input_h = None, None
    if config["input_size_w"] is not None and config["input_size_h"] is not None:
        if config["resize_method"] == "Resize": # opencv resizing
            loader_input_w, loader_input_h = config["input_size_w"], config["input_size_h"]
            assert not config["val_partition"]
        elif config["resize_method"] == "RandomResizedCrop":
            resize_aug =  albumentations.augmentations.RandomResizedCrop(config["input_size_h"],
                                                                         config["input_size_w"])
            train_transform_list.insert(0, resize_aug)
            if not config["val_partition"]:
                val_transform_list.insert(0, resize_aug)
        elif config["resize_method"] == "RandomCrop":
            resize_aug =  albumentations.augmentations.RandomCrop(config["input_size_h"],
                                                                  config["input_size_w"])
            train_transform_list.insert(0, resize_aug)
            if not config["val_partition"]:
                val_transform_list.insert(0, resize_aug)
        elif config["resize_method"] == "Padding":
            # todo give warning when padding is not needed
            assert not config["val_partition"]
            pad_transform = albumentations.augmentations.PadIfNeeded(config["input_size_h"],
                                                                     config["input_size_w"],
                                                                     border_mode=cv2.BORDER_CONSTANT,
                                                                     value=0)
            train_transform_list.insert(0, pad_transform)
            val_transform_list.insert(0, pad_transform)
        else:
            assert False

    train_transform = Compose(train_transform_list)
    val_transform = Compose(val_transform_list)

    train_dataset = Dataset(
        img_ids=train_img_ids,
        img_dir=train_img_dir,
        mask_dir=train_mask_dir,
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        input_w=loader_input_w,
        input_h=loader_input_h,
        transform=train_transform)
    val_dataset = Dataset(
        img_ids=val_img_ids,
        img_dir=val_img_dir,
        mask_dir=val_mask_dir,
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        input_w=loader_input_w,
        input_h=loader_input_h,
        transform=val_transform)
    print("###################### Data ######################")
    print(f"Train Data: {len(train_dataset)}")
    print(f"Val Data: {len(val_dataset)}")

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False)

    if is_testing_enabled:
        test_dataset = Dataset(
            img_ids=test_img_ids,
            img_dir=test_img_dir,
            mask_dir=test_mask_dir,
            img_ext=config['img_ext'],
            mask_ext=config['mask_ext'],
            num_classes=config['num_classes'],
            input_w=loader_input_w,
            input_h=loader_input_h,
            transform=val_transform)
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=config['num_workers'],
            drop_last=False)
        print(f"Test Data: {len(test_dataset)}")
    else:
        test_loader = None
    print("--------------------------------------------------")

    return train_loader, val_loader, test_loader
