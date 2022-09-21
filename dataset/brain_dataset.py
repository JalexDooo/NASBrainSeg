from .base_dataset import Base, BaseDataset, CenterCrop, RandCrop3D, RandomRotion, RandomIntensityChange, RandomFlip, NumpyType, Pad
from .base_dataset import make_image_label, normalization, label_processing

import torch
import glob
import numpy as np
import nibabel as nib
from torchvision.transforms import Compose

from dataset.base_dataset import RandomRotion

class BraTS20(BaseDataset):
    def __init__(self, cfg):
        BaseDataset.__init__(self, cfg)
        if self.cfg.isTrain:
            self.paths = glob.glob(self.cfg.train_path+'/*')
            self.transforms = Compose([
                # search
                # RandCrop3D((128, 128, 48)),
                # RandomRotion(10),
                # RandomIntensityChange((0.1, 0.1)),
                # RandomFlip(0),
                # NumpyType((np.float32, np.float64)),

                # train
                RandCrop3D((192, 192, 48)), # 64, 64, 64
                RandomRotion(10),
                RandomIntensityChange((0.1, 0.1)),
                RandomFlip(0),
                NumpyType((np.float32, np.float64)),
            ])
        else:
            self.paths = glob.glob(self.cfg.val_path+'/*')
            self.transforms = Compose([
                Pad((0, 0, 0, 5, 0)),
                # CenterCrop((192, 192, 48)),
                NumpyType((np.float32, np.int64)),
            ])
    
    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        name = path.split('/')[-1]
        images, labels, affine = make_image_label(path)

        flair, t1, t1ce, t2 = images
        flair = normalization(flair)
        t1 = normalization(t1)
        t1ce = normalization(t1ce)
        t2 = normalization(t2)

        images = []
        images.append(flair)
        images.append(t1)
        images.append(t1ce)
        images.append(t2)
        images = np.asarray(images)
        # print('norm, max, min', images[0, ...].max(), images[0, ...].min(), images.shape)
        images, labels = images[..., np.newaxis], labels[np.newaxis, ...]
        images = np.swapaxes(images, 0, -1)
        
        images, labels = self.transforms([images, labels])
        labels = label_processing(labels)

        images = np.ascontiguousarray(images.transpose(0, 4, 1, 2, 3)[0])
        labels = np.ascontiguousarray(labels[0])

        images = torch.from_numpy(images)
        labels = torch.from_numpy(labels)

        # print('torch, maxmin, ', images[0].max(), images[0].min(), images.shape)
        
        return {'img': images, 'label': labels, 'name': name, 'affine': affine}
    
    def collate(self, batch):
        return [torch.cat(v) for v in zip(*batch)]



