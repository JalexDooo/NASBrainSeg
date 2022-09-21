import collections
import glob
import sys
import torch
import random
import numpy as np
import nibabel as nib
import torch.utils.data as data
from scipy.ndimage import rotate

from abc import abstractmethod


class BaseDataset(data.Dataset):
    def __init__(self, cfg):
        self.cfg = cfg

    @abstractmethod
    def __len__(self):
        return 0

    @abstractmethod
    def __getitem__(self, idx):
        pass

def load_nii_to_array(path):
    image = nib.load(path)
    affine = image.affine
    image = image.get_data()
    return image, affine

def make_image_label(path):
    pathes = glob.glob(path + '/*.nii.gz')
    image = []
    seg = np.zeros((240, 240, 155))
    for p in pathes:
        if 'flair.nii' in p:
            flair, aff = load_nii_to_array(p)
        elif 't2.nii' in p:
            t2, aff = load_nii_to_array(p)
        elif 't1.nii' in p:
            t1, aff = load_nii_to_array(p)
        elif 't1ce.nii' in p:
            t1ce, aff = load_nii_to_array(p)
        else:
            seg, aff = load_nii_to_array(p)
    image.append(flair)
    image.append(t1)
    image.append(t1ce)
    image.append(t2)
    label = seg
    return np.asarray(image), np.asarray(label), aff

def normalization(image):
    img = image[image > 0]
    image = (image - img.mean()) / img.std()
    return image

def label_processing(label):
    lbl = (label==1)*1.0 + (label==2)*2.0 + (label==4)*3.0
    return np.array(lbl)

# ---------------------torchvision-transforms-----------------------
class Base(object):
    def sample(self, *shape):
        return shape
    
    def transform(self, img, k=0):
        return img
    
    def __call__(self, img, dim=3, reuse=False):
        if not reuse:
            im = img if isinstance(img, np.ndarray) else img[0]
            shape = im.shape[1:dim+1]
            self.sample(*shape)
            
        if isinstance(img, (collections.Sequence if (sys.version_info < (3,8)) else collections.abc.Sequence)):
            return [self.transform(x, k) for k, x in enumerate(img)]
        
        return self.transform(img)
    
    def __str__(self):
        return 'Identity()'

class CenterCrop(Base):
    def __init__(self, size):
        self.size = size
        self.buffer = None
    
    def sample(self, *shape):
        size = self.size
        if not isinstance(self.size,list):
            size = list(self.size)
        else:
            size = self.size
        # print('size, shape, ', size, shape)
        start = [(s-i)//2 for i, s in zip(size, shape)]
        self.buffer = [slice(None)] + [slice(s, s+k) for s,k in zip(start,size)]
        return size
    
    def transform(self, img, k=0):
        return img[tuple(self.buffer)]
    def __str__(self):
        return 'CenterCrop({})'.format(self.size)
    
class RandCrop(CenterCrop):
    def sample(self, *shape):
        size = self.size
        start = [random.randint(0, s-size) for s in shape]
        self.buffer = [slice(None)] + [slice(s, s+size) for s in start]
        return [size]*len(shape)

    def __str__(self):
        return 'RandCrop({})'.format(self.size)

class RandCrop3D(CenterCrop):
    def sample(self, *shape): # shape : [240,240,155]
        assert len(self.size)==3 # random crop [H,W,T] from img [240,240,155]
        if not isinstance(self.size,list):
            size = list(self.size)
        else:
            size = self.size
        start = [random.randint(0, s-i) for i,s in zip(size,shape)]
        self.buffer = [slice(None)] + [slice(s, s+k) for s,k in zip(start,size)]
        return size

    def __str__(self):
        return 'RandCrop({})'.format(self.size)

class RandomRotion(Base):
    def __init__(self,angle_spectrum=10):
        assert isinstance(angle_spectrum,int)
        # axes = [(2, 1), (3, 1),(3, 2)]
        axes = [(1, 0), (2, 1),(2, 0)]
        self.angle_spectrum = angle_spectrum
        self.axes = axes

    def sample(self,*shape):
        self.axes_buffer = self.axes[np.random.choice(list(range(len(self.axes))))] # choose the random direction
        self.angle_buffer = np.random.randint(-self.angle_spectrum, self.angle_spectrum) # choose the random direction
        return list(shape)

    def transform(self, img, k=0):
        """ Introduction: The rotation function supports the shape [H,W,D,C] or shape [H,W,D]
        :param img: if x, shape is [1,H,W,D,c]; if label, shape is [1,H,W,D]
        :param k: if x, k=0; if label, k=1
        """
        bsize = img.shape[0]

        for bs in range(bsize):
            if k == 0:
                # [[H,W,D], ...]
                # print(img.shape) # (1, 128, 128, 128, 4)
                channels = [rotate(img[bs,:,:,:,c], self.angle_buffer, axes=self.axes_buffer, reshape=False, order=0, mode='constant', cval=-1) for c in
                            range(img.shape[4])]
                img[bs,...] = np.stack(channels, axis=-1)

            if k == 1:
                img[bs,...] = rotate(img[bs,...], self.angle_buffer, axes=self.axes_buffer, reshape=False, order=0, mode='constant', cval=-1)

        return img

    def __str__(self):
        return 'RandomRotion(axes={},Angle:{}'.format(self.axes_buffer,self.angle_buffer)

class RandomIntensityChange(Base):
    def __init__(self,factor):
        shift,scale = factor
        assert (shift >0) and (scale >0)
        self.shift = shift
        self.scale = scale

    def transform(self,img,k=0):
        if k==1:
            return img

        shift_factor = np.random.uniform(-self.shift,self.shift,size=[1,img.shape[1],1,1,img.shape[4]]) # [-0.1,+0.1]
        scale_factor = np.random.uniform(1.0 - self.scale, 1.0 + self.scale,size=[1,img.shape[1],1,1,img.shape[4]]) # [0.9,1.1)
        # shift_factor = np.random.uniform(-self.shift,self.shift,size=[1,1,1,img.shape[3],img.shape[4]]) # [-0.1,+0.1]
        # scale_factor = np.random.uniform(1.0 - self.scale, 1.0 + self.scale,size=[1,1,1,img.shape[3],img.shape[4]]) # [0.9,1.1)
        return img * scale_factor + shift_factor

    def __str__(self):
        return 'random intensity shift per channels on the input image, including'

class RandomFlip(Base):
    # mirror flip across all x,y,z
    def __init__(self,axis=0):
        # assert axis == (1,2,3) # For both data and label, it has to specify the axis.
        self.axis = (1,2,3)
        self.x_buffer = None
        self.y_buffer = None
        self.z_buffer = None

    def sample(self, *shape):
        self.x_buffer = np.random.choice([True,False])
        self.y_buffer = np.random.choice([True,False])
        self.z_buffer = np.random.choice([True,False])
        return list(shape) # the shape is not changed

    def transform(self,img,k=0): # img shape is (1, 240, 240, 155, 4)
        if self.x_buffer:
            img = np.flip(img,axis=self.axis[0])
        if self.y_buffer:
            img = np.flip(img,axis=self.axis[1])
        if self.z_buffer:
            img = np.flip(img,axis=self.axis[2])
        return img

class NumpyType(Base):
    def __init__(self, types, num=-1):
        self.types = types # ('float32', 'int64')
        self.num = num

    def transform(self, img, k=0):
        if self.num > 0 and k >= self.num:
            return img
        # make this work with both Tensor and Numpy
        return img.astype(self.types[k])

    def __str__(self):
        s = ', '.join([str(s) for s in self.types])
        return 'NumpyType(({}))'.format(s)

class Pad(Base):
    def __init__(self, pad): # [0, 0, 0, 5, 0]
        self.pad = pad
        self.px = tuple(zip([0]*len(pad), pad))
    
    def sample(self, *shape):
        shape = list(shape)
        for i in range(len(shape)):
            shape[i] += self.pad[i+1]
        return shape
    
    def transform(self, img, k=0):
        dim = len(img.shape)
        return np.pad(img, self.px[:dim], mode='constant')
    
    def __str__(self):
        return 'Pad(({}, {}, {}))'.format(*self.pad)

