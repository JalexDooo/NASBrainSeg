import torch as t
import warnings


class Config(object):
    # global
    train_path = '/sunjindong/dataset/MICCAI_BraTS2020_TrainingData'
    # val_path = '/sunjindong/dataset/MICCAI_BraTS2020_ValidationData/MICCAI_BraTS2020_ValidationData'
    val_path = '/sunjindong/dataset/MICCAI_BraTS2020_ValidationData'
    task = 'search' # search or train or val
    gpu_ids = [0,1]
    isTrain = True
    model = 'SearchNet'
    continue_train = False
    lr = 0.0001
    lr_policy = 'power'
    lr_decay_iters = 30
    load_iter = 0
    epoch = 'latest'
    verbose = False
    train_epoch_count = 0
    des = 'high'

    # search
    search_batch = 4
    search_epoch = 200
    search_checkpoint = './ckpt/search/'

    # decode search
    decode_path = './ckpt/search/SearchNet/iter_193_net_Search.pth'

    # train
    train_batch = 16
    train_epoch = 200
    train_epoch_decay = 300
    train_epoch_count = 0
    train_checkpoint = './ckpt/train/'

    def _print(self):
        print('---------------user config:------------------')
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('_'):
                print(k, getattr(self, k))
        print('---------------------------------------------')

    def _parse(self, kwargs):
        """
        update config
        """
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("Warning: opt has not attribute %s" % k)
            setattr(self, k, v)


cfg = Config()
