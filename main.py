import enum
import os
import torch
import numpy as np
import config
from dataset.brain_dataset import BraTS20
from torch.utils.data import DataLoader
import models
import nibabel as nib


def global_init():
    print('working with pytorch version {}'.format(torch.__version__))
    print('with cuda version {}'.format(torch.version.cuda))
    print('cudnn enabled: {}'.format(torch.backends.cudnn.enabled))
    print('cudnn version: {}'.format(torch.backends.cudnn.version()))
    if torch.cuda.is_available():
        torch.backends.cudnn.benckmark = True
    else:
        raise Exception('CUDA is not available!!!!')

def search(**kwargs):
    cfg = config.cfg
    cfg._parse(kwargs)
    cfg.model = 'SearchNet'
    cfg.search_batch = int(len(cfg.gpu_ids)*2)

    # dataset debug
    train_dataset = BraTS20(cfg)
    train_dataloader = DataLoader(train_dataset, batch_size=cfg.search_batch, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)

    model = getattr(models, cfg.model)(cfg)
    model.setup(cfg)
    for epoch in range(cfg.search_epoch+cfg.train_epoch_decay+1):
        for i, data in enumerate(train_dataloader):
            model.set_input(data)
            model.optimize_parameters()
            if i % 5 == 0:
                print('Epoch: {}, i: {}, loss: {}'.format(epoch, i, model.get_current_losses()))
        print('saving model....')
        model.save_networks('iter_{}'.format(epoch))

        model.update_learning_rate()
        if model.cfg.lr <= 1e-6:
            print('lr is too small...finish')
            break
    model.save_networks('latest')

def val_search(**kwargs):
    # global_init()
    cfg = config.cfg
    cfg._parse(kwargs)
    cfg.model = 'SearchNet'
    cfg.isTrain = False
    cfg._print()

    output_path = './output/{}-{}'.format(cfg.model, cfg.load_iter)
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    # dataset debug
    val_dataset = BraTS20(cfg)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True, drop_last=False)

    model = getattr(models, cfg.model)(cfg)
    model.eval()
    model.setup(cfg)

    print(model.betas())

    criterion = torch.nn.CrossEntropyLoss()

    for i, data in enumerate(val_dataloader):
        d = data
        name = d['name']
        affine = d['affine']
        model.set_input(d)
        model.test()
        visuals = model.get_current_visuals()

        img = visuals['img'].data.cpu().detach().numpy()[0][0]
        label = visuals['label'].data.cpu().detach().numpy()[0]

        pred = visuals['pred'].data.cpu().detach()[0]
        print(pred.shape, label.shape)
        loss = criterion(torch.Tensor(pred[np.newaxis, ...]), torch.Tensor(label[np.newaxis, ...]).long())
        print('loss: ', float(loss))
        _, pred = torch.max(pred, dim=0)
        
        # pred = pred.argmax(0)
        pred = pred.int().numpy()

        print((pred==0).sum(), (pred==1).sum(), (pred==2).sum(), (pred==3).sum())

        print(pred.shape, img.shape, label.shape)

        print('affine ', affine.shape)
        print('name  ', name)

        output = nib.Nifti1Image(pred, affine[0])
        nib.save(output, output_path+'/{}.nii.gz'.format(name[0]))

        output = nib.Nifti1Image(img, affine[0])
        nib.save(output, output_path+'/img_{}.nii.gz'.format(name[0]))

        # output = nib.Nifti1Image(label, affine[0])
        # nib.save(output, output_path+'/label_{}.nii.gz'.format(name[0]))

        if i >= 3:
            break

"""
tensor([[0.9987, 1.0813, 1.0000],
        [1.0087, 1.0230, 0.9949],
        [1.0082, 0.9938, 1.0072],
        [1.0120, 0.9885, 0.9962],
        [1.0030, 1.0056, 1.0037],
        [0.9941, 1.0096, 0.9995],
        [0.9520, 0.9291, 0.9757],
        [0.6069, 0.9195, 1.0010]], requires_grad=True)
"""

def compute_architecture():
    a = np.array([[0.9987, 1.0813, 1.0000], # net.beta()
        [1.0087, 1.0230, 0.9949],
        [1.0082, 0.9938, 1.0072],
        [1.0120, 0.9885, 0.9962],
        [1.0030, 1.0056, 1.0037],
        [0.9941, 1.0096, 0.9995],
        [0.9520, 0.9291, 0.9757],
        [0.6069, 0.9195, 1.0010]])
    
    def judge(num):
        tmp = num//3
        return num - tmp*3
    
    def confict(p):
        if p[0] == 2:
            return False
        for kk in range(1, 8):
            if p[kk]-p[kk-1]==2 or p[kk]-p[kk-1]==-2:
                return False
        return True

    def generate(num):
        tmp = num
        p = []
        for kk in range(8):
            c = judge(tmp)
            p.append(c)
            tmp //= 3
        return p

    # 3^8 paths
    paths = 3**8
    path = []
    value = []
    
    for i in range(paths):
        rec_path = generate(i)
        if confict(rec_path):
            path.append(rec_path)

    print('possible count: ', len(path)) # all possible paths
    for i in path:
        summ = 1.0
        for j, k in enumerate(i):
            summ *= a[j][k]
        value.append(summ)

    # all possible paths weights
    value = np.array(value)
    low, lowmid, mid, highmid, high = np.percentile(value, (0, 25, 50, 75, 100), interpolation='nearest')
    low_index = np.where(value==low)[0][0]
    lowmid_index = np.where(value==lowmid)[0][0]
    mid_index = np.where(value==mid)[0][0]
    highmid_index = np.where(value==highmid)[0][0]
    high_index = np.where(value==high)[0][0]
    print('value: ', low, lowmid, mid, highmid, high)

    low_path = path[low_index]
    lowmid_path = path[lowmid_index]
    mid_path = path[mid_index]
    highmid_path = path[highmid_index]
    high_path = path[high_index]

    return low_path, lowmid_path, mid_path, highmid_path, high_path

def get_path_dir(path):
    dir = []
    for i, _ in enumerate(path):
        if i == 0:
            dir.append( -(path[i]-0) )
        else:
            dir.append( -(path[i]-path[i-1]) )
    print(dir)
    return dir
            

def get_architecture():
    a = np.array([[0.9987, 1.0813, 1.0000], # net.beta()
        [1.0087, 1.0230, 0.9949],
        [1.0082, 0.9938, 1.0072],
        [1.0120, 0.9885, 0.9962],
        [1.0030, 1.0056, 1.0037],
        [0.9941, 1.0096, 0.9995],
        [0.9520, 0.9291, 0.9757],
        [0.6069, 0.9195, 1.0010]])

    path = [1, 1, 0, 0, 1, 1, 2, 2]
    path_dir = [-1, 0, 1, 0, -1, 0, -1, 0] # -1:down, 0:same, 1:up
    return path, path_dir

def got():
    high = compute_architecture()[-1]
    highmid = compute_architecture()[-2]
    mid = compute_architecture()[-3]
    lowmid = compute_architecture()[-4]
    low = compute_architecture()[-5]
    print('high: {},\n highmid: {},\n mid: {},\n lowmid: {},\n low: {}'.format(high, highmid, mid, lowmid, low))
    """
    high: [1, 1, 0, 0, 1, 1, 2, 2],
    highmid: [1, 1, 1, 0, 0, 1, 0, 1],
    mid: [1, 2, 2, 1, 1, 1, 1, 1],
    lowmid: [1, 1, 0, 1, 0, 0, 0, 0],
    low: [0, 0, 1, 1, 0, 0, 1, 0]
    """

def train(**kwargs):
    global_init()
    cfg = config.cfg
    cfg._parse(kwargs)
    
    cfg.isTrain = True
    cfg.train_batch = int(len(cfg.gpu_ids)*2)

    # dataset
    train_dataset = BraTS20(cfg)
    train_dataloader = DataLoader(train_dataset, batch_size=cfg.train_batch, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)

    cfg.model = 'SegNet'
    # cfg.net_arch, cfg.net_arch_dir = get_architecture()
    if cfg.des == 'high':
        cfg.net_arch = compute_architecture()[-1]
    elif cfg.des == 'highmid':
        cfg.net_arch = compute_architecture()[-2]
    elif cfg.des == 'mid':
        cfg.net_arch = compute_architecture()[-3]
    elif cfg.des == 'lowmid':
        cfg.net_arch = compute_architecture()[-4]
    elif cfg.des == 'low':
        cfg.net_arch = compute_architecture()[-5]
    elif cfg.des == 'ushape':
        cfg.net_arch = [1, 2, 2, 2, 2, 2, 1, 0]
    
    cfg.net_arch_dir = get_path_dir(cfg.net_arch)


    model = getattr(models, cfg.model)(cfg)
    model.setup(cfg)
    cfg._print()

    for epoch in range(cfg.train_epoch+cfg.train_epoch_decay+1):
        if epoch:
            model.update_learning_rate()
        for i, data in enumerate(train_dataloader):
            model.set_input(data)
            model.optimize_parameters()
            if i % 5 == 0:
                print('Epoch: {}, i: {}, loss: {}'.format(epoch, i, model.get_current_losses()))
        print('saving model....')
        model.save_networks('iter_{}'.format(epoch))
        # lr = model.optimizers[0].param_groups[0]['lr']
        # if lr <= 1e-6:
        #     break
    model.save_networks('latest')

def val(**kwargs):
    # global_init()
    cfg = config.cfg
    cfg._parse(kwargs)
    cfg.model = 'SegNet'
    cfg.isTrain = False
    cfg._print()

    

    # dataset debug
    val_dataset = BraTS20(cfg)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True, drop_last=False)

    # cfg.net_arch, cfg.net_arch_dir = get_architecture()
    if cfg.des == 'high':
        cfg.net_arch = compute_architecture()[-1]
    elif cfg.des == 'highmid':
        cfg.net_arch = compute_architecture()[-2]
    elif cfg.des == 'mid':
        cfg.net_arch = compute_architecture()[-3]
    elif cfg.des == 'lowmid':
        cfg.net_arch = compute_architecture()[-4]
    elif cfg.des == 'low':
        cfg.net_arch = compute_architecture()[-5]
    elif cfg.des == 'ushape':
        cfg.net_arch = [1, 2, 2, 2, 2, 2, 1, 0]

    output_path = './output/{}-{}-{}'.format(cfg.model, cfg.load_iter, cfg.des)
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    
    cfg.net_arch_dir = get_path_dir(cfg.net_arch)


    model = getattr(models, cfg.model)(cfg)
    model.setup(cfg)
    model.eval()

    for i, data in enumerate(val_dataloader):
        d = data
        name = d['name']
        affine = d['affine']
        model.set_input(d)
        model.test()
        visuals = model.get_current_visuals()

        img = visuals['img'].data.cpu().detach().numpy()[0][0]
        label = visuals['label'].data.cpu().detach().numpy()[0]

        pred = visuals['pred'].data.cpu().detach()[0]
        _, pred = torch.max(pred, dim=0)
        
        # pred = pred.argmax(0)
        pred = pred.int().numpy()

        pred[pred==3] = 4

        # outprocessing
        # if (pred==1).sum()<=100:
        #     pred[pred==1] = 1
        if (pred==4).sum()<=400:
            pred[pred==4] = 1
        # if (pred==2).sum()<=100:
        #     pred[pred==2] = 0

        print((pred==0).sum(), (pred==1).sum(), (pred==2).sum(), (pred==4).sum())

        pred = pred[:, :, :155]

        print(pred.shape, img.shape, label.shape)

        print('affine ', affine.shape)
        print('name  ', name)

        output = nib.Nifti1Image(pred, affine[0])
        nib.save(output, output_path+'/{}.nii.gz'.format(name[0]))

        # output = nib.Nifti1Image(img, affine[0])
        # nib.save(output, output_path+'/img_{}.nii.gz'.format(name[0]))

        # output = nib.Nifti1Image(label, affine[0])
        # nib.save(output, output_path+'/label_{}.nii.gz'.format(name[0]))

        # if i >= 3:
        #     break

def debug(**kwargs):
    cfg = config.cfg
    cfg._parse(kwargs)
    cfg.model = 'SegNet'
    cfg._print()

    # dataset
    train_dataset = BraTS20(cfg)
    train_dataloader = DataLoader(train_dataset, batch_size=cfg.train_batch, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)

    model = getattr(models, 'SegNet')(cfg)
    model.setup(cfg)

    for epoch in range(cfg.search_epoch+cfg.train_epoch_decay+1):
        model.update_learning_rate()
        for i, data in enumerate(train_dataloader):
            model.set_input(data)
            assert False
            model.optimize_parameters()
            if i % 5 == 0:
                print('Epoch: {}, i: {}, loss: {}'.format(epoch, i, model.get_current_losses()))
        print('saving model....')
        model.save_networks('iter_{}'.format(epoch))
        lr = model.optimizers[0].param_groups[0]['lr']
        if lr <= 1e-6:
            break
    model.save_networks('latest')

def tttt():
    for epoch in range(200):
        # lrd = np.power(1 - epoch/(100+100), 0.999)
        lrd = np.power(0.97, epoch)
        lr = 0.01 * lrd
        print(epoch, lr)
        # wget http://www.clamav.net/downloads/production/clamav-0.105.0.tar.gz

# python3 main.py model_flops_params_caltime --model='liu2023_adhdc'
def model_flops_params_caltime(**kwargs): # 192, 192, 48
    import time
    cfg = config.cfg
    cfg._parse(kwargs)
    cfg.des = 'high'
    
    cfg.isTrain = True
    cfg.train_batch = int(len(cfg.gpu_ids)*2)

    cfg.net_arch, cfg.net_arch_dir = get_architecture()
    model = getattr(models, cfg.model)(net_arch=cfg.net_arch, net_arch_dir=cfg.net_arch_dir)
    # model = getattr(models, cfg.model)(cfg)
    model.eval()

    # 3D
    a = torch.Tensor(np.random.randn(1, 4, 192, 192, 48))
    # a = torch.Tensor(np.random.randn(1, 4, 128, 128, 128))

    # 3D
    # val_dataset = Multi(cfg, is_train=False)
    # val_dataloader = DataLoader(val_dataset, batch_size=cfg.search_batch, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
    # for a in val_dataloader:
    #     a = a
    #     break

    # 2D
    # a = torch.Tensor(np.random.randn(1, 4, 192, 192))
    st = time.time()
    # b = model(a)
    
    from thop import profile
    from thop import clever_format
    flops, params = profile(model, inputs=(a,))
    flops, params = clever_format([flops, params], '%.3f')
    print('flops & params: {}, {}'.format(flops, params))
    end = time.time()
    print('TIME: ', end-st) # 2D need multiply 48

if __name__ == "__main__":
    import fire
    fire.Fire()

"""
''''''''''
# search
# cd {path} && python -u main.py search --{param}={} # {param} in config.py
# train
# cd {path} && python -u main.py train --{param}={}
# val
# cd {path} && python -u main.py val --{param}={}
''''''''''

cd /sunjindong/NAS_BraSeg && python -u main.py search --gpu_ids=[0,1,2,3] --task='search'
cd /sunjindong/NAS_BraSeg && python -u main.py desearch --gpu_ids=[0] --task='search'
cd /sunjindong/NAS_BraSeg && python -u main.py train --gpu_ids=[0,1] --task='train'

cd /sunjindong/NAS_BraSeg && python -u main.py train --gpu_ids=[0,1,2,3] --task='train' --des='high' --load_iter=140 --continue_train=True
cd /sunjindong/NAS_BraSeg && python -u main.py train --gpu_ids=[0,1] --task='train' --des='low'
cd /sunjindong/NAS_BraSeg && python -u main.py train --gpu_ids=[0,1] --task='train' --des='mid'
cd /sunjindong/NAS_BraSeg && python -u main.py train --gpu_ids=[0,1] --task='train' --des='lowmid'
cd /sunjindong/NAS_BraSeg && python -u main.py train --gpu_ids=[0,1] --task='train' --des='highmid'
cd /sunjindong/NAS_BraSeg && python -u main.py train --gpu_ids=[0,1] --task='train' --des='ushape'

# BraTS20
cd /sunjindong/NAS_BraSeg && python -u main.py train --gpu_ids=[0,1,2,3,4,5,6,7] --task='train' --des='high'
cd /sunjindong/NAS_BraSeg && python -u main.py train --gpu_ids=[0,1,2,3,4,5,6,7] --task='train' --des='highmid'
cd /sunjindong/NAS_BraSeg && python -u main.py train --gpu_ids=[0,1,2,3,4,5,6,7] --task='train' --des='mid'
cd /sunjindong/NAS_BraSeg && python -u main.py train --gpu_ids=[0,1,2,3,4,5,6,7] --task='train' --des='lowmid'
cd /sunjindong/NAS_BraSeg && python -u main.py train --gpu_ids=[0,1,2,3,4,5,6,7] --task='train' --des='low'
cd /sunjindong/NAS_BraSeg && python -u main.py train --gpu_ids=[0,1,2,3,4,5,6,7] --task='train' --des='ushape'
cd /sunjindong/NAS_BraSeg && python -u main.py train --gpu_ids=[0,1,2,3,4,5,6,7] --task='train' --des='vshape'


cd /sunjindong/NAS_BraSeg && python -u main.py val --gpu_ids=[0] --task='val' --load_iter=500  --des='high'
cd /sunjindong/NAS_BraSeg && python -u main.py val --gpu_ids=[0] --task='val' --load_iter=500  --des='low'
cd /sunjindong/NAS_BraSeg && python -u main.py val --gpu_ids=[0] --task='val' --load_iter=500  --des='mid'
cd /sunjindong/NAS_BraSeg && python -u main.py val --gpu_ids=[0] --task='val' --load_iter=500  --des='highmid'
cd /sunjindong/NAS_BraSeg && python -u main.py val --gpu_ids=[0] --task='val' --load_iter=500  --des='lowmid'
cd /sunjindong/NAS_BraSeg && python -u main.py val --gpu_ids=[0] --task='val' --load_iter=500  --des='ushape'

# FeTS21
cd /sunjindong/NAS_BraSeg && python -u main.py val --gpu_ids=[0] --task='val' --load_iter=500  --des='high' --val_path='/sunjindong/dataset/MICCAI_FeTS2021_ValidationData'
cd /sunjindong/NAS_BraSeg && python -u main.py val --gpu_ids=[0] --task='val' --load_iter=500  --des='low' --val_path='/sunjindong/dataset/MICCAI_FeTS2021_ValidationData'
cd /sunjindong/NAS_BraSeg && python -u main.py val --gpu_ids=[0] --task='val' --load_iter=500  --des='mid' --val_path='/sunjindong/dataset/MICCAI_FeTS2021_ValidationData'

# BraTS19
cd /sunjindong/NAS_BraSeg && python -u main.py val --gpu_ids=[0] --task='val' --load_iter=500  --des='high' --val_path='/sunjindong/dataset/MICCAI_BraTS_2019_Data_Validation/MICCAI_BraTS_2019_Data_Validation'



python3 -u main.py val --gpu_ids='' --des='high' --task='val' --load_iter=140 --val_path='/Users/jontysun/Downloads/数据集/MICCAI_BraTS2020_ValidationData'

cd /sunjindong/NAS_BraSeg && python -u main.py debug --gpu_ids=[0]
python3 -u main.py debug --gpu_ids='' --task='val' --load_iter=6 --val_path='/Users/jontysun/Downloads/数据集/MICCAI_BraTS2020_TrainingData'


cd /sunjindong/NAS_BraSeg && python -u main.py val_search --gpu_ids='' --task='search' --load_iter=0

python3 -u main.py val_search --gpu_ids='' --task='search' --val_path='/Users/jontysun/Downloads/数据集/MICCAI_BraTS2020_ValidationData'

python3 -u main.py debug --gpu_ids='' --task='train' --train_path='/Users/jontysun/Downloads/数据集/MICCAI_BraTS2020_TrainingData'

python3 -u main.py model_test --gpu_ids='' --task='train' --train_path='/Users/jontysun/Downloads/数据集/MICCAI_BraTS2020_TrainingData'


"""