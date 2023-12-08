# Feature Block Topology Search Based on Differentiable Neural Architecture Search for Multimodal MRI Brain Tumor Segmentation

This repository is the work of "Feature Block Topology Search Based on Differentiable Neural Architecture Search for Multimodal MRI Brain Tumor Segmentation" based on **pytorch** implementation. 


## Requirements
- python 3.6
- pytorch 1.8.1 or later CUDA version
- torchvision
- nibabel
- SimpleITK
- matplotlib
- fire
- Pillow

### Searching
Multiply gpus Searching is recommended.
```
cd {path} && python -u main.py search --{param}={} # {param} in config.py
```
Searching like this
```
cd /sunjindong/NASBrainSeg && python -u main.py search --gpu_ids=[0,1,2,3] --task='search'
```

### Training

Multiply gpus training is recommended. 
```
cd {path} && python -u main.py train --{param}={}
```

Training like this: (Forbidden redundant space)
```
cd /sunjindong/NASBraSeg && python -u main.py train --gpu_ids=[0,1,2,3,4,5,6,7] --task='train' --des='high' --train_path='/sunjindong/dataset/MICCAI_BraTS2020_TrainingData'
```

### Validation
You could obtain the resutls as paper reported by running the following code:

```
cd {path} && python -u main.py val --{param}={}
```
Like this:
```
cd /sunjindong/NASBraSeg && python -u main.py val --gpu_ids=[0] --task='val' --load_iter=500  --des='high'
```
Then make a submission to the online evaluation server.

### Calculation Flops, Prams and Prediction time

```
python3 main.py model_flops_params_caltime --model='liu2023_adhdc'
```

## Citation

If you use our code or model in your work or find it is helpful, please cite the paper:
```
***Unknown***
```

## Acknowledge
None.

