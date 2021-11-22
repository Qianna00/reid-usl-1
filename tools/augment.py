import argparse
import os
import os.path as osp
import time
import copy

import mmcv
import torch
from mmcv.runner import get_dist_info, init_dist
from mmcv.utils import Config, DictAction, collect_env

from reid import __version__
from reid.apis import set_random_seed, train_model
from reid.datasets import build_dataset
from reid.models import build_reid
from reid.utils import get_root_logger

from reid.datasets.builder import DATASETS
from reid.datasets.contrastive import ContrastiveDataset

data_source = dict(type='Market1501', data_root='/root/data/zq/data/market1501/Market-1501-v15.09.15', cam_aware=True)
dataset_type = 'ContrastiveDataset'
train_pipeline = [
    dict(
        type='RandomCamStyle',
        camstyle_root='bounding_box_train_camstyle',
        p=0.5),
    dict(type='LoadImage'),
    dict(
        type='RandomResizedCrop',
        size=(256, 128),
        scale=(0.64, 1.0),
        ratio=(0.33, 0.5),
        interpolation=3),
    dict(type='RandomHorizontalFlip'),
    dict(type='RandomRotation', degrees=10),
    dict(
        type='RandomApply',
        transforms=[dict(type='GaussianBlur', sigma=(0.1, 2.0))],
        p=0.5)
]
test_pipeline = [
    dict(type='LoadImage'),
    dict(type='Resize', size=(256, 128), interpolation=3),
    dict(type='ToTensor'),
    dict(
        type='Normalize',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])
]
data = dict(
    samples_per_gpu=32,
    workers_per_gpu=4,
    sampler=dict(
        type='CamAwareFixedStepIdentitySampler',
        num_instances=4,
        step=50,
        with_camid=True),
    train=dict(
        type=dataset_type, data_source=data_source, pipeline=train_pipeline),
    test=dict(
        type='ReIDDataset',
        data_source=dict(type='Market1501',
                         data_root='/root/data/zq/data/market1501/Market-1501-v15.09.15'),
        pipeline=test_pipeline,
        test_mode=True))

@DATASETS.register_module()
class AugmentDataset(ContrastiveDataset):
    def __getitem__(self, idx):
        img, pid, camid = self.get_sample(idx)
        label = self.pid_dict[pid] if not self.test_mode else pid
        results = dict(img=img, label=label, pid=pid, camid=camid, idx=idx)

        img1 = self.pipeline(copy.deepcopy(results))['img']
        img2 = self.pipeline(copy.deepcopy(results))['img']

        return img1, img2

dataset = build_dataset(data["train"])

for i in range(len(dataset)):
    img1, img2 = dataset.__getitem__(i)

    print(type(img1))

