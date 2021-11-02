_base_ = '../_base_/default_runtime.py'

model = dict(
    type='TMP',
    pretrained='/root/data/zq/pretrained_models/resnet50-19c8e357.pth',
    backbone=dict(
        type='ResNet',
        depth=50,
        out_indices=[3],
        strides=(1, 2, 2, 1),
        norm_cfg=dict(type='BN'),
        norm_eval=False),
    neck=dict(
        type='BNNeck',
        feat_dim=2048,
        norm_cfg=dict(type='BN1d'),
        with_bias=False,
        with_avg_pool=True,
        avgpool=dict(type='AvgPoolNeck')),
    head=dict(type='CamAwareSCLHead', temperature=0.05))

data_source = dict(type='Market1501', data_root='/root/data/zq/data/market1501/Market-1501-v15.09.15')
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
        p=0.5),
    dict(type='ToTensor'),
    dict(
        type='Normalize',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    dict(type='RandomErasing', value=[0.485, 0.456, 0.406])
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
        type='FixedStepIdentitySampler',
        num_instances=4,
        step=50,
        with_camid=True),
    train=dict(
        type=dataset_type, data_source=data_source, pipeline=train_pipeline),
    test=dict(
        type='ReIDDataset',
        data_source=data_source,
        pipeline=test_pipeline,
        test_mode=True))

custom_hooks = [
    dict(
        type='LabelGenerationHook',
        extractor=dict(
            dataset=dict(
                type='ReIDDataset',
                data_source=data_source,
                pipeline=test_pipeline),
            samples_per_gpu=32,
            workers_per_gpu=4),
        label_generator=dict(
            type='SelfPacedGenerator', eps=[0.75], min_samples=4, k1=30, k2=6),
        interval=4)
]
# optimizer
# paramwise_cfg = {
    # r'(bn|gn)(\d+)?.(weight|bias)': dict(weight_decay=0., lars_exclude=True),
    # r'bias': dict(weight_decay=0., lars_exclude=True)
# }
# optimizer = dict(
    # type='LARS',
    # lr=0.3,
    # weight_decay=0.0000015,
    # momentum=0.9,
    # paramwise_cfg=paramwise_cfg)
optimizer = dict(type='SGD', lr=0.1, weight_decay=5e-4, momentum=0.9)
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=0.,
    warmup='linear',
    warmup_iters=5,
    warmup_ratio=0.0001,  # cannot be 0
    warmup_by_epoch=True)
total_epochs = 100
