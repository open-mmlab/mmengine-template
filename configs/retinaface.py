cfg_re50 = {
    'name': 'Resnet50',
    'min_sizes': [[16, 32], [64, 128], [256, 512]],
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'gpu_train': True,
    'batch_size': 24,
    'ngpu': 4,
    'epoch': 100,
    'decay1': 70,
    'decay2': 90,
    'image_size': 840,
    'pretrain': True,
    'return_layers': {
        'layer2': 1,
        'layer3': 2,
        'layer4': 3
    },
    'in_channel': 256,
    'out_channel': 256,
    'score_thresh': 0.02,
    'nms_thresh': 0.4,
    'num_classes': 2
}

model = dict(
    type='RetinaFace',
    data_preprocessor=dict(
        type='ImgDataPreprocessor',
        # bgr order
        mean=[0, 0, 0],
        std=[1, 1, 1]),
    cfg=cfg_re50)

default_scope = 'mmengine_template'

train_dataloader = dict(
    dataset=dict(
        type='WiderFaceDataset',
        annotations='./data/widerface/train/label.txt',
        pipeline=dict(
            type='RetinaFacePipeline',
            img_dim=cfg_re50['image_size'],
            rgb_means=(104, 117, 123),
        ),
        training=True),
    batch_size=cfg_re50['batch_size'] // cfg_re50['ngpu'],
    num_workers=2,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=True),
)

test_dataloader = dict(
    dataset=dict(
        type='WiderFaceDataset',
        annotations='./data/widerface/val/label.txt',
        pipeline=dict(
            type='RetinaFacePipeline',
            img_dim=cfg_re50['image_size'],
            rgb_means=(104, 117, 123),
        ),
        training=False),
    batch_size=24,
    num_workers=2,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
)
val_dataloader = test_dataloader

# Why env_cfg is necessary
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

test_cfg = dict(type='TestLoop')
val_cfg = dict(type='TestLoop')

# NOTE will not throw error if val_interval is set, and val_dataloader and
# val_cfg are not set
train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=cfg_re50['epoch'], val_interval=30)

param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=100,
        by_epoch=True,
        milestones=[cfg_re50['decay1'], cfg_re50['decay2']],
        gamma=0.1)
]

optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=1e-3, momentum=0.9, weight_decay=5e-4))
test_evaluator = dict(
    type='WiderFaceMetric',
    annotations='./data/widerface/eval_tools/ground_truth/',
    saved_path='./widerface_txt',
    dist_backend='torch_cuda')

val_evaluator = dict(
    type='WiderFaceMetric',
    annotations='./data/widerface/eval_tools/ground_truth/',
    saved_path='./widerface_txt',
    dist_backend='torch_cuda')
