_base_ = [
    '/opt/ml/detection/level2-object-detection-level2-cv-10/mmdetection/configs/_youngwoo_/_base_/datasets/coco_detection_albu_1024_4.py',
    '/opt/ml/detection/level2-object-detection-level2-cv-10/mmdetection/configs/_youngwoo_/_base_/schedules/schedule_cosann_adamw.py',
    '/opt/ml/detection/level2-object-detection-level2-cv-10/mmdetection/configs/_youngwoo_/_base_/default_runtime.py',
]
# model settings
pretrained = '/opt/ml/detection/level2-object-detection-level2-cv-10/mmdetection/configs/_youngwoo_/nvidia_efficientnet-b4_210412.pth'
model = dict(
    type='RetinaNet',
    backbone=dict(
        type='EfficientNet',
        model_name='tf_efficientnet_b4',
        pretrained=pretrained),
    neck=dict(
        type='FPN',
        in_channels=[56, 112, 160, 272, 448],
        out_channels=224,
        num_outs=5),
    bbox_head=dict(
        type='RetinaHead',
        num_classes=10,
        in_channels=224,#256->224
        stacked_convs=4,
        feat_channels=224,#256->224
        anchor_generator=dict(
                     type='AnchorGenerator',
                     octave_base_scale=4,
                     scales_per_octave=3,
                     ratios=[0.5, 0.7, 1.0, 1.5, 2.0],
                     strides=[8, 16, 32, 64, 128]),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=1.5, #2->1.5
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=0.11, loss_weight=1.0)))
# training and testing settings
train_cfg = dict(
    assigner=dict(
        type='MaxIoUAssigner',
        pos_iou_thr=0.5,
        neg_iou_thr=0.4,
        min_pos_iou=0,
        ignore_iof_thr=-1),
    allowed_border=-1,
    pos_weight=-1,
    debug=False)
test_cfg = dict(
    nms_pre=1000,
    min_bbox_size=0,
    score_thr=0.05,
    nms=dict(type='nms', iou_thr=0.5),
    max_per_img=100)
# dataset settings
# ...
# optimizer
# ...
optimizer = dict(
    type='AdamW',
    lr=0.0001,
    betas=(0.9, 0.999),
    weight_decay=4e-5,
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))
optimizer_config = dict(grad_clip=dict(_delete_=True, max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=1.0 / 10,
    min_lr_ratio=5e-5
    )
runner = dict(type='EpochBasedRunner', max_epochs=100) # epoch 100