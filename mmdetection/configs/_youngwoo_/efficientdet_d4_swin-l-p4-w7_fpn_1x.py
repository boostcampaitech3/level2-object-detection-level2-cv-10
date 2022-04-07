_base_ = [
    '/opt/ml/detection/level2-object-detection-level2-cv-10/mmdetection/configs/_youngwoo_/_base_/datasets/coco_detection_albu_1024_2.py',
    '/opt/ml/detection/level2-object-detection-level2-cv-10/mmdetection/configs/_youngwoo_/_base_/schedules/schedule_cosann_adamw.py',
    '/opt/ml/detection/level2-object-detection-level2-cv-10/mmdetection/configs/_youngwoo_/_base_/default_runtime.py',
]
pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window7_224_22k.pth'
# model settings
model = dict(
    type='RetinaNet',
    backbone=dict(
        type='SwinTransformer',
        embed_dims=192,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        with_cp=False,
        convert_weights=True,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)), 
    neck=dict(
        type='FPN',
        in_channels=[192, 384, 768, 1536],
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