# dataset settings
dataset_type = 'CocoDataset'
data_root = '/opt/ml/detection/dataset/'

classes = ("General trash", "Paper", "Paper pack", "Metal", "Glass",
            "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

albu_train_transforms = [
    dict(
        type='ShiftScaleRotate',
        shift_limit=0.0625,
        scale_limit=0.0,
        rotate_limit=30,
        interpolation=1,
        p=0.3),
    dict(
        type='RandomBrightnessContrast',
        brightness_limit=[0.1, 0.3],
        contrast_limit=[0.1, 0.3],
        p=0.2),
    dict(
        type='OneOf',
        transforms=[
            dict(
                type='RGBShift',
                r_shift_limit=10,
                g_shift_limit=10,
                b_shift_limit=10,
                p=1.0),
            dict(
                type='HueSaturationValue',
                hue_shift_limit=20,
                sat_shift_limit=30,
                val_shift_limit=20,
                p=1.0)
        ],
        p=0.2),
    dict(type='JpegCompression', quality_lower=85, quality_upper=95, p=0.2),
    dict(type='ChannelShuffle', p=0.1),
    dict(
        type='OneOf',
        transforms=[
            dict(type='Blur', blur_limit=3, p=1.0),
            dict(type='MedianBlur', blur_limit=3, p=1.0)
        ],
        p=0.1),
    dict(type='CLAHE', clip_limit=4, p=0.3),
]

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    # dict(type='Resize', img_scale=(800, 800), keep_ratio=True), # 이미지 size!
    # dict(type='Pad', size_divisor=32),
    dict(
        type='AutoAugment',
        policies=[[
            dict(
                type='Resize',
                img_scale=[(480, 1024), (512, 1024), (544, 1024), (576, 1024),
                            (608, 1024), (640, 1024), (672, 1024), (704, 1024),
                            (736, 1024), (768, 1024), (800, 1024)],
                multiscale_mode='value',
                keep_ratio=True)
                ],
                [
                    dict(
                        type='Resize',
                        img_scale=[(400, 1024), (500, 1024), (600, 1024)],
                        multiscale_mode='value',
                        keep_ratio=True),
                    dict(
                        type='RandomCrop',
                        crop_type='absolute_range',
                        crop_size=(384, 600),
                        allow_negative_crop=True),
                    dict(
                        type='Resize',
                        img_scale=[(480, 1024), (512, 1024), (544, 1024),
                                    (576, 1024), (608, 1024), (640, 1024),
                                    (672, 1024), (704, 1024), (736, 1024),
                                    (768, 1024), (800, 1024)],
                        multiscale_mode='value',
                        override=True,
                        keep_ratio=True)
                ]]),
    dict(
        type='Albu',
        transforms=albu_train_transforms,
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_labels'],
            min_visibility=0.0,
            filter_lost_elements=True),
        keymap={
            'img': 'image',
            'gt_bboxes': 'bboxes'
        },
        update_pad_shape=False,
        skip_img_without_anno=True),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(
        type='Collect',
        keys=['img', 'gt_bboxes', 'gt_labels'],
        meta_keys=('filename', 'ori_shape', 'img_shape', 'img_norm_cfg',
                   'pad_shape', 'scale_factor'))
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 1024), # 800 -> 1024로 변경해봄
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=4, # batch size 2 -> 4로 변경
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        classes=classes, # 우리 데이터대로 추가
        ann_file=data_root + 'train0.json', # 아직 CV 안 나눴으므로 전체 json 넘김
        img_prefix=data_root, # 수정
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        classes=classes, # 우리 데이터대로 추가
        ann_file=data_root + 'val0.json', # 수정
        img_prefix=data_root, # 수정
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        classes=classes, # 우리 데이터대로 추가
        ann_file=data_root + 'test.json', # 수정
        img_prefix=data_root, # 수정
        pipeline=test_pipeline))

evaluation = dict(interval=1, metric='bbox', classwise=True, save_best='bbox_mAP_50')