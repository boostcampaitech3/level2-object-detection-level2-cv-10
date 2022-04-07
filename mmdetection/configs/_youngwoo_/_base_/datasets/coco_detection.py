# dataset settings
dataset_type = 'CocoDataset'
data_root = '/opt/ml/detection/dataset/' 

classes = ("General trash", "Paper", "Paper pack", "Metal", "Glass",
            "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# pipeline -> augmentation, TTA
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1024, 1024), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 1024),
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
