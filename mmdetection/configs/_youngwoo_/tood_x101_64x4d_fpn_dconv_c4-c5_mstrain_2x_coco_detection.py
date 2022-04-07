_base_ = [
    '/opt/ml/detection/level2-object-detection-level2-cv-10/mmdetection/configs/_youngwoo_/tood_x101_64x4d_fpn_mstrain_2x_coco.py',
    '/opt/ml/detection/level2-object-detection-level2-cv-10/mmdetection/configs/_youngwoo_/_base_/datasets/coco_detection.py'
]
model = dict(
    backbone=dict(
        dcn=dict(type='DCNv2', deformable_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, False, True, True),
    ),
    bbox_head=dict(num_dcn=2))

runner = dict(type='EpochBasedRunner', max_epochs=24) # epoch