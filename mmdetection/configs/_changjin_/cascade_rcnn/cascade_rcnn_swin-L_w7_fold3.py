_base_ = [
    '../_base_/models/cascade_rcnn_swin-L_fpn.py',
    '../_base_/datasets/coco_detection_albu2.py',
    '../_base_/schedules/schedule_cosine.py', '../_base_/default_runtime.py'
]
