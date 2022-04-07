_base_ = [
    '../_base_/models/cascade_swin_384_fpn.py',
    '../_base_/datasets/coco_detection_albu.py',
    '../_base_/schedules/schedule_cosann_adamw.py', '../_base_/default_runtime.py'
]