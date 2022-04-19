_base_ = [
    '../_base_/models/cascade_rcnn_swin_l_rfp.py',
    '../_base_/datasets/coco_detection_albu.py',
    '../_base_/schedules/schedule_cosann_adamw.py', '../_base_/default_runtime.py'
]
# swin과 rfp 결합이 어려워서 보류한다.
