_base_ = [
    '../_base_/models/universenet_swin_l_gfl.py',
    '../_base_/datasets/coco_detection_albu.py',
    '../_base_/schedules/schedule_universenet.py', '../_base_/default_runtime.py'
]

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # mlflow
        dict(
            type='MlflowLoggerHook',
            exp_name='0404_universenet',
        )
    ])
