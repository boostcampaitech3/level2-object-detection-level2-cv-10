checkpoint_config = dict(max_keep_ckpts=5, interval=1)
# checkpoint_config = dict(max_keep_ckpts=5, interval=1) # max 저장 개수 지정
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
        dict(type='MlflowLoggerHook',
            exp_name='Faster_RCNN_exp2')
    ])
# yapf:enable
custom_hooks = [dict(type='NumClassCheckHook')]

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]

# disable opencv multithreading to avoid system being overloaded
opencv_num_threads = 0
# set multi-process start method as `fork` to speed up the training
mp_start_method = 'fork'
