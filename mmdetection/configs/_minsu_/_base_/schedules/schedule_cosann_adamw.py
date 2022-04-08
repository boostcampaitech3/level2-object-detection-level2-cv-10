# optimizer
optimizer = dict(
    type='AdamW',
    lr=0.01,
    betas=(0.9, 0.999),
    weight_decay=0.0001,
    )
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=1.0 / 10,
    min_lr_ratio=1e-6,
    by_epoch=True # by epoch으로!
    )
runner = dict(type='EpochBasedRunner', max_epochs=50) # epoch은 15로
