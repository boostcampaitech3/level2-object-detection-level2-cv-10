_base_ = './tood_r101_fpn.py'
model = dict(bbox_head=dict(anchor_type='anchor_based'))
