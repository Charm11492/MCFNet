#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.

import torch.nn as nn

from .yolo_head import YOLOXHead
# from .yolo_pafpn import YOLOPAFPN
# from .yolo_pafpn_concat import YOLOPAFPN
# from .yolo_pafpn_concat import YOLOPAFPN
from .yolo_pafpn_concatadd import YOLOPAFPN
class YOLOX(nn.Module):
    """
    YOLOX model module. The module list is defined by create_yolov3_modules function.
    The network returns loss values from three YOLO layers during training
    and detection results during test.
    """

    def __init__(self, backbone=None, head=None):
        super().__init__()
        if backbone is None:
            backbone = YOLOPAFPN()
        if head is None:
            head = YOLOXHead(80)

        self.backbone = backbone
        self.head = head

    def forward(self, x, targets=None):
   
        # fpn output content features of [dark3, dark4, dark5]
        img = x["event"].to('cuda')
        img_rgb = x["image"].to('cuda')
        
        fpn_outs = self.backbone(img_rgb,img)
        # cv_rgb,cv_event,fpn_outs = self.backbone(img_rgb,img)

        if self.training:
            assert targets is not None
            loss, iou_loss, conf_loss, cls_loss, l1_loss, num_fg = self.head(
                fpn_outs, targets, x
            )
            # print(self.head)
            # loss, iou_loss, conf_loss, cls_loss, l1_loss,kl_div_loss, num_fg = self.head(
            #     cv_rgb,cv_event,fpn_outs, targets, x
            # )
            outputs = {
                "total_loss": loss,
                "iou_loss": iou_loss,
                "l1_loss": l1_loss,
                "conf_loss": conf_loss,
                "cls_loss": cls_loss,
           
                "num_fg": num_fg,
            }
        else:
            outputs = self.head(fpn_outs)
            # outputs = self.head(cv_rgb,cv_event,fpn_outs)

        return outputs

    def visualize(self, x, targets, save_prefix="assign_vis_"):
        fpn_outs = self.backbone(x)
        self.head.visualize_assign_result(fpn_outs, targets, x, save_prefix)
