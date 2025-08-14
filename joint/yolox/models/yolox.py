#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.

import torch.nn as nn
import torch
from .yolo_head import YOLOXHead
# from .yolo_pafpn import YOLOPAFPN
from .yolo_pafpn_concat import YOLOPAFPN
import cv2
import sys

sys.path.append('/root/data1/code/YOLOX-mamba/YOLOX-main')
from yolox.flow_models.model import EVFlowNet
from yolox.loss.flow import EventWarping
from utils.iwe import deblur_events, compute_pol_iwe,compute_pol_iwe2
from yolox.data.data_augment import TrainTransform

def preprocess(inputs, rgb, targets, tsize, input_size=(480,640)):
    #rgb_input_size(960,1280) event_input_size(480,640)
        scale_y = tsize[0] / input_size[0]
        scale_x = tsize[1] / input_size[1]
        if scale_x != 1 or scale_y != 1:
            inputs = nn.functional.interpolate(
                inputs, size=tsize, mode="bilinear", align_corners=False
            )
            rgb = nn.functional.interpolate(
                rgb, size=tsize, mode="bilinear", align_corners=False
            )
            targets[..., 1::2] = targets[..., 1::2] * scale_x
            targets[..., 2::2] = targets[..., 2::2] * scale_y
        return inputs, rgb, targets

class YOLOX(nn.Module):
    """
    YOLOX model module. The module list is defined by create_yolov3_modules function.
    The network returns loss values from three YOLO layers during training
    and detection results during test.
    """

    def __init__(self,device,flow, backbone=None, head=None):
        super().__init__()
        if backbone is None:
            backbone = YOLOPAFPN()
        if head is None:
            head = YOLOXHead(8)
        
        self.hsv_prob = 0
        # prob of applying flip aug
        self.flip_prob = 0
        self.flow = flow
        self.backbone = backbone
        self.head = head
        self.device = device
        
    def forward(self, input_size, x, targets=None):
        # 输入voxel
        # FlowNet处理,得到flow
        voxel = x["inp_voxel"].to(self.device)
        image = x["image"].to(self.device)
        image = image.to(torch.float32)
        voxel = voxel.to(torch.float32)
       
        flow_map = self.flow(voxel)
        # image of warped events
        
        if targets is not None:
            iwe = compute_pol_iwe(
                flow_map["flow"][-1].to(self.device),
                x["inp_list"].to(self.device),
                [480, 640],
                # [256,320],
                x["inp_pol_mask"][:, 0:1, :].permute(0, 2, 1).to(self.device),
                x["inp_pol_mask"][:, 1:2, :].permute(0, 2, 1).to(self.device),
                x["mapping"],
                flow_scaling=128,
                round_idx=False,
            )
            preproc=TrainTransform(
                    max_labels=120,
                    flip_prob=self.flip_prob,
                    hsv_prob=self.hsv_prob
                    )
            # iwe = iwe.squeeze(0)
            # iwe, targets = preproc(iwe, targets, (480,640))
            # iwe = iwe.unsqueeze(0)
            # iwe, targets = preprocess(iwe, targets, input_size)

            imgs = []
            t = []
            rgbs =[]
            for i in range(iwe.shape[0]):
                img, rgb, target = preproc(iwe[i],image[i], targets[i], (480,640))
                t.append(target)
                imgs.append(img)
                rgbs.append(rgb)
            image = torch.stack(rgbs,dim=0)
            # 将所有图像堆叠成 (n, c, H, W) 格式
            iwe = torch.stack(imgs, dim=0)
            targets = torch.stack(t, dim=0)
            # iwe, targets = preprocess(iwe, targets, input_size)
            iwe, image,targets = preprocess(iwe, image, targets, input_size)

        # 调整维度顺序，将形状变为 (1, 3, h, w)
        else:
            iwe = compute_pol_iwe2(
                flow_map["flow"][-1].to(self.device),
                x["inp_list"].to(self.device),
                [480, 640],
                # [256,320],
                x["inp_pol_mask"][:, 0:1, :].permute(0, 2, 1).to(self.device),
                x["inp_pol_mask"][:, 1:2, :].permute(0, 2, 1).to(self.device),
                x["mapping"],
                x["name"],
                flow_scaling=128,
                round_idx=False,
            )
            iwe = iwe.permute(0, 3, 1, 2).to(self.device)
            image = image.permute(0, 3, 1, 2).to(self.device)
    
        # print(iwe.shape)
        
        # self.dataset = MosaicDetection(
        #     dataset=self.dataset,
        #     mosaic=not no_aug,
        #     img_size=self.input_size,
        #     preproc=TrainTransform(
        #         max_labels=120,
        #         flip_prob=self.flip_prob,
        #         hsv_prob=self.hsv_prob
        #      ),
        #     degrees=self.degrees,
        #     translate=self.translate,
        #     mosaic_scale=self.mosaic_scale,
        #     mixup_scale=self.mixup_scale,
        #     shear=self.shear,
        #     enable_mixup=self.enable_mixup,
        #     mosaic_prob=self.mosaic_prob,
        #     mixup_prob=self.mixup_prob,
        # ).to('gpu')
        # fpn output content features of [dark3, dark4, dark5]
        
        # fpn_outs = self.backbone(iwe)
        fpn_outs = self.backbone(image,iwe)

        # print(iwe)
        # print(fpn_outs)
        if self.training:
            assert targets is not None
            loss, iou_loss, conf_loss, cls_loss, l1_loss, num_fg = self.head(
                fpn_outs, targets, iwe
            )
            flow_lossfunction = EventWarping()
            flow_loss = 0.5*flow_lossfunction(flow_map["flow"][-1:], x["inp_list"], x["inp_pol_mask"].permute(0, 2, 1))
            loss = loss + flow_loss
            outputs = {
                "total_loss": loss,
                "iou_loss": iou_loss,
                "l1_loss": l1_loss,
                "conf_loss": conf_loss,
                "cls_loss": cls_loss,
                "num_fg": num_fg,
                "flow_loss": flow_loss
            }
        else:
            outputs = self.head(fpn_outs)

        return outputs

    def visualize(self, x, targets, save_prefix="assign_vis_"):
        fpn_outs = self.backbone(x)
        self.head.visualize_assign_result(fpn_outs, targets, x, save_prefix)
    
    
    
