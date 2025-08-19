#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.

import cv2
import numpy as np

__all__ = ["vis"]


# def vis(img, boxes, scores, cls_ids, conf=0.5, class_names=None):

#     for i in range(len(boxes)):
#         box = boxes[i]
#         cls_id = int(cls_ids[i])
#         score = scores[i]
#         if score < conf:
#             continue
#         x0 = int(box[0])
#         y0 = int(box[1])
#         x1 = int(box[2])
#         y1 = int(box[3])
#         # print(box)
#         color = (_COLORS[cls_id] * 255).astype(np.uint8).tolist()
#         text = '{}:{:.1f}%'.format(class_names[cls_id], score * 100)
#         txt_color = (0, 0, 0) if np.mean(_COLORS[cls_id]) > 0.5 else (255, 255, 255)
#         font = cv2.FONT_HERSHEY_SIMPLEX
#         font_scale = 0.8  # 增大字体
#         thickness = 2  # 增加文字的粗细

#         txt_size = cv2.getTextSize(text, font, font_scale, thickness)[0]

#         # 增加矩形框的线条粗细
#         rectangle_thickness = 4  # 增加框的粗细
#         cv2.rectangle(img, (x0, y0), (x1, y1), color, rectangle_thickness)

#         # 背景矩形框，用于文字
#         txt_bk_color = (_COLORS[cls_id] * 255 * 0.7).astype(np.uint8).tolist()
#         cv2.rectangle(
#             img,
#             (x0, y0 + 1),
#             (x0 + txt_size[0] + 1, y0 + int(1.5 * txt_size[1])),
#             txt_bk_color,
#             -1
#         )

#         # 绘制文字，增加字体大小和粗细
#         cv2.putText(img, text, (x0, y0 + txt_size[1]), font, font_scale, txt_color, thickness)

#     return img

def vis(img, boxes, scores, cls_ids, conf=0.5, class_names=None):
    for i in range(len(boxes)):
        box = boxes[i]
        cls_id = int(cls_ids[i])
        score = scores[i]
        if score < conf:
            continue
        x0 = int(box[0])
        y0 = int(box[1])
        x1 = int(box[2])
        y1 = int(box[3])

        color = (_COLORS[cls_id] * 255).astype(np.uint8).tolist()
        text = '{}:{:.1f}%'.format(class_names[cls_id], score * 100)
        txt_color = (0, 0, 0) if np.mean(_COLORS[cls_id]) > 0.5 else (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX

        txt_size = cv2.getTextSize(text, font, 0.8, 1)[0]  # 字体大小调整为0.8，粗细为2
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)

        # 显示框与文字在检测框的上方
        txt_bk_color = (_COLORS[cls_id] * 255 * 0.7).astype(np.uint8).tolist()
        cv2.rectangle(
            img,
            (x0, y0 - txt_size[1] - 5),  # 上方位置
            (x0 + txt_size[0], y0),
            txt_bk_color,
            -1
        )
        cv2.putText(img, text, (x0, y0 - 5), font, 0.8, txt_color, thickness=2)

    return img
_COLORS = np.array(
    [
        0.000, 0.000, 1.000,  # 红色
        1.000, 0.000, 0.000,  # 蓝色
        0.929, 0.694, 0.125,
        0.494, 0.184, 0.556,
        0.466, 0.674, 0.188,
        0.301, 0.745, 0.933,
        0.635, 0.078, 0.184,
        0.300, 0.300, 0.300,
        0.600, 0.600, 0.600,
        1.000, 0.000, 0.000,
        1.000, 0.500, 0.000,
        0.749, 0.749, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 1.000,
        0.667, 0.000, 1.000,
        0.333, 0.333, 0.000,
        0.333, 0.667, 0.000,
        0.333, 1.000, 0.000,
        0.667, 0.333, 0.000,
        0.667, 0.667, 0.000,
        0.667, 1.000, 0.000,
        1.000, 0.333, 0.000,
        1.000, 0.667, 0.000,
        1.000, 1.000, 0.000,
        0.000, 0.333, 0.500,
        0.000, 0.667, 0.500,
        0.000, 1.000, 0.500,
        0.333, 0.000, 0.500,
        0.333, 0.333, 0.500,
        0.333, 0.667, 0.500,
        0.333, 1.000, 0.500,
        0.667, 0.000, 0.500,
        0.667, 0.333, 0.500,
        0.667, 0.667, 0.500,
        0.667, 1.000, 0.500,
        1.000, 0.000, 0.500,
        1.000, 0.333, 0.500,
        1.000, 0.667, 0.500,
        1.000, 1.000, 0.500,
        0.000, 0.333, 1.000,
        0.000, 0.667, 1.000,
        0.000, 1.000, 1.000,
        0.333, 0.000, 1.000,
        0.333, 0.333, 1.000,
        0.333, 0.667, 1.000,
        0.333, 1.000, 1.000,
        0.667, 0.000, 1.000,
        0.667, 0.333, 1.000,
        0.667, 0.667, 1.000,
        0.667, 1.000, 1.000,
        1.000, 0.000, 1.000,
        1.000, 0.333, 1.000,
        1.000, 0.667, 1.000,
        0.333, 0.000, 0.000,
        0.500, 0.000, 0.000,
        0.667, 0.000, 0.000,
        0.833, 0.000, 0.000,
        1.000, 0.000, 0.000,
        0.000, 0.167, 0.000,
        0.000, 0.333, 0.000,
        0.000, 0.500, 0.000,
        0.000, 0.667, 0.000,
        0.000, 0.833, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 0.167,
        0.000, 0.000, 0.333,
        0.000, 0.000, 0.500,
        0.000, 0.000, 0.667,
        0.000, 0.000, 0.833,
        0.000, 0.000, 1.000,
        0.000, 0.000, 0.000,
        0.143, 0.143, 0.143,
        0.286, 0.286, 0.286,
        0.429, 0.429, 0.429,
        0.571, 0.571, 0.571,
        0.714, 0.714, 0.714,
        0.857, 0.857, 0.857,
        0.000, 0.447, 0.741,
        0.314, 0.717, 0.741,
        0.50, 0.5, 0
    ]
).astype(np.float32).reshape(-1, 3)
