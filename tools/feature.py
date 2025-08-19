#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import os
import time
from loguru import logger
import sys
import matplotlib.pyplot as plt
import numpy as np
sys.path.append('/root/data1/standby/YOLOX-mamba')
import cv2
from yolox.data.datasets.mapping import *
import torch

from yolox.data.data_augment import ValTransform
from yolox.data.datasets import COCO_CLASSES
from yolox.exp import get_exp
from yolox.utils import fuse_model,  postprocess, vis
from yolox.utils.draw_cam import *
IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]


def make_parser():
    parser = argparse.ArgumentParser("YOLOX Demo!")
    parser.add_argument(
        "demo", default="image", help="demo type, eg. image, video and webcam"
    )
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    parser.add_argument(
        "--path", default="/root/data1/dataset/DSEC/train/images/zurich_city_09_a/images/left/rectified/001006.png", help="path to images or video"
    )
    parser.add_argument("--camid", type=int, default=0, help="webcam demo camera id")
    parser.add_argument(
        "--save_result",
        action="store_true",
        help="whether to save the inference result of image/video",
    )

    # exp file
    parser.add_argument(
        "-f",
        "--exp_file",
        default="/root/data1/standby/YOLOX-mamba/exps/example/yolox_voc/yolox_DSEC.py",
        type=str,
        help="please input your experiment description file",
    )
    parser.add_argument("-c", "--ckpt", default="/root/data1/standby/YOLOX-mamba/YOLOX_outputs/yolox_gradcam/epoch_6_ckpt.pth", type=str, help="ckpt for eval")
    parser.add_argument(
        "--device",
        default="cpu",
        type=str,
        help="device to run our model, can either be cpu or gpu",
    )
    parser.add_argument("--conf", default=0.3, type=float, help="test conf")
    parser.add_argument("--nms", default=0.3, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision evaluating.",
    )
    parser.add_argument(
        "--legacy",
        dest="legacy",
        default=False,
        action="store_true",
        help="To be compatible with older versions",
    )
    parser.add_argument(
        "--fuse",
        dest="fuse",
        default=False,
        action="store_true",
        help="Fuse conv and bn for testing.",
    )
    parser.add_argument(
        "--trt",
        dest="trt",
        default=False,
        action="store_true",
        help="Using TensorRT model for testing.",
    )
    return parser


def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            ext = os.path.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names


class FeatureExtractor():
    def __init__(self, layer_num):
        self.features = None
        self.layer_num = layer_num
        
    def hook(self, module, input, output):
        self.features = output.detach()
def visualize_feature_maps_mean(feature_maps, save_path):
    """通过平均值可视化特征图，添加诊断信息
    
    Args:
        feature_maps: 特征图张量 (192, 80, 80)
        save_path: 保存路径
    """
    # 获取特征图并计算平均值
    features = feature_maps[0].cpu().numpy()  # [C, H, W]
    mean_projection = np.mean(features, axis=0)  # [H, W]
    
    # 打印特征图的统计信息
    print(f"Feature map shape: {features.shape}")
    print(f"Value range: [{features.min():.3f}, {features.max():.3f}]")
    print(f"Mean projection shape: {mean_projection.shape}")
    
    # 保存原始数值到文本文件
    np.savetxt(save_path.replace('.png', '_raw.txt'), mean_projection, fmt='%.6f')
    
    # 检查是否有任何形式的分块
    pixel_diff_h = np.abs(mean_projection[1:] - mean_projection[:-1])
    pixel_diff_v = np.abs(mean_projection[:,1:] - mean_projection[:,:-1])
    print(f"Average horizontal difference: {pixel_diff_h.mean():.6f}")
    print(f"Average vertical difference: {pixel_diff_v.mean():.6f}")
    print(f"Max horizontal difference: {pixel_diff_h.max():.6f}")
    print(f"Max vertical difference: {pixel_diff_v.max():.6f}")
    
    # 归一化到0-255范围并保存为图像
    scale_factor = 4  # 放大倍数
    h, w = mean_projection.shape
    mean_projection = cv2.resize(mean_projection, 
                               (w * scale_factor, h * scale_factor),
                               interpolation=cv2.INTER_LINEAR)
    mean_projection_norm = ((mean_projection - mean_projection.min()) / 
                          (mean_projection.max() - mean_projection.min()) * 255).astype(np.uint8)
    
    # 保存原始灰度图
    cv2.imwrite(save_path.replace('.png', '_gray.png'), mean_projection_norm)
    
    # 应用颜色映射并保存
    colored = cv2.applyColorMap(mean_projection_norm, cv2.COLORMAP_JET)
    cv2.imwrite(save_path, colored)

def visualize_feature_maps_max(feature_maps, save_path):
    """通过最大值可视化特征图，添加诊断信息
    
    Args:
        feature_maps: 特征图张量 (192, 80, 80)
        save_path: 保存路径
    """
    # 获取特征图并计算最大值
    features = feature_maps[0].cpu().numpy()  # [C, H, W]
    max_projection = np.max(features, axis=0)  # [H, W]
    
    # 打印特征图的统计信息
    print(f"Feature map shape: {features.shape}")
    print(f"Value range: [{features.min():.3f}, {features.max():.3f}]")
    print(f"Max projection shape: {max_projection.shape}")
    
    # 保存原始数值到文本文件
    np.savetxt(save_path.replace('.png', '_raw.txt'), max_projection, fmt='%.6f')
    
    # 检查是否有任何形式的分块
    pixel_diff_h = np.abs(max_projection[1:] - max_projection[:-1])
    pixel_diff_v = np.abs(max_projection[:,1:] - max_projection[:,:-1])
    print(f"Average horizontal difference: {pixel_diff_h.mean():.6f}")
    print(f"Average vertical difference: {pixel_diff_v.mean():.6f}")
    print(f"Max horizontal difference: {pixel_diff_h.max():.6f}")
    print(f"Max vertical difference: {pixel_diff_v.max():.6f}")
    
    # 归一化到0-255范围并保存为图像
    scale_factor = 4  # 放大倍数
    h, w = max_projection.shape
    max_projection = cv2.resize(max_projection, 
                               (w * scale_factor, h * scale_factor),
                               interpolation=cv2.INTER_LINEAR)
    
    max_projection_norm = ((max_projection - max_projection.min()) / 
                         (max_projection.max() - max_projection.min()) * 255).astype(np.uint8)
    
    # 保存原始灰度图
    cv2.imwrite(save_path.replace('.png', '_gray.png'), max_projection_norm)
    
    # 应用颜色映射并保存
    colored = cv2.applyColorMap(max_projection_norm, cv2.COLORMAP_JET)
    cv2.imwrite(save_path, colored)
class Predictor(object):
    def __init__(
            self,
            model,
            exp,
            cls_names=COCO_CLASSES,
            trt_file=None,
            decoder=None,
            device="cpu",
            fp16=False,
            legacy=False,
    ):
        self.model = model
        self.cls_names = cls_names
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = (960,1280)
        self.device = device
        # self.fp16 = fp16
        self.preproc = ValTransform(legacy=legacy)
        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, exp.test_size[0], exp.test_size[1]).cuda()
            self.model(x)
            self.model = model_trt

    ####   new code
    def get_tensor(self, img_path,event_path):
        # print(img_path)
        img_rgb = cv2.imread(img_path)
        img_file = img_path
        mapx = np.zeros((480, 640), dtype=np.float32)
        mapy = np.zeros((480, 640), dtype=np.float32)
        K_dist = None
        dist_coeffs = None
        R_rect0 = None
        K_rect = None
        if 'interlaken' in img_file or 'thun' in img_file:
            mapx, mapy, K_dist, dist_coeffs, R_rect0, K_rect = interlaken_thun()
        if 'zurich_city' in img_file and ('00' in img_file or '01' in img_file or '02' in img_file or 
            '03' in img_file or '09' in img_file or '10' in img_file or '12' in img_file or '14' in img_file):
            mapx, mapy, K_dist, dist_coeffs, R_rect0, K_rect = zurich_city_00_01_02_03_09_10_12_14()
        if 'zurich_city' in img_file and ('04' in img_file or '05' in img_file or '11' in img_file):
            mapx, mapy, K_dist, dist_coeffs, R_rect0, K_rect = zurich_city_04_05_11()
        if 'zurich_city' in img_file and ('06' in img_file or '07' in img_file or '13' in img_file):
            mapx, mapy, K_dist, dist_coeffs, R_rect0, K_rect = zurich_city_06_07_13()
        if 'zurich_city' in img_file and ('08' in img_file or '15'):
            mapx, mapy, K_dist, dist_coeffs, R_rect0, K_rect = zurich_city_08_15()
        mapping = cv2.initUndistortRectifyMap(K_dist, dist_coeffs, R_rect0, K_rect, resolution, cv2.CV_32FC2)[0]
       
        mapx_resized = cv2.resize(mapx, (1280, 960), interpolation=cv2.INTER_LINEAR)
        mapy_resized = cv2.resize(mapy, (1280, 960), interpolation=cv2.INTER_LINEAR)
        img_rgb = cv2.remap(img_rgb, mapx_resized, mapy_resized, cv2.INTER_LINEAR)

        # cv2.imwrite(f'/root/data1/standby/YOLOX-mamba/datasets/14c.jpg', img_rgb)
        image = cv2.imread(event_path)
        # print(image)

        # print(image.shape)
        output = {}
        output["event"] = image
        output["image"] = img_rgb
        output, _ = self.preproc(output, None, self.test_size)
        image = output["event"]
        img_rgb = output["image"]
        image = torch.from_numpy(image).unsqueeze(0)
        image = image.float()
        img_rgb = torch.from_numpy(img_rgb).unsqueeze(0)
        img_rgb = img_rgb.float()
        output = {}
        output["event"] = image
        output["image"] = img_rgb
        return output
    ####   new code
    def inference(self, img):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = os.path.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        ratio = min(self.test_size[0] / img.shape[0], self.test_size[1] / img.shape[1])
        img_info["ratio"] = ratio

        img, _ = self.preproc(img, None, self.test_size)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.float()
        if self.device == "gpu":
            img = img.cuda()
            # if self.fp16:
            #     img = img.half()  # to FP16

        with torch.no_grad():
            t0 = time.time()
            outputs = self.model(img)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = postprocess(
                outputs, self.num_classes, self.confthre,
                self.nmsthre, class_agnostic=True
            )
            logger.info("Infer time: {:.4f}s".format(time.time() - t0))
        return outputs, img_info

    def visual(self, output, img_info, cls_conf=0.35):
        ratio = img_info["ratio"]
        img = img_info["raw_img"]
        if output is None:
            return img
        output = output.cpu()

        bboxes = output[:, 0:4]

        # preprocessing: resize
        bboxes /= ratio

        cls = output[:, 6]
        scores = output[:, 4] * output[:, 5]

        vis_res = vis(img, bboxes, scores, cls, cls_conf, self.cls_names)
        return vis_res


def image_demo(predictor, vis_folder, path, current_time, save_result):
    if os.path.isdir(path):
        files = get_image_list(path)
    else:
        files = [path]
    files.sort()
    for image_name in files:
        outputs, img_info = predictor.inference(image_name)
        result_image = predictor.visual(outputs[0], img_info, predictor.confthre)
        if save_result:
            save_folder = os.path.join(
                vis_folder, time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
            )
            os.makedirs(save_folder, exist_ok=True)
            save_file_name = os.path.join(save_folder, os.path.basename(image_name))
            logger.info("Saving detection result in {}".format(save_file_name))
            cv2.imwrite(save_file_name, result_image)
        ch = cv2.waitKey(0)
        if ch == 27 or ch == ord("q") or ch == ord("Q"):
            break


def imageflow_demo(predictor, vis_folder, current_time, args):
    cap = cv2.VideoCapture(args.path if args.demo == "video" else args.camid)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    fps = cap.get(cv2.CAP_PROP_FPS)
    if args.save_result:
        save_folder = os.path.join(
            vis_folder, time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
        )
        os.makedirs(save_folder, exist_ok=True)
        if args.demo == "video":
            save_path = os.path.join(save_folder, os.path.basename(args.path))
        else:
            save_path = os.path.join(save_folder, "camera.mp4")
        logger.info(f"video save_path is {save_path}")
        vid_writer = cv2.VideoWriter(
            save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
        )
    while True:
        ret_val, frame = cap.read()
        if ret_val:
            outputs, img_info = predictor.inference(frame)
            result_frame = predictor.visual(outputs[0], img_info, predictor.confthre)
            if args.save_result:
                vid_writer.write(result_frame)
            else:
                cv2.namedWindow("yolox", cv2.WINDOW_NORMAL)
                cv2.imshow("yolox", result_frame)
            ch = cv2.waitKey(1)
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                break
        else:
            break


def main(exp, args):
    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    file_name = os.path.join(exp.output_dir, args.experiment_name)
    os.makedirs(file_name, exist_ok=True)

    vis_folder = None
    if args.save_result:
        vis_folder = os.path.join(file_name, "vis_res")
        os.makedirs(vis_folder, exist_ok=True)

    if args.trt:
        args.device = "gpu"

    logger.info("Args: {}".format(args))

    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    model = exp.get_model()
    # logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))

    # if args.device == "gpu":
    model.cuda()
        # if args.fp16:
        #     model.half()  # to FP16
    model.eval()

        
    # 选择要可视化的层
    # 这里需要根据你的模型结构选择正确的层
    target_layer = model.backbone.backbone.dark3_event  # 这只是一个例子，需要根据实际模型结构调整
    
    # 创建特征提取器
    feature_extractor = FeatureExtractor(layer_num=5)  # layer_num可以根据需要修改
    
    # 注册hook
    hook_handle = target_layer.register_forward_hook(feature_extractor.hook)

    if not args.trt:
        if args.ckpt is None:
            ckpt_file = os.path.join(file_name, "best_ckpt.pth")
        else:
            ckpt_file = args.ckpt
        logger.info("loading checkpoint")
        ckpt = torch.load(ckpt_file, map_location="cpu")
        # load the model state dict
        model.load_state_dict(ckpt["model"])
        logger.info("loaded checkpoint done.")

    if args.fuse:
        logger.info("\tFusing model...")
        model = fuse_model(model)

    if args.trt:
        assert not args.fuse, "TensorRT model is not support model fusing!"
        trt_file = os.path.join(file_name, "model_trt.pth")
        assert os.path.exists(
            trt_file
        ), "TensorRT model is not found!\n Run python3 tools/trt.py first!"
        model.head.decode_in_inference = False
        decoder = model.head.decode_outputs
        logger.info("Using TensorRT to inference")
    else:
        trt_file = None
        decoder = None

        # 假设`Predictor` 类已经初始化
    predictor = Predictor(
        model, exp, COCO_CLASSES, trt_file, decoder,
        args.device, args.fp16, args.legacy,
    )

    # 定义根目录
    rgb_root_dir = "/root/data1/dataset/DSEC/train/images"
    event_root_dir = "/root/data1/dataset/DSEC/iwe"

    # 遍历所有序列文件夹
    # 
    # filelist = [  'interlaken_00_b','zurich_city_13_a','zurich_city_13_b'
    # ]
    filelist = [  'zurich_city_09_a'
    ]
     # 遍历所有序列文件夹
    for sequence in os.listdir(event_root_dir):
        if sequence in filelist:
            rgb_sequence_path = os.path.join(rgb_root_dir, sequence, "images/left/rectified")
            event_sequence_path = os.path.join(event_root_dir, sequence)
            
            for img_name in os.listdir(event_sequence_path):
                rgb_path = os.path.join(rgb_sequence_path, img_name)
                event_path = os.path.join(event_sequence_path, img_name)
                
                if os.path.exists(event_path):
                    img = predictor.get_tensor(rgb_path, event_path)
                    # print(img)
                    # 运行模型获取特征
                    with torch.no_grad():
                        # _ = model(img["image"].cuda(), img["event"].cuda())
                        _ = model(img)
                    
                    # 可视化特征图
                    save_dir = os.path.join(vis_folder, sequence)
                    os.makedirs(save_dir, exist_ok=True)
                    save_path = os.path.join(save_dir, f'feature_maps_{img_name}.png')
                    print(save_path)
                    # visualize_feature_maps(feature_extractor.features, save_path)
                     # 最大值投影
                    save_path_max = os.path.join(save_dir, f'feature_max_{img_name}.png')
                    visualize_feature_maps_max(feature_extractor.features, save_path_max)
                    
                    # 平均值投影
                    save_path_mean = os.path.join(save_dir, f'feature_mean_{img_name}.png')
                    visualize_feature_maps_mean(feature_extractor.features, save_path_mean)
    
    
    # 移除hook
    hook_handle.remove()

    # # 遍历所有序列文件夹
    # for sequence in os.listdir(event_root_dir):
    #     if sequence in filelist:
    #         rgb_sequence_path = os.path.join(rgb_root_dir, sequence, "images/left/rectified")
    #         event_sequence_path = os.path.join(event_root_dir, sequence)

    #         # 遍历序列文件夹中的所有图像
    #         for img_name in os.listdir(event_sequence_path):
    #             # 构造每一张图像的完整路径
        
    #             rgb_path = os.path.join(rgb_sequence_path, img_name)
    #             event_path = os.path.join(event_sequence_path, img_name)

    #             # 确保事件图像和 RGB 图像匹配存在
    #             if os.path.exists(event_path):
    #                 # 获取图像的 tensor 表示
    #                 img = predictor.get_tensor(rgb_path, event_path)
                    
    #                 # 进行 Grad-CAM 可视化
    #                 get_cam(model, img, rgb_path,event_path, exp.test_size)
                    
    #                 # 记录当前时间
    #                 current_time = time.localtime()

    #             else:
    #                 print(f"Event image {event_path} not found for RGB image {rgb_path}")

if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)

    main(exp, args)