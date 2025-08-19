# 对单个图像可视化
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, \
    deprocess_image, \
    preprocess_image
import cv2
import numpy as np
import os
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from yolox.data.datasets.mapping import *

def get_cam(model, image_tensor, image_path,event_path, test_size):
    # 1.加载模型
    # 2.选择目标层

    # for name in model.named_modules():
    #     print(name)

    target_layer = [model.backbone.backbone.dark4]
    #linear_transform_layer4
    # 3. 构建输入图像的Tensor形式
    # rgb_img = cv2.imread(image_path, 1)[:, :, ::-1]  # 1是读取rgb
    rgb_img = cv2.imread(image_path, 1)[:, :, ::-1] 
    mapx = np.zeros((480, 640), dtype=np.float32)
    mapy = np.zeros((480, 640), dtype=np.float32)
    K_dist = None
    dist_coeffs = None
    R_rect0 = None
    K_rect = None
    img_file = image_path
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
    rgb_img = cv2.remap(rgb_img, mapx_resized, mapy_resized, cv2.INTER_LINEAR)
    
    if test_size is not None:
        # rgb_img = cv2.resize(rgb_img, [test_size[0], test_size[1]])
        if len(rgb_img.shape) == 3:
            padded_img = np.ones((test_size[0], test_size[1], 3), dtype=np.uint8) * 114
           
        else:
            padded_img = np.ones(test_size, dtype=np.uint8) * 114
    
        r = min(test_size[0] / rgb_img.shape[0], test_size[1] / rgb_img.shape[1])
        resized_img = cv2.resize(
            rgb_img,
            (int(rgb_img.shape[1] * r), int(rgb_img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
      
        padded_img[: int(rgb_img.shape[0] * r), : int(rgb_img.shape[1] * r)] = resized_img
      

    rgb_img = np.float32(padded_img) / 255
    # cv2.imwrite(f'/root/data1/standby/YOLOX-mamba/datasets/14c_0006.jpg', rgb_img)
    #
    # # preprocess_image作用：归一化图像，并转成tensor
    # input_tensor = preprocess_image(rgb_img, mean=[0.485, 0.456, 0.406],
    #                                 std=[0.229, 0.224, 0.225])  # torch.Size([1, 3, 224, 224])
    # Create an input tensor image for your model..
    # Note: input_tensor can be a batch tensor with several images!

    # Construct the CAM object once, and then re-use it on many images:
    # 4.初始化GradCAM，包括模型，目标层以及是否使用cuda
    # cam = GradCAM(model=model, target_layers=target_layer, use_cuda=True)
    cam = GradCAM(model=model, target_layers=target_layer)

    # If target_category is None, the highest scoring category
    # will be used for every image in the batch.
    # target_category can also be an integer, or a list of different integers
    # for every image in the batch.
    # 5.选定目标类别，如果不设置，则默认为分数最高的那一类
    target_category = None

    # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
    # 6. 计算cam
    grayscale_cam = cam(input_tensor=image_tensor, targets=target_category)  # [batch, 224,224]
    # print(grayscale_cam.shape)
    sequence_name = os.path.basename(os.path.dirname(event_path))
 # 提取序列名，如 zurich_city_09_a
    image_name = os.path.splitext(os.path.basename(image_path))[0]  # 提取图像名（不含扩展名），如 001106
    
    # 创建保存路径，包含序列名和图像名
    save_dir = "/root/data1/standby/mamba"
    save_dir = os.path.join(save_dir,sequence_name)
    os.makedirs(save_dir, exist_ok=True)
    save_path = f"{save_dir}/{image_name}.jpg"

    for i in range(grayscale_cam.shape[0]):  # 遍历每个批次
        single_grayscale_cam = grayscale_cam[i]  # 选取单个热力图
        single_grayscale_cam =cv2.resize(
            single_grayscale_cam,
            (int(rgb_img.shape[1]), int(rgb_img.shape[0])),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        visualization = show_cam_on_image(rgb_img, single_grayscale_cam)  # (224, 224, 3)
        
        # 保存可视化结果，使用包含序列名和图像名的文件名
        cv2.imwrite(save_path.replace('.jpg', f'_{i}.jpg'), visualization)