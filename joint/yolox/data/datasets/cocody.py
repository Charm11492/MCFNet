#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.
import copy
import os

import h5py
import cv2
import numpy as np
from pycocotools.coco import COCO
import torch
import sys

sys.path.append('/root/data1/code/YOLOX-mamba/YOLOX-main')
# from ..dataloading import get_yolox_datadir
from .datasets_wrapper import CacheDataset, cache_read_img
from .mapping import *
import hdf5plugin



def remove_useless_info(coco):
    """
    Remove useless info in coco dataset. COCO object is modified inplace.
    This function is mainly used for saving memory (save about 30% mem).
    """
    if isinstance(coco, COCO):
        dataset = coco.dataset
        dataset.pop("info", None)
        dataset.pop("licenses", None)
        for img in dataset["images"]:
            img.pop("license", None)
            img.pop("coco_url", None)
            img.pop("date_captured", None)
            img.pop("flickr_url", None)
        if "annotations" in coco.dataset:
            for anno in coco.dataset["annotations"]:
                anno.pop("segmentation", None)
def event_formatting(xs, ys, ts, ps):
        """
        Reset sequence-specific variables.
        :param xs: [N] numpy array with event x location
        :param ys: [N] numpy array with event y location
        :param ts: [N] numpy array with event timestamp
        :param ps: [N] numpy array with event polarity ([-1, 1])
        :return xs: [N] tensor with event x location
        :return ys: [N] tensor with event y location
        :return ts: [N] tensor with normalized event timestamp
        :return ps: [N] tensor with event polarity ([-1, 1])
        """

        xs = torch.from_numpy(xs.astype(np.float32))
        ys = torch.from_numpy(ys.astype(np.float32))
        ts = torch.from_numpy(ts.astype(np.float32))
        ps = torch.from_numpy(ps.astype(np.float32)) * 2 - 1
        # print(torch.max(ts))
        # ts = (ts - ts[0]) / (ts[-1] - ts[0])
        # print(torch.max(ts))
        return xs, ys, ts, ps

def create_list_encoding(xs, ys, ts, ps):
        """
        Creates a four channel tensor with all the events in the input partition.
        :param xs: [N] tensor with event x location
        :param ys: [N] tensor with event y location
        :param ts: [N] tensor with normalized event timestamp
        :param ps: [N] tensor with event polarity ([-1, 1])
        :return [N x 4] event representation
        """

        return torch.stack([ts, ys, xs, ps], dim=1)

def create_polarity_mask(ps):
        """
        Creates a two channel tensor that acts as a mask for the input event list.
        :param ps: [N] tensor with event polarity ([-1, 1])
        :return [N x 2] event representation
        """

        inp_pol_mask = torch.stack([ps, ps])
        inp_pol_mask[0, :][inp_pol_mask[0, :] < 0] = 0
        inp_pol_mask[1, :][inp_pol_mask[1, :] > 0] = 0
        inp_pol_mask[1, :] *= -1
        return inp_pol_mask


class COCODataset(CacheDataset):
    """
    COCO dataset class.
    """

    def __init__(
        self,
        data_dir=None,
        voxel_dir=None,
        json_file="instances_train2017.json",
        name="train2017",
        img_size=(416, 416),
        preproc=None,
        cache=False,
        cache_type="ram",
    ):
        """
        COCO dataset initialization. Annotation data are read into memory by COCO API.
        Args:
            data_dir (str): dataset root directory
            json_file (str): COCO json file name
            name (str): COCO data name (e.g. 'train2017' or 'val2017')
            img_size (int): target image size after pre-processing
            preproc: data augmentation strategy
        """
     
        self.data_dir = data_dir
        self.voxel_dir = voxel_dir
        self.json_file = json_file
        self.event_dir = "/root/data1/dataset/DSEC/train/events"

        self.coco = COCO(self.json_file)
        # remove_useless_info(self.coco)
        self.ids = self.coco.getImgIds()
        self.num_imgs = len(self.ids)
        self.class_ids = sorted(self.coco.getCatIds())
        self.cats = self.coco.loadCats(self.coco.getCatIds())
        self._classes = tuple([c["name"] for c in self.cats])
 
        self.img_size = img_size
        self.preproc = preproc
        self.annotations = self._load_coco_annotations()


        path_filename = [os.path.join(name, anno[3]) for anno in self.annotations]
        super().__init__(
            input_dimension=img_size,
            # num_imgs=self.num_imgs,
            data_dir=data_dir,
            cache_dir_name=f"cache_{name}",
            path_filename=path_filename,
            cache=cache,
            cache_type=cache_type
        )

    def __len__(self):
        return len(self.coco.getImgIds())

    def _load_coco_annotations(self):
        return [self.load_anno_from_ids(_ids) for _ids in self.ids]

    def load_anno_from_ids(self, id_):
        im_ann = self.coco.loadImgs(id_)[0]
        width = im_ann["width"]
        height = im_ann["height"]
        anno_ids = self.coco.getAnnIds(imgIds=[int(id_)], iscrowd=False)
        annotations = self.coco.loadAnns(anno_ids)
        objs = []
        for obj in annotations:
            x1 = np.max((0, obj["bbox"][0]))
            y1 = np.max((0, obj["bbox"][1]))
            x2 = np.min((width, x1 + np.max((0, obj["bbox"][2]))))
            y2 = np.min((height, y1 + np.max((0, obj["bbox"][3]))))
            if obj["area"] > 0 and x2 >= x1 and y2 >= y1:
                obj["clean_bbox"] = [x1, y1, x2, y2]
                objs.append(obj)

        num_objs = len(objs)

        res = np.zeros((num_objs, 5))
        for ix, obj in enumerate(objs):
            cls = self.class_ids.index(obj["category_id"])
            res[ix, 0:4] = obj["clean_bbox"]
            res[ix, 4] = cls

        r = min(self.img_size[0] / height, self.img_size[1] / width)
        res[:, :4] *= r

        img_info = (height, width)
        resized_info = (int(height * r), int(width * r))

        file_name = (
            im_ann["file_name"]
            if "file_name" in im_ann
            else "{:012}".format(id_) + ".jpg"
        )

        return (res, img_info, resized_info, file_name)

    def load_anno(self, index):
        return self.annotations[index][0]

    def load_resized_img(self, index):
        img = self.load_image(index)
        r = min(self.img_size[0] / img.shape[0], self.img_size[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        return resized_img

    def load_image(self, index):
        # print(index.item())
        # image_info = self.coco.loadImgs(index.item())
        file_name = self.annotations[index][3].split('/')
        # print(self.event_file)
        img_file = os.path.join(self.data_dir,file_name[0],'images/left/rectified',file_name[-1].replace('.npz','.png'))
        event_file = os.path.join(self.event_dir,file_name[0], 'h5', file_name[-1].replace('.npz','.h5'))
        voxel_file = os.path.join(self.voxel_dir,file_name[0], 'h5', file_name[-1].replace('.npz','.npy'))
        # print(img_file)
        img_rgb = cv2.imread(img_file)
        # print(event_file)
        xs = np.zeros((0))
        ys = np.zeros((0))
        ts = np.zeros((0))
        ps = np.zeros((0))
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
       
        img_rgb = cv2.remap(img_rgb, mapx, mapy, cv2.INTER_LINEAR)
        
        event_file = os.path.join(event_file)
        event_h5_file = h5py.File(event_file, 'r')
        xs = np.array(event_h5_file['events']['x'])
        ys = np.array(event_h5_file['events']['y'])
        ps = np.array(event_h5_file['events']['p'])
        ts = np.array(event_h5_file['events']['t'])
        event_h5_file.close()
        xs, ys, ts, ps = event_formatting(xs, ys, ts, ps)
        num = int(file_name[-1].split('.')[0])
        #读取更多的事件
        if(num>1):
            new_filename = "{:06d}.h5".format(num-1)
            event_file1=os.path.join(self.event_dir,file_name[0], 'h5',new_filename)
            # print(event_file1)
            event_file1 = os.path.join(event_file1)
            event_h5_file1 = h5py.File(event_file1, 'r')
            x1 = np.array(event_h5_file1['events']['x'])
            y1= np.array(event_h5_file1['events']['y'])
            p1 = np.array(event_h5_file1['events']['p'])
            t1 = np.array(event_h5_file1['events']['t'])
            x1, y1, t1, p1 = event_formatting(x1, y1, t1, p1)
            xs = torch.cat((x1,xs),dim=0)
            ys = torch.cat((y1,ys),dim=0)
            ps = torch.cat((p1,ps),dim=0)
            ts = torch.cat((t1,ts),dim=0)
            event_h5_file1.close()

        ts = (ts - ts[0]) / (ts[-1] - ts[0])
        inp_list = create_list_encoding(xs, ys, ts, ps)
        width = 640
        height = 480
        
        event_rate = torch.bincount(inp_list[:, 2].long() + inp_list[:, 1].long() * width, minlength=height * width)
        event_rate = event_rate.float()
   
        alpha = 2  # 调整超参数

        max_event_rate = torch.max(event_rate)
        # 设置一个事件率阈值
        # 初始化 time_window，确保和 event_rate 形状相同
        time_window = torch.ones_like(event_rate)

        max_event_rate = torch.max(event_rate)

        # 针对 event_rate > 0.8 * max_event_rate 和 event_rate < 5 进行赋值
        # mask_high = event_rate > 0.8 * max_event_rate
        mask_low = event_rate < 20

        # time_window[mask_high] = 1 - event_rate[mask_high] / max_event_rate
        time_window[mask_low] = torch.log(1 + (max_event_rate - event_rate[mask_low]) ** alpha) / torch.log(1 + max_event_rate ** alpha)
        # print(time_window[mask_low])

        # 对其余像素点的时间窗口设置为 0.5
        # mask_middle = (event_rate <= 0.8 * max_event_rate) & (event_rate >= 5)
        mask_middle = event_rate >= 20
        time_window[mask_middle] = 0.5

        # 防止时间窗口过小
        time_window[time_window < 0.01] = 0.01

        # 根据事件发生率更新时间窗口
        t_pos = inp_list.clone()
        t_pos[:, 0] = time_window[t_pos[:, 2].long() + t_pos[:, 1].long() * width]

        # 过滤事件
        event = inp_list[inp_list[:, 0] >= 1 - t_pos[:, 0]]
        if ts.shape[0] < 200000:
            event = inp_list
        event[:, 0] = (event[:, 0] - 0.5) * 2

        # 将小于零的值置为零
        event[:, 0] = torch.clamp(event[:, 0], min=0)

        ps = event[:,3]
        inp_pol_mask = create_polarity_mask(ps)
        # print(inp_list)
        # print(inp_pol_mask.shape)
        # print(inp_list.shape)

        
        # event_file = os.path.join(self.data_dir, file_name[0], 'iwe', file_name[-1].replace('.npz','.png'))
        
        # image = np.load(img_file)
        voxel = torch.from_numpy(np.load(voxel_file)).permute(2,0,1)
        # print(image)

        # print(image.shape)

        store=os.path.join('/root/data1/dataset/DSEC/iwe_2/',file_name[0],file_name[-1].replace('.npz','.png'))
        output = {}
        output["inp_voxel"] = voxel
        output["inp_list"] = event
        output["inp_pol_mask"] = inp_pol_mask
        output["mapping"] = mapping
        output["image"] = img_rgb
        output["name"]=store
        # output["iwe"] = torch.zeros((480, 640), dtype=torch.float32)
    
        return output

    @cache_read_img(use_cache=True)
    def read_img(self, index):
        return self.load_image(index)

    def pull_item(self, index):
        id_ = self.ids[index]
        label, origin_image_size, _, _ = self.annotations[index]
        img = self.read_img(index)

        return img, copy.deepcopy(label), origin_image_size, np.array([id_])

    @CacheDataset.mosaic_getitem
    def __getitem__(self, index):
        """
        One image / label pair for the given index is picked up and pre-processed.

        Args:
            index (int): data index

        Returns:
            img (numpy.ndarray): pre-processed image
            padded_labels (torch.Tensor): pre-processed label data.
                The shape is :math:`[max_labels, 5]`.
                each label consists of [class, xc, yc, w, h]:
                    class (float): class index.
                    xc, yc (float) : center of bbox whose values range from 0 to 1.
                    w, h (float) : size of bbox whose values range from 0 to 1.
            info_img : tuple of h, w.
                h, w (int): original shape of the image
            img_id (int): same as the input index. Used for evaluation.
        """
        img, target, img_info, img_id = self.pull_item(index)

        # if self.preproc is not None:
        #     img, target = self.preproc(img, target, self.input_dim)
        return img, target, img_info, img_id
