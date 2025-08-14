import torch
import numpy as np
import time
import os
import csv
import cv2
import argparse
# from retinanet import model
# from retinanet.dataloader import CSVDataset_event, collater, Resizer, AspectRatioBasedSampler, \
#     Augmenter, \
#     Normalizer
from torchvision import transforms
from torch.utils.data import DataLoader
import skimage.transform
import hdf5plugin
from torch_scatter import scatter_max, scatter_min
import h5py
from concurrent.futures import ProcessPoolExecutor

CLIP_COUNT = False
CLIP_COUNT_RATE = 0.99
DISC_ALPHA = 3.0
EXP_TAU = 0.3
resolution = (int(640), int(480))
Kr1_1 = np.array([
    [1164.6238115833075, 0, 713.5791168212891],
    [0, 1164.6238115833075, 570.9349365234375],
    [0,0,1]
])

Kr0_1 = np.array([
    [569.7632987676102, 0, 335.0999870300293],
    [0, 569.7632987676102, 221.23667526245117],
    [0,0,1]
])
T10_1 = np.array([
    [0.9996874046885865, 0.009652146488870916, 0.023063585478994113, -0.04410263392688484],
    [-0.009722042371104245, 0.9999484753460813, 0.0029203673010648615, 0.0005281285423087664],
    [-0.023034209322743096, -0.0031436795631953228, 0.9997297347181744, -0.01229891454144492],
    [0,0,0,1]
])
R_rect1_1 = np.array([
    [0.9998572179847892, -0.013025778024398856, -0.010764420587133948],
    [0.013060715513432202, 0.9999096430275752, 0.003181743349841093],
    [0.01072200326407413, -0.0033218800890692088, 0.9999369998948329]
])
R_rect0_1 = np.array([
    [0.9999313912417018, -0.0023139054373197965, 0.011482972222461762],
    [0.002353841678837691, 0.9999912245858043, -0.003465570766066675],
    [-0.011474852451585301, 0.0034923620961737592, 0.9999280629966356]
])
Mr1_r0 = np.matmul(Kr1_1, R_rect1_1)
Mr1_r0 = np.matmul(Mr1_r0, T10_1[:3,:3])
Mr1_r0 = np.matmul(Mr1_r0, np.linalg.inv(R_rect0_1))
Mr1_r0 = np.matmul(Mr1_r0, np.linalg.inv(Kr0_1))

map_x_1 = np.zeros((480, 640), dtype=np.float32)
map_y_1 = np.zeros((480, 640), dtype=np.float32)
for u in range(640):
    for v in range(480):
        Pr0 = np.array([u,v,1]).transpose()
        Pr1 = np.matmul(Mr1_r0, Pr0)
        map_x_1[v, u] = Pr1[0]/Pr1[2]
        map_y_1[v, u] = Pr1[1]/Pr1[2]
# all interlaken and thun files
def interlaken_thun():
    # Kr1：camRect1:：camera_matrix
    # Kr0：camRect0:：camera_matrix
    # T10：extrinsics:：T_10
    # R_rect0：extrinsics:：R_rect0
    # R_rect1：extrinsics:：R_rect1

    # K_dist <- intrinsics[cam0][camera_matrix]3X3
    # dist_coeffs <- intrinsics[cam0][distortion_coeffs]5X1
    # R_rect0 <- extrinsics[R_rect0]
    # K_rect <- intrinsics[camRect0][camera_matrix]3X3
    K_dist = np.array([
        [555.6627242364661, 0, 342.5725306057865],
        [0, 555.8306341927942, 215.26831427862848],
        [0,0,1]
    ])
    dist_coeffs = np.array([[-0.09094341408134071], [0.18339771556281387], [-0.0006982341741678465], [0.00041396758898911876]])
    K_rect = np.array([
        [569.7632987676102, 0, 335.0999870300293],
        [0, 569.7632987676102, 221.23667526245117],
        [0,0,1]])
    map_x = map_x_1
    map_y = map_y_1
    return map_x, map_y, K_dist, dist_coeffs, R_rect0_1, K_rect

Kr1_2 = np.array([
    [1150.8249465165975, 0, 724.4121398925781],
    [0, 1150.8249465165975, 569.1058044433594],
    [0,0,1]
])
Kr0_2 = np.array([
    [583.3081203392971, 0, 336.83414459228516],
    [0, 583.3081203392971, 220.91131019592285],
    [0,0,1]
])
T10_2 = np.array([
    [0.9997112144904777, 0.00986845600843356, 0.02191121169595574, -0.04410796144531285],
    [-0.00996970822323083, 0.9999401008187403, 0.00451660188111385, 0.000933594786315412],
    [-0.02186532734534335, -0.0047337459393644215, 0.9997497182342509, -0.01216624740013352],
    [0,0,0,1]
])
R_rect1_2 = np.array([
    [0.9998378277031966, -0.01324800441141771, -0.01219871603357244],
    [0.013290206303291856, 0.9999059517045085, 0.0033849907410636757],
    [0.012152724392852067, -0.0035465652420619918, 0.9999198633714681]
])
R_rect0_2 = np.array([
    [0.9999534546309575, -0.0030745983972593127, 0.009145240090297576],
    [0.003087094888434143, 0.9999943200819906, -0.0013526451518654562],
    [-0.009141029305467696, 0.001380814416433941, 0.9999572665543184]
])
Mr1_r0 = np.matmul(Kr1_2, R_rect1_2)
Mr1_r0 = np.matmul(Mr1_r0, T10_2[:3,:3])
Mr1_r0 = np.matmul(Mr1_r0, np.linalg.inv(R_rect0_2))
Mr1_r0 = np.matmul(Mr1_r0, np.linalg.inv(Kr0_2))

map_x_2 = np.zeros((480, 640), dtype=np.float32)
map_y_2 = np.zeros((480, 640), dtype=np.float32)
for u in range(640):
    for v in range(480):
        Pr0 = np.array([u,v,1]).transpose()
        Pr1 = np.matmul(Mr1_r0, Pr0)
        map_x_2[v, u] = Pr1[0]/Pr1[2]
        map_y_2[v, u] = Pr1[1]/Pr1[2] 
# zurich_city 00 01 02 03 09 10 12 14 
def zurich_city_00_01_02_03_09_10_12_14():
    # Kr1：intrinsics:：camRect1:：camera_matrix
    # Kr0：intrinsics:：camRect0:：camera_matrix
    # T10：extrinsics:：T_10
    # R_rect0：extrinsics:：R_rect0
    # R_rect1：extrinsics:：R_rect1

    # K_dist <- intrinsics[cam0][camera_matrix]3X3
    # dist_coeffs <- intrinsics[cam0][distortion_coeffs]5X1
    # R_rect0 <- extrinsics[R_rect0]
    # K_rect <- intrinsics[camRect0][camera_matrix]3X3
    K_dist = np.array([
        [556.7176612320709, 0, 342.4201113309635],
        [0, 556.5737848320229, 215.1085137623697],
        [0,0,1]
    ])
    dist_coeffs = np.array([[-0.09798194451582616], 
                            [0.2097934453326764], 
                            [-0.0003578417123372964], 
                            [6.716111923650996e-05]])
    K_rect = np.array([
        [583.3081203392971, 0, 336.83414459228516],
        [0, 583.3081203392971, 220.91131019592285],
        [0,0,1]])
    map_x = map_x_2
    map_y = map_y_2
    return map_x, map_y, K_dist, dist_coeffs, R_rect0_2, K_rect


Kr1_3 = np.array([
        [1150.8943600390282, 0, 723.4334411621094],
        [0, 1150.8943600390282, 572.102180480957],
        [0,0,1]
    ])
Kr0_3 = np.array([
    [569.2873535700672, 0, 336.2678413391113],
    [0, 569.2873535700672, 222.2889060974121],
    [0,0,1]
])
T10_3= np.array([
    [0.9997329831508507, 0.00994674446197701, 0.020857245142004693, -0.043722240320426424],
    [-0.01003579267550241, 0.999940949009329, 0.004169095789442527, 0.0010155694745410755],
    [-0.020814544570561252, -0.004377301558648307, 0.9997737713930034, -0.013372668558381158],
    [0,0,0,1]
])
R_rect1_3 = np.array([
    [0.9998858610925897, -0.013510711178262034, -0.006762061119800281],
    [0.013535205789223095, 0.9999019509726164, 0.0035897974036225495],
    [0.00671289739037555, -0.0036809135568848755, 0.9999706935125713]
])
R_rect0_3 = np.array([
    [0.9998660626332526, -0.0031936428516894507, 0.01605171142316844],
    [0.00322963955629375, 0.9999923268645124, -0.002217124361550897],
    [-0.016044507552843316, 0.0022686686479106185, 0.999868704840767]
])
Mr1_r0 = np.matmul(Kr1_3, R_rect1_3)
Mr1_r0 = np.matmul(Mr1_r0, T10_3[:3,:3])
Mr1_r0 = np.matmul(Mr1_r0, np.linalg.inv(R_rect0_3))
Mr1_r0 = np.matmul(Mr1_r0, np.linalg.inv(Kr0_3))

map_x_3 = np.zeros((480, 640), dtype=np.float32)
map_y_3 = np.zeros((480, 640), dtype=np.float32)
for u in range(640):
    for v in range(480):
        Pr0 = np.array([u,v,1]).transpose()
        Pr1 = np.matmul(Mr1_r0, Pr0)
        map_x_3[v, u] = Pr1[0]/Pr1[2]
        map_y_3[v, u] = Pr1[1]/Pr1[2]
# zurich_city 04 05 11
def zurich_city_04_05_11():
    # Kr1：intrinsics:：camRect1:：camera_matrix
    # Kr0：intrinsics:：camRect0:：camera_matrix
    # T10：extrinsics:：T_10
    # R_rect0：extrinsics:：R_rect0
    # R_rect1：extrinsics:：R_rect1
    
    # K_dist <- intrinsics[cam0][camera_matrix]3X3
    # dist_coeffs <- intrinsics[cam0][distortion_coeffs]5X1
    # R_rect0 <- extrinsics[R_rect0]
    # K_rect <- intrinsics[camRect0][camera_matrix]3X3
    K_dist = np.array([
        [553.4686750102932, 0, 346.65339162053317],
        [0, 553.3994078799127, 216.52092103243012],
        [0,0,1]
    ])
    dist_coeffs = np.array([[-0.09356476362537607], 
                            [0.19445779814646236], 
                            [7.642434980998821e-05], 
                            [0.0019563864604273664]])
    K_rect = np.array([
        [569.2873535700672, 0, 336.2678413391113],
        [0, 569.2873535700672, 222.2889060974121],
        [0,0,1]])
    map_x = map_x_3
    map_y = map_y_3
    return map_x, map_y, K_dist, dist_coeffs, R_rect0_3, K_rect



Kr1_4 = np.array([
        [1148.2313838177965, 0, 726.4117584228516],
        [0, 1148.2313838177965, 568.2191543579102],
        [0,0,1]
    ])
Kr0_4 = np.array([
    [575.0645811377547, 0, 334.9762382507324],
    [0, 575.0645811377547, 221.3972873687744],
    [0,0,1]
])
T10_4 = np.array([
    [0.9996906180243795, 0.00977982770444263, 0.022869700568754786, -0.04410614563244475],
    [-0.00989243777798553, 0.9999394708630998, 0.004816044521365818, 0.0009581031126012151],
    [-0.022821216199882324, -0.005040791613874691, 0.9997268539511494, -0.01301945052983738],
    [0,0,0,1]
])
R_rect1_4 = np.array([
    [0.9997825616785434, -0.014402819773887509, -0.015079394750792686],
    [0.01448681798146471, 0.9998800652632013, 0.005476056430346526],
    [0.014998715553714228, -0.0056933181728534964, 0.9998713040486368]
])
R_rect0_4 = np.array([
    [0.9999510034094157, -0.003940999304622189, 0.009080710599050027],
    [0.003946173482357525, 0.9999920615005341, -0.0005519517723912113],
    [-0.009078463270282605, 0.000587758788003176, 0.9999586171658591]
])
Mr1_r0 = np.matmul(Kr1_4, R_rect1_4)
Mr1_r0 = np.matmul(Mr1_r0, T10_4[:3,:3])
Mr1_r0 = np.matmul(Mr1_r0, np.linalg.inv(R_rect0_4))
Mr1_r0 = np.matmul(Mr1_r0, np.linalg.inv(Kr0_4))

map_x_4 = np.zeros((480, 640), dtype=np.float32)
map_y_4 = np.zeros((480, 640), dtype=np.float32)
for u in range(640):
    for v in range(480):
        Pr0 = np.array([u,v,1]).transpose()
        Pr1 = np.matmul(Mr1_r0, Pr0)
        map_x_4[v, u] = Pr1[0]/Pr1[2]
        map_y_4[v, u] = Pr1[1]/Pr1[2]
# zurich_city 06 07 13
def zurich_city_06_07_13():
    # Kr1：intrinsics:：camRect1:：camera_matrix
    # Kr0：intrinsics:：camRect0:：camera_matrix
    # T10：extrinsics:：T_10
    # R_rect0：extrinsics:：R_rect0
    # R_rect1：extrinsics:：R_rect1
    
    # K_dist <- intrinsics[cam0][camera_matrix]3X3
    # dist_coeffs <- intrinsics[cam0][distortion_coeffs]5X1
    # R_rect0 <- extrinsics[R_rect0]
    # K_rect <- intrinsics[camRect0][camera_matrix]3X3
    K_dist = np.array([
        [554.1362963953508, 0, 341.32299310026224],
        [0, 554.2132539175158, 215.63800729794482],
        [0,0,1]
    ])
    dist_coeffs = np.array([[-0.0952796190432605], 
                            [0.196301204026214], 
                            [-0.0005728102553113103], 
                            [-0.00020630258342443618]])
    K_rect = np.array([
        [575.0645811377547, 0, 334.9762382507324],
        [0, 575.0645811377547, 221.3972873687744],
        [0,0,1]
    ])
    map_x = map_x_4
    map_y = map_y_4
    return map_x, map_y, K_dist, dist_coeffs, R_rect0_4, K_rect



Kr1_5 = np.array([
    [1148.9330037048228, 0, 726.3772430419922],
    [0, 1148.9330037048228, 569.4966430664062],
    [0,0,1]
])
Kr0_5 = np.array([
    [576.0330202256714, 0, 335.0866508483887],
    [0, 576.0330202256714, 221.45818328857422],
    [0,0,1]
])
T10_5 = np.array([
    [0.9997004955089873, 0.009773558241868315, 0.022436506822022646, -0.04385159791981603],
    [-0.009892556031143009, 0.9999375524130001, 0.005198904640979429, 0.0008799590778068782],
    [-0.02238429391900841, -0.0054193019465711, 0.9997347520978532, -0.013144020303621641],
    [0,0,0,1]
])
R_rect1_5 = np.array([
    [0.9997759246688632, -0.014305886465989391, -0.01560263006488639],
    [0.014389814904230091, 0.9998825195614897, 0.005280180147431992],
    [0.015525259403355326, -0.005503515947969887, 0.9998643296131076]
])
R_rect0_5 = np.array([
    [0.9999707562494566, -0.00405289038564792, 0.00648542407337759],
    [0.004062774320559291, 0.9999906044591649, -0.0015115747465131528],
    [-0.006479236892553544, 0.001537879356781588, 0.9999778269623653]
])
Mr1_r0 = np.matmul(Kr1_5, R_rect1_5)
Mr1_r0 = np.matmul(Mr1_r0, T10_5[:3,:3])
Mr1_r0 = np.matmul(Mr1_r0, np.linalg.inv(R_rect0_5))
Mr1_r0 = np.matmul(Mr1_r0, np.linalg.inv(Kr0_5))

map_x_5 = np.zeros((480, 640), dtype=np.float32)
map_y_5 = np.zeros((480, 640), dtype=np.float32)
for u in range(640):
    for v in range(480):
        Pr0 = np.array([u,v,1]).transpose()
        Pr1 = np.matmul(Mr1_r0, Pr0)
        map_x_5[v, u] = Pr1[0]/Pr1[2]
        map_y_5[v, u] = Pr1[1]/Pr1[2]
# zurich_city 08 15
def zurich_city_08_15():
    # Kr1：intrinsics:：camRect1:：camera_matrix
    # Kr0：intrinsics:：camRect0:：camera_matrix
    # T10：extrinsics:：T_10
    # R_rect0：extrinsics:：R_rect0
    # R_rect1：extrinsics:：R_rect1

    # K_dist <- intrinsics[cam0][camera_matrix]3X3
    # dist_coeffs <- intrinsics[cam0][distortion_coeffs]5X1
    # R_rect0 <- extrinsics[R_rect0]
    # K_rect <- intrinsics[camRect0][camera_matrix]3X3
    K_dist = np.array([
        [554.8898093454824, 0, 339.9858572775444],
        [0, 554.9228438411409, 214.84582716740985],
        [0,0,1]
    ])
    dist_coeffs = np.array([[-0.09602689312501277], 
                            [0.2001766345015985], 
                            [-0.0008818303716875279], 
                            [-0.0012239075418665132]])
    K_rect = np.array([
        [576.0330202256714, 0, 335.0866508483887],
        [0, 576.0330202256714, 221.45818328857422],
        [0,0,1]
    ])
    map_x = map_x_5
    map_y = map_y_5
    return map_x, map_y, K_dist, dist_coeffs, R_rect0_5, K_rect
class EventRepresentation:
    def convert(self, x: torch.Tensor, y: torch.Tensor, pol: torch.Tensor, time: torch.Tensor):
        raise NotImplementedError


class VoxelGrid(EventRepresentation):
    def __init__(self, channels: int, height: int, width: int, normalize: bool):
        self.voxel_grid = torch.zeros((channels, height, width), dtype=torch.float, requires_grad=False)
        self.nb_channels = channels
        self.normalize = normalize
    def convert(self, x: torch.Tensor, y: torch.Tensor, pol: torch.Tensor, time: torch.Tensor):
        assert x.shape == y.shape == pol.shape == time.shape
        assert x.ndim == 1

        C, H, W = self.voxel_grid.shape
        with torch.no_grad():
            self.voxel_grid = self.voxel_grid.to(pol.device)
            voxel_grid = self.voxel_grid.clone()
            # 对timestamp归一化
            t_norm = time
            t_norm = (C - 1) * (t_norm-t_norm[0]) / (t_norm[-1]-t_norm[0])

            # x,y,t坐标取整
            x0 = x.int()
            y0 = y.int()
            t0 = t_norm.int()

            # value = 2*pol-1
            value=pol

            # 将事件插值到相邻的8个点当中
            for xlim in [x0,x0+1]:
                for ylim in [y0,y0+1]:
                    for tlim in [t0,t0+1]:

                        mask = (xlim < W) & (xlim >= 0) & (ylim < H) & (ylim >= 0) & (tlim >= 0) & (tlim < self.nb_channels)
                        # 计算插值
                        # 离插值点的距离越小，值越大
                        interp_weights = value * (1 - (xlim-x).abs()) * (1 - (ylim-y).abs()) * (1 - (tlim - t_norm).abs())

                        # 计算索引值
                        index = H * W * tlim.long() + \
                                W * ylim.long() + \
                                xlim.long()
                        # 按索引赋值，相加
                        voxel_grid.put_(index[mask], interp_weights[mask], accumulate=True)

            #进行归一化
            if self.normalize:
                mask = torch.nonzero(voxel_grid, as_tuple=True)
                if mask[0].size()[0] > 0:
                    mean = voxel_grid[mask].mean()
                    std = voxel_grid[mask].std()
                    if std > 0:
                        voxel_grid[mask] = (voxel_grid[mask] - mean) / std
                    else:
                        voxel_grid[mask] = voxel_grid[mask] - mean

        return voxel_grid

import torch


def VoxelMaxDelta(event,height,width,channels: int):
    voxel_grid = torch.zeros((channels, height, width), dtype=torch.float, requires_grad=False)
    nb_channels = channels
    C, H, W = voxel_grid.shape
    with torch.no_grad():
        voxel_grid = voxel_grid.to(voxel_grid.device)
        voxel_grid = voxel_grid.clone()

        # 对 timestamp 归一化
        event[:,2] = (C - 1) * (event[:,2] - event[0,2]) / (event[-1,2] - event[0,2])
        event[:,2]=event[:,2].float()
        
        for i in range(0,C):
            eventi = event[(event[:, 2] >= i) & (event[:, 2] < i + 1)]
            idx = eventi[:, 0].long() + eventi[:, 1].long() * W
            # 取每一个像素最新事件的时间戳
            time, max_indices = scatter_max(eventi[:, 2], idx, dim=-1, dim_size=H * W)
            maxtime=torch.max(time)
            # 初始化一个形状为 (H * W,) 的一维张量，用于存储极性信息
            
            polarity_tensor = torch.zeros(H * W, dtype=eventi.dtype, device=eventi.device)
            # 将极性信息按照最大时间戳的索引位置进行散列
            polarity_tensor=eventi[max_indices-1,3]
                # 将一维张量转为形状为 (H, W) 的张量
            time=time.view(H,W)
            polarity_matrix = polarity_tensor.view(H, W)
            delta_function = 1-(maxtime - time).abs()
            delta_function=delta_function.float()
            value = ((torch.mul(polarity_matrix,delta_function) + 1) / 2 )* 255
            value=value.float()
            value=value.unsqueeze(0)
            voxel_grid[i,:,:]=value
        voxel_grid=voxel_grid.float()
    return  voxel_grid   

def generate_agile_event_volume_cuda(events, shape, volume_bins=5):
    tick = time.time()
    H, W = shape
    start_time = events[0, 2]
    time_length = events[-1, 2] - events[0, 2]
    events[:,2] = (events[:,2] - start_time) / time_length
    events[:,2]=events[:,2].float()
    mask = events[:, 3] < 0
    events[mask, 3] = 0
    
    x, y, t, p = events.unbind(-1)

    x, y, p = x.long(), y.long(), p.long()

    t_star = (volume_bins * t.float())[:,None,None]

    channels = volume_bins

#     torch.arange(channels): 使用torch.arange创建一个从0到channels-1的张量，表示通道的索引。torch.stack([..., ...], dim=1): 将上述创建的两个张量在维度1上堆叠，形成一个形状为 (channels, 2) 的张量。这里的 dim=1 表示在第一个轴上进行堆叠。
# [None,:,:](1, channels, 2)。
# torch.stack([p,1 - p],dim=1): 将两个张量 p 和 1 - p 在维度1上堆叠，形成一个形状为 (n, 2) 的张量，其中 n 是事件数量。
    adder = torch.stack([torch.arange(channels),torch.arange(channels)],dim = 1).to(x.device)[None,:,:] + 1   #1, 2, 2
    adder = (1 - torch.abs(adder-t_star)) * torch.stack([p,1 - p],dim=1)[:,None,:]  #n, 2, 2
    
    adder = torch.where(adder>=0,adder,torch.zeros_like(adder)).view(adder.shape[0], channels * 2) #n, 4

    img = torch.zeros((H * W, volume_bins * 2)).float().to(x.device)
    img.index_add_(0, x + W * y, adder)
    img = img.view(H * W, volume_bins, 2)

    img = torch.sum(img, dim=-1)
    img = img/2

    # img_viewed = img.view((H, W, img.shape[1] * 2)).permute(2, 0, 1).contiguous()
    img_viewed = img.view((H, W, img.shape[1])).permute(2, 0, 1).contiguous()

    img_viewed = img_viewed / 5 * 255

    torch.cuda.synchronize()
    generate_volume_time = time.time() - tick

    img_viewed = np.where(img_viewed > 255, 255, img_viewed)
    img_viewed = img_viewed.astype(np.uint8)

    return img_viewed

def events_to_voxel_grid(x, y, p, t, device: str='cpu'):
        t = (t - t[0])
        t = (t/t[-1])
     
        pol = p
        height = 480
        width = 640
        num_bins = 5
        # Set event representation
        voxel_grid = VoxelGrid(num_bins, height, width, normalize=True)
        return voxel_grid.convert(
                x,
                y,
                pol,
                t)


#DiST
def reshape_then_acc_adj_sort(event_tensor):
    # Accumulate events to create a 2 * H * W image
    DISC_ALPHA = 3.0
    CLIP_COUNT_RATE = 0.99
    H = 480
    W = 640
    # event_tensor[event_tensor[:, 0] < 0] = 0 
    # event_tensor[event_tensor[:, 0] >= W] = W - 1 
    # event_tensor[event_tensor[:, 1] < 0] = 0 
    # event_tensor[event_tensor[:, 1] >= H] = H - 1

    pos = event_tensor[event_tensor[:, 3] > 0]
    neg = event_tensor[event_tensor[:, 3] < 0]
    # print('x:{}'.format(pos[:, 0]))
    # print(pos[:, 0])
    # print('y:{}'.format(pos[:, 1]))
    # print(pos[:, 1])
    # Get pos, neg counts
    pos_count = torch.bincount(pos[:, 0].long() + pos[:, 1].long() * W, minlength=H * W).reshape(H, W)
    pos_count = pos_count.float()

    neg_count = torch.bincount(neg[:, 0].long() + neg[:, 1].long() * W, minlength=H * W).reshape(H, W)
    neg_count = neg_count.float()

    # clip count
    pos_unique_count = torch.unique(pos_count, return_counts=True)[1]
    pos_sum_subset = torch.cumsum(pos_unique_count, dim=0)
    pos_th_clip = pos_sum_subset[pos_sum_subset < H * W * CLIP_COUNT_RATE].shape[0]
    pos_count[pos_count > pos_th_clip] = pos_th_clip

    neg_unique_count = torch.unique(neg_count, return_counts=True)[1]
    neg_sum_subset = torch.cumsum(neg_unique_count, dim=0)
    neg_th_clip = neg_sum_subset[neg_sum_subset < H * W * CLIP_COUNT_RATE].shape[0]
    neg_count[neg_count > neg_th_clip] = neg_th_clip

    start_time = event_tensor[0, 2]
    time_length = event_tensor[-1, 2] - event_tensor[0, 2]

    norm_pos_time = (pos[:, 2] - start_time) / time_length
    norm_neg_time = (neg[:, 2] - start_time) / time_length

    # Get pos, neg time
    pos_idx = pos[:, 0].long() + pos[:, 1].long() * W
    neg_idx = neg[:, 0].long() + neg[:, 1].long() * W
    pos_out, _ = scatter_max(norm_pos_time, pos_idx, dim=-1, dim_size=H * W)
    pos_min_out, _ = scatter_min(norm_pos_time, pos_idx, dim=-1, dim_size=H * W)
    pos_out = pos_out.reshape(H, W).float()
    pos_min_out = pos_min_out.reshape(H, W).float()
    neg_out, _ = scatter_max(norm_neg_time, neg_idx, dim=-1, dim_size=H * W)
    neg_min_out, _ = scatter_min(norm_neg_time, neg_idx, dim=-1, dim_size=H * W)
    neg_out = neg_out.reshape(H, W).float()
    neg_min_out = neg_min_out.reshape(H, W).float()

    pos_min_out[pos_count == 0] = 1.0
    neg_min_out[neg_count == 0] = 1.0

    # Get temporal discount
    pos_disc = torch.zeros_like(pos_count)
    neg_disc = torch.zeros_like(neg_count)

    patch_size = 5

    pos_neighbor_count = patch_size ** 2 * torch.nn.functional.avg_pool2d(pos_count.unsqueeze(0), patch_size, stride=1, padding=patch_size // 2)
    neg_neighbor_count = patch_size ** 2 * torch.nn.functional.avg_pool2d(neg_count.unsqueeze(0), patch_size, stride=1, padding=patch_size // 2)

    pos_disc = (torch.nn.functional.max_pool2d(pos_out.unsqueeze(0), patch_size, stride=1, padding=patch_size // 2) + 
        torch.nn.functional.max_pool2d(-pos_min_out.unsqueeze(0), patch_size, stride=1, padding=patch_size // 2)) / \
        (pos_neighbor_count)
    neg_disc = (torch.nn.functional.max_pool2d(neg_out.unsqueeze(0), patch_size, stride=1, padding=patch_size // 2) + 
        torch.nn.functional.max_pool2d(-neg_min_out.unsqueeze(0), patch_size, stride=1, padding=patch_size // 2)) / \
        (neg_neighbor_count)

    pos_out[pos_count > 0] = (pos_out[pos_count > 0] - DISC_ALPHA * pos_disc.squeeze()[pos_count > 0])
    pos_out[pos_out < 0] = 0
    pos_out[pos_neighbor_count.squeeze() == 1.0] = 0
    neg_out[neg_count > 0] = (neg_out[neg_count > 0] - DISC_ALPHA * neg_disc.squeeze()[neg_count > 0])
    neg_out[neg_out < 0] = 0
    neg_out[neg_neighbor_count.squeeze() == 1.0] = 0

    pos_out = pos_out.reshape(H * W)
    neg_out = neg_out.reshape(H * W)

    pos_val, pos_idx = torch.sort(pos_out)
    neg_val, neg_idx = torch.sort(neg_out)
    
    pos_unq, pos_cnt = torch.unique_consecutive(pos_val, return_counts=True)
    neg_unq, neg_cnt = torch.unique_consecutive(neg_val, return_counts=True)

    pos_sort = torch.zeros_like(pos_out)
    neg_sort = torch.zeros_like(neg_out)

    pos_sort[pos_idx] = torch.repeat_interleave(torch.arange(pos_unq.shape[0]), pos_cnt).float() / pos_unq.shape[0]
    neg_sort[neg_idx] = torch.repeat_interleave(torch.arange(neg_unq.shape[0]), neg_cnt).float() / neg_unq.shape[0]

    pos_sort = pos_sort.reshape(H, W)
    neg_sort = neg_sort.reshape(H, W)

    result = torch.stack([pos_sort, neg_sort], dim=2)
    # print(np.array(result).shape)
    result = result.permute(2, 0, 1)
    result = result.float()

    return result

def reshape_then_acc_exp(event_tensor):
    # Accumulate events to create a 2 * H * W image

    H = 480
    W = 640

    # 按极性进行划分
    pos = event_tensor[event_tensor[:, 3] > 0]
    neg = event_tensor[event_tensor[:, 3] < 0]
    # 取开始时间和时间片长度
    start_time = event_tensor[0, 2]
    time_length = event_tensor[-1, 2] - event_tensor[0, 2]

    # Get pos, neg time
    # 对时间进行归一化
    norm_pos_time = (pos[:, 2] - start_time) / time_length
    norm_neg_time = (neg[:, 2] - start_time) / time_length
    # 取事件的索引值
    pos_idx = pos[:, 0].long() + pos[:, 1].long() * W
    neg_idx = neg[:, 0].long() + neg[:, 1].long() * W
    # 取每一个像素最新事件的时间戳
    pos_out, _ = scatter_max(norm_pos_time, pos_idx, dim=-1, dim_size=H * W)
    pos_out = pos_out.reshape(H, W)
    # 指数衰减
    pos_out_exp = torch.exp(-(1 - pos_out) / EXP_TAU)
    neg_out, _ = scatter_max(norm_neg_time, neg_idx, dim=-1, dim_size=H * W)
    neg_out = neg_out.reshape(H, W)
    # neg_out_exp = torch.exp(-(1 - neg_out) / EXP_TAU)
    neg_out_exp = torch.exp(-(1 - neg_out) / EXP_TAU)

    # concate
    result = torch.stack([pos_out_exp, neg_out_exp], dim=2)

    result = result.permute(2, 0, 1)
    result = result.float()
    return result

def events_to_image1(inp_events):

    pos = inp_events[:, :, 0]
    neg = inp_events[:, :, 1]

    # Calculate the 99th and 1st percentiles using torch.quantile()
    pos_max = torch.quantile(pos, 0.99).to(inp_events.device)
    pos_min = torch.quantile(pos, 0.01).to(inp_events.device)
    neg_max = torch.quantile(neg, 0.99).to(inp_events.device)
    neg_min = torch.quantile(neg, 0.01).to(inp_events.device)

    max = pos_max if pos_max > neg_max else neg_max

    if pos_min != max:
        pos = (pos - pos_min) / (max - pos_min)
    if neg_min != max:
        neg = (neg - neg_min) / (max - neg_min)

    pos = torch.clamp(pos, 0, 1).to(inp_events.device)
    neg = torch.clamp(neg, 0, 1).to(inp_events.device)

    event_image = torch.ones((inp_events.shape[0], inp_events.shape[1], 3), device=inp_events.device)
       
    event_image *= 0
    mask_pos = pos > 0
    mask_neg = neg > 0
    mask_not_pos = pos == 0
    mask_not_neg = neg == 0

    event_image[:, :, 0][mask_pos] = 0
    event_image[:, :, 1][mask_pos] = pos[mask_pos]
    event_image[:, :, 2][mask_pos * mask_not_neg] = 0
    event_image[:, :, 2][mask_neg] = neg[mask_neg]
    event_image[:, :, 0][mask_neg] = 0
    event_image[:, :, 1][mask_neg * mask_not_pos] = 0
        
    return event_image

def events_to_image(xs, ys, ps, sensor_size=(180, 240)):
    """
    Accumulate events into an image.
    """

    device = xs.device
    img_size = list(sensor_size)
    img = torch.zeros(img_size).to(device)

    if xs.dtype is not torch.long:
        xs = xs.long().to(device)
    if ys.dtype is not torch.long:
        ys = ys.long().to(device)
    # print(ps)
    img.index_put_((ys, xs), ps, accumulate=True)
    
    return img

def events_to_voxel(xs, ys, ts, ps, num_bins=5, sensor_size=(480, 640)):
    """
    Generate a voxel grid from input events using temporal bilinear interpolation.
    """

    assert len(xs) == len(ys) and len(ys) == len(ts) and len(ts) == len(ps)

    voxel = []
    ts = ts * (num_bins - 1)
    zeros = torch.zeros(ts.size())
    for b_idx in range(num_bins):
        weights = torch.max(zeros, 1.0 - torch.abs(ts - b_idx))
        voxel_bin = events_to_image(xs, ys, ps * weights, sensor_size=sensor_size)
        voxel.append(voxel_bin)
    
    return torch.stack(voxel)

def load_classes(csv_reader):
    result = {}

    for line, row in enumerate(csv_reader):
        line += 1

        try:
            class_name, class_id = row
        except ValueError:
            raise(ValueError('line {}: format should be \'class_name,class_id\''.format(line)))
        class_id = int(class_id)

        if class_name in result:
            raise ValueError('line {}: duplicate class name: \'{}\''.format(line, class_name))
        result[class_name] = class_id
    return result


# Draws a caption above the box in an image
def draw_caption(image, box, caption,colour):
    b = np.array(box).astype(int)
    # cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
    # cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)
    cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, colour, 2)
    cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, colour, 1)


def Normalizer(sample):
    mean = np.array([[[0.485, 0.456, 0.406]]])
    std = np.array([[[0.229, 0.224, 0.225]]])

    return ((sample.astype(np.float32)-mean)/std)

def Resize(image, min_side=480, max_side=640):
    rows, cols, cns = image.shape

    smallest_side = min(rows, cols)

    # rescale the image so the smallest side is min_side
    scale = min_side / smallest_side

    # check if the largest side is now greater than max_side, which can happen
    # when images have a large aspect ratio
    largest_side = max(rows, cols)

    if largest_side * scale > max_side:
        scale = max_side / largest_side

    # resize the image with the computed scale
    image = skimage.transform.resize(image, (int(round(rows*scale)), int(round((cols*scale)))))
    # return torch.from_numpy(image)
    return image

# def Homographic_transformation(src):

#     dst = cv2.remap(src, map_x, map_y, cv2.INTER_LINEAR)
#     return dst
def visualizeVolume(volume):
    # 创建一个空图像用于显示结果，颜色通道为 3
    img = np.zeros((volume.shape[0], volume.shape[1], 3), dtype=np.uint8)
    
    # 初始颜色：红色 (H=0), 绿色 (H=60), 饱和度 (S=255)，亮度 (V=255)
    img[:,:,0] = 0    # 红色色调 (HSV中的色调值)
    img[:,:,1] = 255  # 设置饱和度通道为最大值
    img[:,:,2] = 255  # 设置亮度通道为最大值（V通道）
    
    # 提取所有 5 个通道数据
    c = volume  # volume 已经包含所有通道数据 (h, w, 5)

    img_buf = np.zeros_like(c[:,:,0])  # 用于存储图像的中间缓冲区

    # 对每个通道进行处理
    for i in range(5):
        # 根据索引设置色调值，交替使用红色和绿色色调
        img_0 = np.uint8(0 if i % 2 == 0 else 60)  # 交替设置红色 (0) 和 绿色 (60)
        tar2 = c[:,:,i].astype(np.uint8)  # 获取当前通道的值
        
        # 更新图像的颜色通道
        img[:,:,0] = np.where(c[:,:,i] > img_buf, img_0, img[:,:,0])  # 更新色调通道 (H通道)
        img[:,:,2] = np.where(c[:,:,i] > img_buf, tar2, img[:,:,2])  # 更新亮度通道 (V通道)
        
        img_buf = np.maximum(c[:,:,i], img_buf)  # 更新缓冲区，保留最大值

    # 将 img_buf 转换为 3 通道的图像
    img_buf_3channels = np.stack([img_buf, img_buf, img_buf], axis=-1)

    # 使用 OpenCV 将 HSV 转换为 BGR 格式
    img_s = cv2.cvtColor(img_buf_3channels, cv2.COLOR_HSV2BGR)

    return img_s
def reshape_then_acc(event_tensor):
    # Accumulate events to create a 2 * H * W image (pos_count and neg_count)
        
    H = 480
    W = 640
    
    # 分离正负事件
    pos = event_tensor[event_tensor[:, 3] > 0]
    neg = event_tensor[event_tensor[:, 3] < 0]
    
    # 计算正极性事件计数并归一化
    pos_count = torch.bincount(pos[:, 0].long() + pos[:, 1].long() * W, minlength=H * W).reshape(H, W)
    pos_max = pos_count.max().float()
    pos_count = pos_count
    
    # 计算负极性事件计数并归一化
    neg_count = torch.bincount(neg[:, 0].long() + neg[:, 1].long() * W, minlength=H * W).reshape(H, W)
    neg_max = neg_count.max().float()
    neg_count = neg_count 
    
    # 堆叠两个通道
    result = torch.stack([pos_count, neg_count], dim=0)  # 直接堆叠为(2, H, W)格式
    result = result.float()
    
    return result

def render(x: np.ndarray, y: np.ndarray, pol: np.ndarray, H: int, W: int) -> np.ndarray:

    assert x.size == y.size == pol.size
    assert H > 0
    assert W > 0
    img = np.full((H, W, 3), fill_value=255, dtype='uint8')
    mask = np.zeros((H, W), dtype='int32')
    pol = pol.astype('int')
    pol[pol == 0] = -1
    mask1 = (x >= 0) & (y >= 0) & (W > x) & (H > y)
    mask[y[mask1], x[mask1]] = pol[mask1]
    img[mask == 0] = [0, 0, 0]
    img[mask == -1] = [255, 0, 0]
    img[mask == 1] = [0, 0,255]
    image = np.transpose(img, (2, 0, 1))
    image = torch.tensor(image)
    return image

def detect_image(image_path):
    past_event=torch.zeros(1, 5)
    pos_rate=torch.zeros(640*480)
    neg_rate=torch.zeros(640*480)

    for img_name in os.listdir(image_path):

        event_file = os.path.join(image_path,img_name)
        # image = torch.from_numpy(np.load(event_file)['arr_0'])

        file = event_file.split('/')
        # print(event_file)
        # img_file = os.path.join(parser.root_img, file[-2], 'images/left/rectified', img_name.replace('.npz', '.png'))
        
        img_file = os.path.join(parser.root_img, file[-3],'images/left/rectified',file[-1].replace('.h5','.png'))
        # #print(img_file)
        # #img_rgb = cv2.imread(img_file)

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

        #img_rgb = cv2.remap(img_rgb, mapx, mapy, cv2.INTER_LINEAR)

        event_file = os.path.join(event_file)
        event_h5_file = h5py.File(event_file, 'r')
        xs = np.array(event_h5_file['events']['x'])
        ys = np.array(event_h5_file['events']['y'])
        # ps = np.array(event_h5_file['events']['p'])
        ps = np.array(event_h5_file['events']['p'])
        ts = np.array(event_h5_file['events']['t'])
        event_h5_file.close()
        
        # Account for zero polarity
        xs = torch.from_numpy(xs.astype(np.float32))
        ys = torch.from_numpy(ys.astype(np.float32))
        ts = torch.from_numpy(ts.astype(np.float32))
        ps = torch.from_numpy(ps.astype(np.float32)) * 2 - 1
        # print(torch.max(ts))
        # ts = (ts - ts[0]) / (ts[-1] - ts[0])
        event = torch.stack([xs, ys, ts, ps], dim=1)
        if event[:, 3].min() >= -0.5:
            event[:, 3][event[:, 3] <= 0.5] = -1
       
        # DiST
        # image = reshape_then_acc_adj_sort(event)
        # image = reshape_then_acc(event)
        # Dynamic_time
        # image,pos_rate,neg_rate = dynamic_time2(event,pos_rate,neg_rate)
        #Dynamic_DiST
        # image,pos_rate,neg_rate = dynamic_time2(event,pos_rate,neg_rate)
        #Dynamic_timestamp
        # image,pos_rate,neg_rate = dynamic_time3(event,pos_rate,neg_rate)
        
        #Dynamic_voxel
        # image,pos_rate,neg_rate=Dynamic_voxel(event,(480,640), pos_rate,neg_rate,5)
        # image,pos_rate,neg_rate=Dynamic_voxel(event,(480,640), pos_rate,neg_rate,5)
        
        #voxel
        # image = events_to_voxel_grid(xs, ys, ps, ts)
        # image = events_to_voxel(xs,ys,ts,ps)
        # image = image.to(torch.float32)

        #voxel_10channel
        # image=generate_agile_event_volume_cuda(event,(480,640), volume_bins=5)
        # print(image.shape)
        #voxelmax
        #image=VoxelMaxDelta(event,height=480,width=640,channels=5)
        #timestamp
        image = reshape_then_acc_exp(event)
        
        # image = render(xs,ys,ps,480,640)
        # print(image.shape)
        mapping = cv2.initUndistortRectifyMap(K_dist, dist_coeffs, R_rect0, K_rect, resolution, cv2.CV_32FC2)[0]

        # print(mapping.shape)
        ev_img = image
        # print(ev_img.shape)
        for i in range(image.shape[0]):
            # print(np.array(image[i,:,:]).shape)
            ev_img[i,:,:] = torch.from_numpy(cv2.remap(np.array(image[i,:,:].squeeze(0)), mapping, None, interpolation=cv2.INTER_CUBIC))
            # ev_img[i,:,:] = torch.from_numpy(cv2.remap(np.squeeze(image[i,:,:]), mapping, None, interpolation=cv2.INTER_CUBIC))

        # # image = torch.from_numpy(ev_img)
        # # print(torch.max(ev_img))
        # print(torch.max(ev_img))
        image = ev_img
        image = image.permute(1, 2, 0)

        # image[:, :, 0] = (image[:, :, 1] + image[:,:,2])/2  # 第一个通道
        # image[:, :, 1] = (image[:, :, 3] + image[:,:,4])/2  # 第一个通道

        # event[:, :, 1] = image[:, :, 3] # 第二个通道

        # # image = image.permute(1, 2, 0).to('cpu').numpy()
        # # event = visualizeVolume(image)
        # image = (image / torch.max(image))
        # event = np.zeros((480, 640, 3))  # 创建输出数组
        # # # img_show=img_show.permute(1, 2, 0).cpu().numpy().astype('uint8')
        # event[:, :, 1] = image[:, :, 0]  # 第一个通道
        # event[:, :, 2] = image[:, :, 1] # 第二个通道
      
        # event = torch.tensor(event, dtype=torch.float32)
    
        # event = np.empty((480, 640, 3))  # 创建输出数组
        # # # img_show=img_show.permute(1, 2, 0).cpu().numpy().astype('uint8')
        # event[:, :, 0] = image[:, :, 0]  # 第一个通道
        # event[:, :, 1] = (image[:, :, 1] + image[:,:,2])/2  # 第一个通道
        # event[:, :, 2] = (image[:, :, 3] + image[:,:,4])/2  # 第一个通道
        # # event = torch.tensor(event, dtype=torch.float32)
        

        # # image = image.permute(1, 2, 0).to('cuda', torch.float32)
        # print(image.shape)
        # event = image
        event = events_to_image1(image)

        # # # print(image)
        # # # img_show = torch.stack([image[0] + image[1], image[2] + image[3], image[4]], dim=0)
        # # # img_show[:2,:, :] /= 2

        # img_show = (img_show / torch.max(img_show)) * 255
        event = event.cpu().numpy()
        event = (event / np.max(event)) * 255
        
        # # img_show=img_show.permute(1, 2, 0).cpu().numpy().astype('uint8')
        event=event.astype('uint8')
        # img_show=img_show.cpu().numpy().astype('uint8')
 
        cv2.imwrite(os.path.join(save_path,os.path.splitext(img_name)[0] +'.png'),event)
        # print(torch.max(ev_img))
        # np.save(os.path.join(save_path,os.path.splitext(img_name)[0]),image)
        # print('Done')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Simple script for visualizing result of training.')

    parser.add_argument('--image_dir', default= '/root/data1/dataset/DSEC/train/events/interlaken_00_g/h5',help='Path to directory containing images')
    parser.add_argument('--csv_train', default='/root/data1/code/event-rgb-fusion-main/DSEC_detection_labels/train.csv',help='Path to file containing training annotations (see readme)')
    parser.add_argument('--csv_classes', default='/root/data1/code/event-rgb-fusion-main/DSEC_detection_labels/labels_filtered_map.csv', help='Path to file containing class list (see readme)')
    parser.add_argument('--root_img', default='/root/data1/dataset/DSEC/train/images', help='dir to toot rgb images in dsec format')
    # / media / storage / DSEC / train / transformed_images
    parser.add_argument('--root_event', default='/root/data1/dataset/DSEC/train/events', help='dir to toot event files in dsec directory structure')

    parser = parser.parse_args()

    # if parser.depth == 18:
    #     retinanet = model.resnet18(num_classes=3, pretrained=False)
    #     # retinanet = torch.load('csv_retinanet_1.pt')
    # elif parser.depth == 50:
    #     retinanet = model.resnet50(num_classes=8,fusion_model=parser.fusion, pretrained=False)
    # else:
    #     raise ValueError('Unsupported model depth, must be one of 18, 34, 50, 101, 152')

    # checkpoint = torch.load(parser.model_path)
    # retinanet.load_state_dict(checkpoint['model_state_dict'])
    # if torch.cuda.is_available():
    #     retinanet = retinanet.cuda()


    # retinanet.training = False
    # retinanet.eval()
    # filelist = [  'interlaken_00_a','interlaken_00_c','interlaken_00_d','interlaken_00_e','interlaken_00_f','interlaken_00_g','thun_00_a','thun_01_a','thun_01_b',
    # 'zurich_city_01_f','zurich_city_02_a','zurich_city_02_b','zurich_city_02_c','zurich_city_02_d','zurich_city_02_e','zurich_city_03_a','zurich_city_04_a',
    # 'zurich_city_04_c','zurich_city_05_a','zurich_city_05_b','zurich_city_06_a','zurich_city_07_a','zurich_city_08_a','zurich_city_09_c',
    # 'zurich_city_11_a','zurich_city_11_b','zurich_city_11_c','zurich_city_12_a','zurich_city_14_a','zurich_city_15_a','interlaken_00_b','interlaken_01_a',
    # 'zurich_city_01_a','zurich_city_01_b','zurich_city_09_a','zurich_city_09_d','zurich_city_10_a','zurich_city_10_b',"zurich_city_00_b",
    # 'zurich_city_01_c','zurich_city_01_d','zurich_city_01_e','zurich_city_04_e','zurich_city_00_a','zurich_city_04_b','zurich_city_04_d',
    # 'zurich_city_04_f','zurich_city_09_e','zurich_city_13_a','zurich_city_14_c', 'zurich_city_13_b','zurich_city_14_b', 'zurich_city_09_b'
    # ]
    # filelist = [  'interlaken_00_b'
    # ]
    # filelist = [  'zurich_city_01_c','zurich_city_01_d','zurich_city_01_e','zurich_city_04_e','zurich_city_00_a','zurich_city_04_c','zurich_city_00_b','zurich_city_04_b',
    # 'zurich_city_04_d','zurich_city_04_f','zurich_city_09_e','zurich_city_13_a','zurich_city_14_c','zurich_city_13_b','zurich_city_09_b','zurich_city_14_b',
    # ]
    filelist = ['zurich_city_09_a', 'zurich_city_09_b']
  
    for filename in filelist:
        print(filename)
        image_dir = '/root/data1/dataset/DSEC/train/events/' + filename + '/h5'
        # save_path = './visulize_fusion_result/2class_filter/' + filename + '/Timestamp_SA/'
        # image_dir = os.path.join(parser.root_event, filename, 'h5')
        #最小窗口0.01
        save_path = '/root/data1/dataset/DSEC/timestamp/'+filename 
        os.makedirs(save_path, exist_ok=True)
        detect_image(image_dir)
  
