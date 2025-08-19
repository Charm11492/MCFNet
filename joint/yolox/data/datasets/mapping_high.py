import torch
import numpy as np
import os
import csv
import cv2

CLIP_COUNT = False
CLIP_COUNT_RATE = 0.99
DISC_ALPHA = 3.0
EXP_TAU = 0.3
resolution = (int(1280), int(960))
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

map_x_1 = np.zeros((960, 1280), dtype=np.float32)
map_y_1 = np.zeros((960, 1280), dtype=np.float32)
for u in range(960):
    for v in range(1280):
        Pr0 = np.array([u,v,1]).transpose()
        Pr1 = np.matmul(Mr1_r0, Pr0)
        map_x_1[v, u] = Pr1[0]/Pr1[2]
        map_y_1[v, u] = Pr1[1]/Pr1[2]
# all interlaken and thun files
def interlaken_thun_high():
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

map_x_2 = np.zeros((960, 1280), dtype=np.float32)
map_y_2 = np.zeros((960, 1280), dtype=np.float32)
for u in range(1280):
    for v in range(960):
        Pr0 = np.array([u,v,1]).transpose()
        Pr1 = np.matmul(Mr1_r0, Pr0)
        map_x_2[v, u] = Pr1[0]/Pr1[2]
        map_y_2[v, u] = Pr1[1]/Pr1[2] 
# zurich_city 00 01 02 03 09 10 12 14 
def zurich_city_00_01_02_03_09_10_12_14_high():
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

map_x_3 = np.zeros((960, 1280), dtype=np.float32)
map_y_3 = np.zeros((960, 1280), dtype=np.float32)
for u in range(1280):
    for v in range(960):
        Pr0 = np.array([u,v,1]).transpose()
        Pr1 = np.matmul(Mr1_r0, Pr0)
        map_x_3[v, u] = Pr1[0]/Pr1[2]
        map_y_3[v, u] = Pr1[1]/Pr1[2]
# zurich_city 04 05 11
def zurich_city_04_05_11_high():
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

map_x_4 = np.zeros((960, 1280), dtype=np.float32)
map_y_4 = np.zeros((960, 1280), dtype=np.float32)
for u in range(1280):
    for v in range(960):
        Pr0 = np.array([u,v,1]).transpose()
        Pr1 = np.matmul(Mr1_r0, Pr0)
        map_x_4[v, u] = Pr1[0]/Pr1[2]
        map_y_4[v, u] = Pr1[1]/Pr1[2]
# zurich_city 06 07 13
def zurich_city_06_07_13_high():
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

map_x_5 = np.zeros((960, 1280), dtype=np.float32)
map_y_5 = np.zeros((960, 1280), dtype=np.float32)
for u in range(960):
    for v in range(1280):
        Pr0 = np.array([u,v,1]).transpose()
        Pr1 = np.matmul(Mr1_r0, Pr0)
        map_x_5[v, u] = Pr1[0]/Pr1[2]
        map_y_5[v, u] = Pr1[1]/Pr1[2]
# zurich_city 08 15
def zurich_city_08_15_high():
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