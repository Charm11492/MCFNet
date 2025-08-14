import torch
import numpy as np
import cv2
import os
import torch.nn.functional as F
def purge_unfeasible(x, res):
    """
    Purge unfeasible event locations by setting their interpolation weights to zero.
    :param x: location of motion compensated events
    :param res: resolution of the image space
    :return masked indices
    :return mask for interpolation weights
    """

    mask = torch.ones((x.shape[0], x.shape[1], 1)).to(x.device)
    mask_y = (x[:, :, 0:1] < 0) + (x[:, :, 0:1] >= res[0])
    mask_x = (x[:, :, 1:2] < 0) + (x[:, :, 1:2] >= res[1])
    mask[mask_y + mask_x] = 0
    return x * mask, mask


def get_interpolation(events, flow, tref, res, flow_scaling, round_idx=False):
    """
    Warp the input events according to the provided optical flow map and compute the bilinar interpolation
    (or rounding) weights to distribute the events to the closes (integer) locations in the image space.
    :param events: [batch_size x N x 4] input events (y, x, ts, p)
    :param flow: [batch_size x 2 x H x W] optical flow map
    :param tref: reference time toward which events are warped
    :param res: resolution of the image space
    :param flow_scaling: scalar that multiplies the optical flow map
    :param round_idx: whether or not to round the event locations instead of doing bilinear interp. (default = False)
    :return interpolated event indices
    :return interpolation weights
    """

    # event propagation
    warped_events = events[:, :, 1:3] + (tref - events[:, :, 0:1]) * flow * flow_scaling

    if round_idx:

        # no bilinear interpolation
        idx = torch.round(warped_events)
        weights = torch.ones(idx.shape).to(events.device)

    else:

        # get scattering indices
        top_y = torch.floor(warped_events[:, :, 0:1])
        bot_y = torch.floor(warped_events[:, :, 0:1] + 1)
        left_x = torch.floor(warped_events[:, :, 1:2])
        right_x = torch.floor(warped_events[:, :, 1:2] + 1)

        top_left = torch.cat([top_y, left_x], dim=2)
        top_right = torch.cat([top_y, right_x], dim=2)
        bottom_left = torch.cat([bot_y, left_x], dim=2)
        bottom_right = torch.cat([bot_y, right_x], dim=2)
        idx = torch.cat([top_left, top_right, bottom_left, bottom_right], dim=1)

        # get scattering interpolation weights
        warped_events = torch.cat([warped_events for i in range(4)], dim=1)
        zeros = torch.zeros(warped_events.shape).to(events.device)
        weights = torch.max(zeros, 1 - torch.abs(warped_events - idx))

    # purge unfeasible indices
    idx, mask = purge_unfeasible(idx, res)

    # make unfeasible weights zero
    weights = torch.prod(weights, dim=-1, keepdim=True) * mask  # bilinear interpolation

    # prepare indices
    idx[:, :, 0] *= res[1]  # torch.view is row-major
    idx = torch.sum(idx, dim=2, keepdim=True)

    return idx, weights


def interpolate(idx, weights, res, polarity_mask=None):
    """
    Create an image-like representation of the warped events.
    :param idx: [batch_size x N x 1] warped event locations
    :param weights: [batch_size x N x 1] interpolation weights for the warped events
    :param res: resolution of the image space
    :param polarity_mask: [batch_size x N x 2] polarity mask for the warped events (default = None)
    :return image of warped events
    """

    if polarity_mask is not None:
        weights = weights * polarity_mask
    iwe = torch.zeros((idx.shape[0], res[0] * res[1], 1)).to(idx.device)
    iwe = iwe.scatter_add_(1, idx.long(), weights)
    iwe = iwe.view((idx.shape[0], 1, res[0], res[1]))
    return iwe


def deblur_events(flow, event_list, res, flow_scaling=128, round_idx=True, polarity_mask=None):
    """
    Deblur the input events given an optical flow map.
    Event timestamp needs to be normalized between 0 and 1.
    :param flow: [batch_size x 2 x H x W] optical flow map
    :param events: [batch_size x N x 4] input events (y, x, ts, p)
    :param res: resolution of the image space
    :param flow_scaling: scalar that multiplies the optical flow map
    :param round_idx: whether or not to round the event locations instead of doing bilinear interp. (default = False)
    :param polarity_mask: [batch_size x N x 2] polarity mask for the warped events (default = None)
    :return iwe: [batch_size x 1 x H x W] image of warped events
    """

    # flow vector per input event
    event_list = event_list.to(torch.float32)
    flow = flow.to(torch.float32)

    flow_idx = event_list[:, :, 1:3].clone()
    
    flow_idx[:, :, 0] *= res[1]  # torch.view is row-major
    # print(flow_idx[:, :, 0])
    flow_idx = torch.sum(flow_idx, dim=2)
    # print(flow_idx)
    # get flow for every event in the list
    flow = flow.view(flow.shape[0], 2, -1).float()
    
    event_flowy = torch.gather(flow[:, 1, :], 1, flow_idx.long())  # vertical component
    event_flowx = torch.gather(flow[:, 0, :], 1, flow_idx.long())  # horizontal component
    event_flowy = event_flowy.view(event_flowy.shape[0], event_flowy.shape[1], 1)
    event_flowx = event_flowx.view(event_flowx.shape[0], event_flowx.shape[1], 1)
    event_flow = torch.cat([event_flowy, event_flowx], dim=2)

    # interpolate forward
    fw_idx, fw_weights = get_interpolation(event_list, event_flow, 1, res, flow_scaling, round_idx=round_idx)
    if not round_idx:
        polarity_mask = torch.cat([polarity_mask for i in range(4)], dim=1)
    
    # image of (forward) warped events
    iwe = interpolate(fw_idx.long(), fw_weights, res, polarity_mask=polarity_mask)

    return iwe

def events_to_image(inp_events, color_scheme="green_red"):
    """ Visualize the input events.
    :param inp_events: [H x W *2] per-pixel and per-polarity event count
    :param color_scheme: green_red/gray
    :return event_image: [H x W x 3] color-coded event image
    """
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

    if color_scheme == "gray":
        event_image *= 0.5
        pos *= 0.5
        neg *= -0.5
        event_image[:, :, 0] += pos + neg
        event_image[:, :, 1] += pos + neg
        event_image[:, :, 2] += pos + neg

    elif color_scheme == "green_red":
       
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


def compute_pol_iwe(flow, event_list, res, pos_mask, neg_mask, mapping,flow_scaling=128, round_idx=True):
    """
    Create a per-polarity image of warped events given an optical flow map.
    :param flow: [batch_size x 2 x H x W] optical flow map
    :param event_list: [batch_size x N x 4] input events (y, x, ts, p)
    :param res: resolution of the image space
    :param pos_mask: [batch_size x N x 1] polarity mask for positive events
    :param neg_mask: [batch_size x N x 1] polarity mask for negative events
    :param mapping: [batch_size x H x W x 2] the mapping coordinates
    :param flow_scaling: scalar that multiplies the optical flow map
    :param round_idx: whether or not to round the event locations instead of doing bilinear interp. (default = True)
    :return img: [batch_size x H x W x C] remapped images
    """
    # data_start_time = time.time()
    iwe_pos = deblur_events(flow, event_list, res, flow_scaling=flow_scaling, round_idx=round_idx, polarity_mask=pos_mask)
    iwe_neg = deblur_events(flow, event_list, res, flow_scaling=flow_scaling, round_idx=round_idx, polarity_mask=neg_mask)
    iwe = torch.cat([iwe_pos, iwe_neg], dim=1)

    iwe = iwe.permute(0, 2, 3, 1)  # 转换为 (batch_size, H, W, C)

    imgs = []
    for i in range(iwe.shape[0]):
        # print(torch.max(iwe[i, :]))
        image = events_to_image(iwe[i, :])  # 转换为三通道图像
        mapping_batch = mapping[i].to(torch.float32).to(image.device).unsqueeze(0)
        
        h, w, c = image.shape

        # 准备网格坐标，归一化到 [-1, 1] 范围
        mapping_batch[..., 0] = mapping_batch[..., 0] / (w - 1) * 2 - 1  # x 坐标归一化
        mapping_batch[..., 1] = mapping_batch[..., 1] / (h - 1) * 2 - 1  # y 坐标归一化

        # 将 image 转换为 (1, C, H, W) 格式
        image = image.permute(2, 0, 1).unsqueeze(0).float().to(image.device)

        # 使用 grid_sample 进行重映射
        resampled_image = F.grid_sample(image, mapping_batch, mode='bilinear', padding_mode='zeros', align_corners=False)

        # 将结果转换为 (H, W, C) 格式
        resampled_image = resampled_image.squeeze(0).permute(1, 2, 0)
        
        resampled_image = resampled_image * 255

        imgs.append(resampled_image)

    # 将所有图像堆叠成 (B, H, W, C) 格式
    img = torch.stack(imgs, dim=0)
    # data_end_time = time.time()
    # data_time = data_end_time - data_start_time
    # print(f"iwe time: {data_time:.4f}s")
    
    return img


def compute_pol_iwe2(flow, event_list, res, pos_mask, neg_mask, mapping, path, flow_scaling=128, round_idx=True):
    # def compute_pol_iwe(flow, event_list, res, pos_mask, neg_mask, mapping,flow_scaling=128, round_idx=True):
    """
    Create a per-polarity image of warped events given an optical flow map.
    :param flow: [batch_size x 2 x H x W] optical flow map
    :param event_list: [batch_size x N x 4] input events (y, x, ts, p)
    :param res: resolution of the image space
    :param pos_mask: [batch_size x N x 1] polarity mask for positive events
    :param neg_mask: [batch_size x N x 1] polarity mask for negative events
    :param mapping: [batch_size x H x W x 2] the mapping coordinates
    :param flow_scaling: scalar that multiplies the optical flow map
    :param round_idx: whether or not to round the event locations instead of doing bilinear interp. (default = True)
    :return img: [batch_size x H x W x C] remapped images
    """
    # data_start_time = time.time()
    iwe_pos = deblur_events(flow, event_list, res, flow_scaling=flow_scaling, round_idx=round_idx,
                            polarity_mask=pos_mask)
    iwe_neg = deblur_events(flow, event_list, res, flow_scaling=flow_scaling, round_idx=round_idx,
                            polarity_mask=neg_mask)
    iwe = torch.cat([iwe_pos, iwe_neg], dim=1)

    iwe = iwe.permute(0, 2, 3, 1)  # 转换为 (batch_size, H, W, C)

    imgs = []
    for i in range(iwe.shape[0]):
        # print(torch.max(iwe[i, :]))
        image = events_to_image(iwe[i, :])  # 转换为三通道图像
        mapping_batch = mapping[i].to(torch.float32).to(image.device).unsqueeze(0)

        h, w, c = image.shape

        # 准备网格坐标，归一化到 [-1, 1] 范围
        mapping_batch[..., 0] = mapping_batch[..., 0] / (w - 1) * 2 - 1  # x 坐标归一化
        mapping_batch[..., 1] = mapping_batch[..., 1] / (h - 1) * 2 - 1  # y 坐标归一化

        # 将 image 转换为 (1, C, H, W) 格式
        image = image.permute(2, 0, 1).unsqueeze(0).float().to(image.device)

        # 使用 grid_sample 进行重映射
        resampled_image = F.grid_sample(image, mapping_batch, mode='bilinear', padding_mode='zeros',
                                        align_corners=False)

        # 将结果转换为 (H, W, C) 格式
        resampled_image = resampled_image.squeeze(0).permute(1, 2, 0)

        resampled_image = resampled_image * 255

        resampled_image_np = resampled_image.detach().cpu().numpy().astype(np.uint8)
        path = os.path.join(path[0])
        name = path.split('/')
        filedir = os.path.dirname(path)
        # print(filedir)
        if not os.path.exists(filedir):
            os.makedirs(filedir)
        img_file = os.path.join(filedir, name[-1])
        # print(img_file)
        cv2.imwrite(img_file, resampled_image_np)

        imgs.append(resampled_image)

    # 将所有图像堆叠成 (B, H, W, C) 格式
    img = torch.stack(imgs, dim=0)
    # data_end_time = time.time()
    # data_time = data_end_time - data_start_time
    # print(f"iwe time: {data_time:.4f}s")

    return img