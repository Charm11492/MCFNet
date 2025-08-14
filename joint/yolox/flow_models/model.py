"""
Adapted from UZH-RPG https://github.com/uzh-rpg/rpg_e2vid
"""

import copy

import numpy as np
import torch

from .base import BaseModel
from .model_util import copy_states, CropParameters
from .submodules import ResidualBlock, ConvGRU, ConvLayer
from .unet import UNetRecurrent, MultiResUNet


class EVFlowNet(BaseModel):
    """
    EV-FlowNet architecture for (dense/sparse) optical flow estimation from event-data.
    "EV-FlowNet: Self-Supervised Optical Flow for Event-based Cameras", Zhu et al. 2018.
    """

    def __init__(self, unet_kwargs, num_bins):
        super().__init__()
        self.crop = None
        # self.mask = unet_kwargs["mask_output"]
        EVFlowNet_kwargs = {
            "base_num_channels": unet_kwargs["base_num_channels"],
            "num_encoders": 4,
            "num_residual_blocks": 2,
            "num_output_channels": 2,
            "skip_type": "concat",
            "norm": None,
            "num_bins": num_bins,
            "use_upsample_conv": True,
            "kernel_size": unet_kwargs["kernel_size"],
            "channel_multiplier": 2,
            "final_activation": "tanh",
        }
        self.num_encoders = EVFlowNet_kwargs["num_encoders"]
        unet_kwargs.update(EVFlowNet_kwargs)
        unet_kwargs.pop("name", None)
        unet_kwargs.pop("eval", None)
        unet_kwargs.pop("encoding", None)  # TODO: remove
        unet_kwargs.pop("mask_output", None)
        unet_kwargs.pop("mask_smoothing", None)  # TODO: remove
        if "flow_scaling" in unet_kwargs.keys():
            unet_kwargs.pop("flow_scaling", None)

        self.multires_unet = MultiResUNet(unet_kwargs)

    def reset_states(self):
        pass

    def init_cropping(self, width, height, safety_margin=0):
        self.crop = CropParameters(width, height, self.num_encoders, safety_margin)

    def forward(self, inp_voxel):
        """
        :param inp_voxel: N x num_bins x H x W
        :return: output dict with list of [N x 2 X H X W] (x, y) displacement within event_tensor.
        """

        # pad input
        x = inp_voxel
        if self.crop is not None:
            x = self.crop.pad(x)

        # forward pass
        multires_flow = self.multires_unet.forward(x)

        # upsample flow estimates to the original input resolution
        flow_list = []
        for flow in multires_flow:
            flow_list.append(
                torch.nn.functional.interpolate(
                    flow,
                    scale_factor=(
                        multires_flow[-1].shape[2] / flow.shape[2],
                        multires_flow[-1].shape[3] / flow.shape[3],
                    ),
                )
            )

        # crop output
        if self.crop is not None:
            for i, flow in enumerate(flow_list):
                flow_list[i] = flow[:, :, self.crop.iy0 : self.crop.iy1, self.crop.ix0 : self.crop.ix1]
                flow_list[i] = flow_list[i].contiguous()

        # # mask flow
        # if self.mask:
        #     mask = torch.sum(inp_cnt, dim=1, keepdim=True)
        #     mask[mask > 0] = 1
        #     for i, flow in enumerate(flow_list):
        #         flow_list[i] = flow * mask

        return {"flow": flow_list}
