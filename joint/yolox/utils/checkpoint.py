#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.
import os
import shutil
from loguru import logger

import torch


# def load_ckpt(model, ckpt):
#     model_state_dict = model.state_dict()
#     load_dict = {}
#     for key_model, v in model_state_dict.items():
#         if key_model not in ckpt:
#             logger.warning(
#                 "{} is not in the ckpt. Please double check and see if this is desired.".format(
#                     key_model
#                 )
#             )
#             continue
#         v_ckpt = ckpt[key_model]
#         if v.shape != v_ckpt.shape:
#             logger.warning(
#                 "Shape of {} in checkpoint is {}, while shape of {} in model is {}.".format(
#                     key_model, v_ckpt.shape, key_model, v.shape
#                 )
#             )
#             continue
#         load_dict[key_model] = v_ckpt

#     model.load_state_dict(load_dict, strict=False)
#     return model

def load_ckpt(model, ckpt, model_dir, device):
    model_state_dict = model.state_dict()
    load_dict = {}
    model_loaded = torch.load(model_dir, map_location=device)
    if "state_dict" in model_loaded.keys():
        model_loaded = model_loaded["state_dict"]
    for key_model, v in model_state_dict.items():
        # Check if the parameter is in ckpt
        if key_model.startswith(('backbone.', 'head.')):
            if key_model in ckpt:
                v_ckpt = ckpt[key_model]
                if v.shape != v_ckpt.shape:
                    logger.warning(
                        "Shape of {} in checkpoint is {}, while shape of {} in model is {}.".format(
                            key_model, v_ckpt.shape, key_model, v.shape
                        )
                    )
                else:
                    load_dict[key_model] = v_ckpt
            else:
                logger.warning(
                    "{} is not in the ckpt. Please double check and see if this is desired.".format(
                        key_model
                    )
                )

        # Check if the parameter (with 'flow.' prefix removed) is in model_loaded
        if key_model.startswith('flow.'):
            key_model_no_flow = key_model[len('flow.'):]
            if key_model_no_flow in model_loaded:
                v_loaded = model_loaded[key_model_no_flow]
                if v.shape != v_loaded.shape:
                    logger.warning(
                        "Shape of {} in model_dir is {}, while shape of {} in model is {}.".format(
                            key_model, v_loaded.shape, key_model, v.shape
                        )
                    )
                else:
                    load_dict[key_model] = v_loaded
            else:
                logger.warning(
                    "{} is not in the model_dir. Please double check and see if this is desired.".format(
                        key_model
                    )
                )

    model.load_state_dict(load_dict, strict=False)
    return model

def save_checkpoint(state, is_best, save_dir, model_name=""):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    filename = os.path.join(save_dir, model_name + "_ckpt.pth")
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(save_dir, "best_ckpt.pth")
        shutil.copyfile(filename, best_filename)
