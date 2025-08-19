#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import torch
class CustomDict:
    def __init__(self, data_dict):
        self.data_dict = data_dict

    def record_stream(self, stream):
        for value in self.data_dict.values():
            if hasattr(value, "record_stream"):
                value.record_stream(stream)

    def __getitem__(self, key):
        return self.data_dict[key]

    def __setitem__(self, key, value):
        self.data_dict[key] = value

    def __repr__(self):
        return repr(self.data_dict)

class DataPrefetcher:
    """
    DataPrefetcher is inspired by code of following file:
    https://github.com/NVIDIA/apex/blob/master/examples/imagenet/main_amp.py
    It could speedup your pytorch dataloader. For more information, please check
    https://github.com/NVIDIA/apex/issues/304#issuecomment-493562789.
    """

    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.input_cuda = self._input_cuda_for_image
        self.record_stream = DataPrefetcher._record_stream_for_image
        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_target, _, _ = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return

        with torch.cuda.stream(self.stream):
            self.input_cuda()
            self.next_target = self.next_target.cuda(non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        if input is not None:
            self.record_stream(input)
        if target is not None:
            target.record_stream(torch.cuda.current_stream())
        self.preload()
        return input, target

    def _input_cuda_for_image(self):
        # self.next_input = self.next_input.cuda(non_blocking=True)
        for key in self.next_input:
            self.next_input[key] = self.next_input[key].cuda(non_blocking=True)

    @staticmethod
    def _record_stream_for_image(input):
        # input.record_stream(torch.cuda.current_stream())
        custom_input = CustomDict(input)
        custom_input.record_stream(torch.cuda.current_stream())
