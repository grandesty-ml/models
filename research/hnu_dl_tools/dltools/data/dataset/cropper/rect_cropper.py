#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: liang kang
@contact: gangkanli1219@gmail.com
@time: 2018/4/4 17:23
@desc: 
"""
import itertools
from abc import abstractmethod

import numpy as np
from dltools.data.dataset.cropper.image_cropper import BaseCropper


class RectCropper(BaseCropper):
    """
    用于切割矩形内容的图像的基类

    切割过程通常为 从左到右，从上到下
    """

    def __init__(self, image_path, crop_size, stride,
                 logger=None, is_write=True):
        """

        Parameters
        ----------
        image_path: 图像路径
        logger: 日志对象
        crop_size: 子图尺寸
        stride: 切割步长
        is_write: 是否写入图像到文件
        """
        self._size = np.asarray(crop_size)
        self._stride = np.asarray(stride)
        self._seat_y = None
        self._seat_x = None
        self._is_write = is_write
        super(RectCropper, self).__init__(image_path, logger)

    @abstractmethod
    def _set_up(self):
        """
        基本配置，计算提取子图的每一个位置

        Returns
        -------

        """
        shape = np.asarray(self.image.shape[:2])
        self._size = np.minimum(self._size, shape)
        self._seat_y = [
            x for x in range(0, shape[0] - self._size[0], self._stride[0])]
        if len(self._seat_y) == 0:
            self._seat_y.append(0)
        if self._size[0] + self._seat_y[-1] < shape[0]:
            self._seat_y.append(shape[0] - self._size[0])
        self._seat_x = [
            x for x in range(0, shape[1] - self._size[1], self._stride[1])]
        if len(self._seat_x) == 0:
            self._seat_x.append(0)
        if self._size[1] + self._seat_x[-1] < shape[1]:
            self._seat_x.append(shape[1] - self._size[1])

    @abstractmethod
    def _update(self):
        """
        执行切割，遍历每一个位置，提取子图

        Returns
        -------

        """
        for x, y in itertools.product(self._seat_x, self._seat_y):
            self._buf_data['x'] = x
            self._buf_data['y'] = y
            self._crop_image()
            if self._is_write:
                self._write_image()
            self._buf_data.clear()
