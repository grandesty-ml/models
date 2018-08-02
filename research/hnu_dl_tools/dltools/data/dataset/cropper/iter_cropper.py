#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: liang kang
@contact: gangkanli1219@gmail.com
@time: 2018/4/4 21:24
@desc: 
"""
import itertools
from abc import abstractmethod

from dltools.data.dataset.cropper.rect_cropper import RectCropper


class IterCropper(RectCropper):
    """
    切割图像并可以迭代遍历
    """

    def __init__(self, image_path,
                 crop_size, stride,
                 logger=None):
        """

        Parameters
        ----------
        image_path: 图像路径
        logger: 日志对象
        crop_size: 子图尺寸
        stride: 切割步长
        """
        self._iter = None
        super(IterCropper, self).__init__(image_path,
                                          crop_size, stride,
                                          logger, False)
        self._generate_iter()

    def _generate_iter(self):
        """
        生成迭代器
        Returns
        -------

        """
        print(self._seat_x, self._seat_y)
        self._iter = itertools.product(self._seat_x, self._seat_y)

    @abstractmethod
    def _crop_image(self):
        """
        切割子图

        Returns
        -------

        """
        x, y = self._buf_data['x'], self._buf_data['y']
        xmax = x + self._size[1]
        ymax = y + self._size[0]
        sub_image = self.image[y:ymax, x:xmax, :]
        self._buf_data['iter_data'] = (y, x), sub_image

    def __iter__(self):
        """
        执行切割，遍历每一个位置，提取子图

        Returns
        -------

        """
        for x, y in self._iter:
                self._buf_data['x'] = x
                self._buf_data['y'] = y
                self._crop_image()
                data = self._buf_data['iter_data']
                self._buf_data.clear()
                yield data
