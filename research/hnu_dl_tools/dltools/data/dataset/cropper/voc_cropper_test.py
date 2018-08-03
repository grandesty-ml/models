#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: liang kang
@contact: gangkanli1219@gmail.com
@time: 2018/6/5 11:51
@desc: 
"""
import itertools
from abc import abstractmethod
from typing import Iterable

import numpy as np
import pytest
from dltools.data.dataset.cropper.voc_cropper import VOCImageCropper
from dltools.utils.log import get_console_logger


class _VOCImageCropper(VOCImageCropper):

    def __init__(self, image_path, crop_size, stride, logger):
        """

        Parameters
        ----------
        image_path: 图像路径
        crop_size: 子图尺寸
        stride: 切割步长
        logger
        """
        super(_VOCImageCropper, self).__init__(
            image_path, None, None, crop_size, stride,
            logger=logger)
        self.result = []

    @property
    def image_info(self):
        return self._image_info

    @image_info.setter
    def image_info(self, value):
        self._image_info = value

    @abstractmethod
    def _preprocess(self):
        pass

    @abstractmethod
    def _update(self):
        for x, y in itertools.product(self._seat_x, self._seat_y):
            self._buf_data['x'] = x
            self._buf_data['y'] = y
            self._crop_image()
            if self._is_write:
                self.result.append(self._buf_data['objects'])
            self._buf_data.clear()

    @abstractmethod
    def update(self):
        self._logger.info('cropping ...')
        self._update()
        return True


def test_rect_cropper():
    logger = get_console_logger('VOC')
    img = np.ones([20, 20, 3])
    image_info = {'shape': {'width': 20, 'height': 20},
                  'objects': [{'name': 'a', 'label': 1, 'box': [1, 1, 10, 10]},
                              {'name': 'b', 'label': 2, 'box': [9, 9, 18, 18]}]}
    cropper = _VOCImageCropper(img, (10, 10), (10, 10), logger=logger)
    cropper.image_info = image_info
    cropper.update()
    results = cropper.result
    count = 0
    for result in results:
        res = result[0]
        if res['name'] == 'a' and res['label'] == 1:
            assert np.allclose(np.asarray(res['box']), np.array([1, 1, 9, 9]))
            logger.info('对象 {} 匹配成功!'.format(res['name']))
            count += 1
        if res['name'] == 'b' and res['label'] == 2:
            assert np.allclose(np.asarray(res['box']), np.array([0, 0, 8, 8]))
            logger.info('对象 {} 匹配成功!'.format(res['name']))
            count += 1
    assert count == 2


if __name__ == '__main__':
    pytest.main()
