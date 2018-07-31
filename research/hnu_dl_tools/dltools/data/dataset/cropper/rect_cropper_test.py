#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: liang kang
@contact: gangkanli1219@gmail.com
@time: 2018/5/10 21:23
@desc: 
"""
from abc import abstractmethod

import numpy as np
import pytest
from dltools.data.dataset.cropper.rect_cropper import RectCropper


class _RectCropper(RectCropper):

    def __init__(self, image_path, crop_size, stride):
        """

        Parameters
        ----------
        image_path: 图像路径
        crop_size: 子图尺寸
        stride: 切割步长
        """
        super(_RectCropper, self).__init__(
            image_path, crop_size, stride,
            logger=None, is_write=False)

    @abstractmethod
    def _crop_image(self):
        pass

    @property
    def seat_x(self):
        return self._seat_x

    @property
    def seat_y(self):
        return self._seat_y


def test_rect_cropper():
    seat1 = np.linspace(0, 90, 10, dtype=np.int32).tolist()
    seat1 += [95]

    img = np.ones([105, 105, 3])
    cropper1 = _RectCropper(img, (10, 10), (10, 10))
    assert isinstance(cropper1.image, np.ndarray)
    assert cropper1.update()
    assert np.allclose(seat1, np.asarray(cropper1.seat_x))
    assert np.allclose(seat1, np.asarray(cropper1.seat_y))


if __name__ == '__main__':
    pytest.main()
