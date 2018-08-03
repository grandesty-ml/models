#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: liang kang
@contact: gangkanli1219@gmail.com
@time: 2018/5/10 21:16
@desc: 
"""
from abc import abstractmethod

import cv2
import numpy as np
import pytest
from dltools.data.dataset.cropper.iter_cropper import IterCropper
from dltools.data.dataset import TEST_IMAGE1


class _IterCropper(IterCropper):

    @abstractmethod
    def _update(self):
        pass


def test_iter_image_cropper():
    cropper = _IterCropper(str(TEST_IMAGE1), (100, 100), (100, 100))
    assert isinstance(cropper.image, np.ndarray)
    assert cropper.update()
    for _, img in cropper:
        assert isinstance(img, np.ndarray)
        assert img.shape[0] == img.shape[1] == 100

    img = cv2.imread(str(TEST_IMAGE1))
    img = cv2.resize(img, (200, 200))
    cropper = _IterCropper(img, (100, 100), (100, 100))
    assert isinstance(cropper.image, np.ndarray)
    assert cropper.update()
    for _, img in cropper:
        assert isinstance(img, np.ndarray)
        assert img.shape[0] == img.shape[1] == 100
        assert img.shape[0] % 100 == 0
        assert img.shape[1] % 100 == 0


if __name__ == '__main__':
    pytest.main()
