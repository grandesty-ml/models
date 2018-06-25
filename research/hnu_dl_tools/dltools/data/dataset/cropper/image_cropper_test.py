#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: liang kang
@contact: gangkanli1219@gmail.com
@time: 2018/5/7 10:49
@desc:
"""
from abc import abstractmethod

import numpy as np
import pytest
from dltools.data.dataset.cropper.image_cropper import BaseCropper
from dltools.data.dataset import TEST_IMAGE1


class _BaseCropper(BaseCropper):

    @abstractmethod
    def _set_up(self):
        pass

    @abstractmethod
    def _update(self):
        pass


def test_base_cropper():
    cropper1 = _BaseCropper('', None)
    assert cropper1.image is None
    assert not cropper1.update()

    cropper2 = _BaseCropper(str(TEST_IMAGE1), None)
    assert isinstance(cropper2.image, np.ndarray)
    assert cropper2.update()

    img = np.ones([3, 100, 100, 1])
    cropper3 = _BaseCropper(img, None)
    assert isinstance(cropper3.image, np.ndarray)
    assert cropper3.image.shape[2] == 3
    assert cropper3.update()


if __name__ == '__main__':
    pytest.main()
