#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: liang kang
@contact: gangkanli1219@gmail.com
@time: 2018/3/28 19:40
@desc: 
"""
from dltools.image.imaging import *


def test_central_crop():
    img = np.arange(16).reshape((4, 4, 1))
    img_out = central_crop(img, (2, 2))
    assert np.allclose(img_out, img[1:3, 1:3])


def test_rotate_image():
    img = np.arange(256).reshape((16, 16))
    img_out = rotate_image(img, 90)
    assert np.allclose(np.rot90(img), img_out)


def test_histogram_equalization():
    img = np.arange(256, dtype=np.uint8).reshape((16, 16, 1))
    img = np.concatenate([img, img, img], axis=2)
    img_out = histogram_equalization(img)
    assert np.allclose(img, img_out)

    img = np.arange(0, 256, 5, dtype=np.uint8).reshape((4, 13, 1))
    img = np.concatenate([img, img, img], axis=2)
    img_out = histogram_equalization(img)
    assert np.allclose(img, img_out)
