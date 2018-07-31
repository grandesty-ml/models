#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: liang kang
@contact: gangkanli1219@gmail.com
@time: 2018/3/28 20:00
@desc: 
"""
from dltools.image.convert import *


def test_image_encode_base64():
    image = cv2.imread("../../../object_detection/test_images/image1.jpg")
    shape = image.shape
    assert 3 == len(shape), '这不是RGB图像！'

    string = image_encode_base64(image)
    assert isinstance(string, str)
    assert len(string) < np.prod(shape)


def test_image_decode_base64():
    image = cv2.imread("../../../object_detection/test_images/image1.jpg")
    shape = image.shape
    assert 3 == len(shape), '这不是RGB图像！'
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    string = image_encode_base64(image)
    image_out = image_decode_base64(string)
    sub = np.abs(image.astype(np.float32) - image_out.astype(np.float32))
    assert 2 >= np.sum(sub) / np.prod(shape)
