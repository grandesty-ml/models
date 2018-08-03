#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: liang kang
@contact: gangkanli1219@gmail.com
@time: 2018/4/4 18:22
@desc: 
"""
from pathlib import Path

from dltools.data.dataset.cropper.iter_cropper import IterCropper
from dltools.data.dataset.cropper.image_cropper import BaseCropper
from dltools.data.dataset.cropper.rect_cropper import RectCropper
from dltools.data.dataset.cropper.voc_cropper import VOCImageCropper

__path = Path(__file__)
__path = Path(__path.parent.parent.parent.parent.parent.parent)
TEST_IMAGE1 = __path / 'object_detection' / 'test_images' / 'image1.jpg'
TEST_IMAGE2 = __path / 'object_detection' / 'test_images' / 'image2.jpg'

__all__ = ['IterCropper',
           'BaseCropper',
           'RectCropper',
           'VOCImageCropper',
           'TEST_IMAGE1',
           'TEST_IMAGE2']