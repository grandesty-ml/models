#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: liang kang
@contact: gangkanli1219@gmail.com
@time: 2018/3/9 21:53
@desc: 数组图像处理的方法
"""
import cv2
import numpy as np
from skimage import exposure, transform


def central_crop(image, crop_shape):
    """
    中心裁剪图像

    Parameters
    ----------
    image: a 3-D image array [height, width, channels]
    crop_shape

    Returns
    -------

    """
    shape = image.shape
    assert shape[0] >= crop_shape[0] and shape[1] >= crop_shape[1], \
        'Shape Error!'
    offset = list(map(lambda x, y: (x - y) // 2, shape, crop_shape))
    max_axis = list(map(lambda x, y: x + y, offset, crop_shape))
    sub_image = image[offset[0]:max_axis[0], offset[1]:max_axis[1], :]
    return sub_image


def rotate_image(image, angle):
    """
    逆时针旋转一个图像 angle 度

    Parameters
    ----------
    angle: 要旋转的角度
    image: 原始图像, a 3-D image array [height, width, channels]

    Returns
    -------
    img_buf: 旋转后的图像

    """
    dtype = image.dtype
    if 0 == angle:
        img_buf = image.copy()
    elif 0 == angle % 90:
        if angle > 0:
            img_buf = np.rot90(image, int(angle / 90))
        else:
            k = int(angle / 90) % 4 + 4
            img_buf = np.rot90(image, k)
    else:
        img_buf = transform.rotate(image, angle,
                                   resize=True, preserve_range=True)
        img_buf = img_buf.astype(dtype)
    return img_buf


def histogram_equalization(image):
    """
    实现直方图均衡化
    Parameters
    ----------
    image: 原始图像, a 3-D image array [height, width, channels]

    Returns
    -------

    """
    shape = image.shape
    if 1 == shape[2]:
        img = cv2.equalizeHist(image[:, :, 0])
    else:
        img0 = np.expand_dims(cv2.equalizeHist(image[:, :, 0]), axis=2)
        img1 = np.expand_dims(cv2.equalizeHist(image[:, :, 1]), axis=2)
        img2 = np.expand_dims(cv2.equalizeHist(image[:, :, 2]), axis=2)
        img = np.concatenate([img0, img1, img2], axis=2)
    return img
