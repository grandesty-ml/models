#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: liang kang
@contact: gangkanli1219@gmail.com
@time: 2018/3/19 13:12
@desc: 
"""
import tensorflow as tf


def central_crop(image, crop_height, crop_width):
    """
    对输入图像进行中心裁剪

    Parameters
    ----------
    image: a 3-D image tensor [height, width, channels]
    crop_height: 裁剪后的高
    crop_width: 裁剪后的宽

    Returns
    -------
    3-D tensor with cropped image
    """
    shape = tf.shape(image)
    height, width = shape[0], shape[1]

    amount_to_be_cropped_h = (height - crop_height)
    crop_top = amount_to_be_cropped_h // 2
    amount_to_be_cropped_w = (width - crop_width)
    crop_left = amount_to_be_cropped_w // 2
    image = tf.slice(image, [crop_top, crop_left, 0],
                     [crop_height, crop_width, -1])
    return image
