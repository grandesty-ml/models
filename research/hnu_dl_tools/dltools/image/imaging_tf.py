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


def histogram_equalization(image):
    """
    使用 tensorflow 实现直方图均衡化
    Parameters
    ----------
    image: [height, width, channels],

    Returns
    -------

    """
    shape = tf.shape(image)
    
    def _equalize_histogram(img):
        values_range = tf.constant([0., 255.], dtype=tf.float32)
        histogram = tf.histogram_fixed_width(
            tf.to_float(img), values_range, 256)
        cdf = tf.cumsum(histogram)
        cdf_min = cdf[tf.reduce_min(tf.where(tf.greater(cdf, 0)))]
        img_shape = tf.shape(img)
        pix_cnt = img_shape[0] * img_shape[1]
        px_map = tf.round(
            tf.to_float(cdf - cdf_min) * 255. / tf.to_float(pix_cnt - 1))
        px_map = tf.cast(px_map, image.dtype)
        eq_hist = tf.gather_nd(px_map, tf.cast(img, tf.int32))
        return eq_hist

    channels = tf.split(image, 3, axis=2)
    eq_channels = tf.map_fn(_equalize_histogram,
                            tf.convert_to_tensor(channels))
    image = tf.transpose(eq_channels, [1, 2, 0])
    return tf.reshape(image, shape)
