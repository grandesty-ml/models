#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: liang kang
@contact: gangkanli1219@gmail.com
@time: 2018/3/27 19:43
@desc: 
"""
import os

import cv2
import numpy as np
import tensorflow as tf
import pytest

from dltools.image.imaging import central_crop, histogram_equalization
from dltools.image.imaging_tf import central_crop as central_crop_tf
from dltools.image.imaging_tf import \
    histogram_equalization as histogram_equalization_tf


def test_histogram_equalization():
    image = cv2.imread("../../../object_detection/test_images/image1.jpg")
    shape = image.shape
    assert 3 == len(shape), '这不是RGB图像！'

    img = histogram_equalization(image)

    image_input = tf.placeholder(tf.uint8, shape=[None, None, 3])
    array_img = np.arange(256).reshape([16, 16, 1])
    array_img = np.concatenate([array_img, array_img, array_img], axis=2)
    image_eq = histogram_equalization_tf(image_input)
    with tf.Session() as sess:
        tf.global_variables_initializer()
        image_eq_hist = sess.run(image_eq, feed_dict={image_input: image})
        sub_image = img - image_eq_hist
        assert 1 >= np.sum(np.abs(sub_image)) / np.prod(shape), '显示相差大于 1 ！'

        array_eq = sess.run(image_eq, feed_dict={image_input: array_img})
        np.testing.assert_allclose(array_img, array_eq)


def test_central_crop():
    image = cv2.imread("../../../object_detection/test_images/image1.jpg")
    shape = image.shape
    assert 3 == len(shape), '这不是RGB图像！'

    cropped_image = central_crop(image, (100, 100))

    image_input = tf.placeholder(tf.uint8, shape=[None, None, 3])
    image_crop = central_crop_tf(image_input, 100, 100)
    with tf.Session() as sess:
        tf.global_variables_initializer()
        cropped_image_tf = sess.run(image_crop, feed_dict={image_input: image})
        sub_image = cropped_image - cropped_image_tf
        assert 0 == np.sum(np.abs(sub_image)), '剪切错误！！'


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'
    pytest.main()
