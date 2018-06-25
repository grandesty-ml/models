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
import pytest
import tensorflow as tf
from dltools.data.dataset import TEST_IMAGE1
from dltools.image.imaging import central_crop
from dltools.image.imaging_tf import central_crop as central_crop_tf


def test_central_crop():
    image = cv2.imread(str(TEST_IMAGE1))
    shape = image.shape
    assert 3 == len(shape), '这不是RGB图像！'

    cropped_image = central_crop(image, (100, 100))

    tf.reset_default_graph()
    tf.set_random_seed(1234)
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
