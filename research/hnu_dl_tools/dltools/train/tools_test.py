#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: liang kang
@contact: gangkanli1219@gmail.com
@time: 2018/3/27 22:17
@desc: 
"""
from dltools.train.tools import *


def test_merge_bounding_boxes():
    boxes = [[0.1, 0.1, 0.5, 0.5],
             [0.15, 0.15, 0.5, 0.5],
             [0.45, 0.45, 0.9, 0.9],
             [0.4, 0.45, 0.8, 0.85]]
    classes = [1, 0, 1, 1]
    scores = [0.8, 0.7, 0.9, 0.65]

    new_boxes, new_scores, new_classes = (
        merge_bounding_boxes(boxes, scores, classes))

    assert np.allclose(new_boxes, np.array([[0.45, 0.45, 0.9, 0.9],
                                            [0.1, 0.1, 0.5, 0.5],
                                            [0.15, 0.15, 0.5, 0.5]]))
    assert np.allclose(new_scores, np.array([0.9, 0.8, 0.7]))
    assert np.allclose(new_classes, np.array([1, 1, 0]))
