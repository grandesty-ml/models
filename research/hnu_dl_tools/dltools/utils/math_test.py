#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: liang kang
@contact: gangkanli1219@gmail.com
@time: 2018/3/28 17:12
@desc: 
"""
from dltools.utils.math import *


def test_change_points_shape():
    points_1 = [[0.4, 0.4],
                [0.4, 0.5],
                [0.5, 0.6],
                [0.4, 0.6]]
    points_2 = [[0.2, 0.2],
                [0.2, 0.4],
                [0.4, 0.6],
                [0.2, 0.6]]
    src_shape = (1000, 1000)
    dst_shape = (500, 500)
    offset = (300, 300)
    _offset = (-300, -300)
    out_boxes_1 = change_points_shape(points_1, src_shape, dst_shape, offset)
    assert np.allclose(out_boxes_1, np.asarray(points_2))

    out_boxes_2 = change_points_shape(points_2, dst_shape, src_shape, _offset)
    assert np.allclose(out_boxes_2, np.asarray(points_1))


def test_change_boxes_shape():
    boxes = [[0.4, 0.4, 0.5, 0.5],
             [0.5, 0.5, 0.6, 0.6]]
    dst_boxes = [[0.2, 0.2, 0.4, 0.4],
                 [0.4, 0.4, 0.6, 0.6]]
    src_shape = (1000, 1000)
    dst_shape = (500, 500)
    offset = (300, 300)
    _offset = (-300, -300)

    out_boxes_1 = change_boxes_shape(boxes, src_shape, dst_shape, offset)
    assert np.allclose(out_boxes_1, np.asarray(dst_boxes))

    out_boxes_2 = change_boxes_shape(dst_boxes, dst_shape, src_shape, _offset)
    assert np.allclose(out_boxes_2, np.asarray(boxes))


def test_get_overlap_area():
    rect1 = [0.4, 0.4, 0.6, 0.6]
    rect2 = [0.5, 0.5, 0.7, 0.7]
    rect3 = [0.2, 0.2, 0.3, 0.3]

    area1 = get_overlap_area(rect1, rect2)
    assert np.allclose(area1, 0.01)

    area2 = get_overlap_area(rect1, rect3)
    assert np.allclose(area2, 0.0)


def test_get_point_from_box():
    boxes = [[0.4, 0.4, 0.5, 0.5],
             [0.5, 0.5, 0.6, 0.6]]
    points1 = [[0.45, 0.45],
               [0.55, 0.55]]
    weights = (0.8, 0.2)
    points2 = [[0.42, 0.48],
               [0.52, 0.58]]

    point_out1 = get_point_from_box(boxes)
    assert np.allclose(np.asarray(points1), np.asarray(point_out1))

    point_out2 = get_point_from_box(boxes, weights)
    assert np.allclose(np.asarray(points2), np.asarray(point_out2))


def test_get_revolve_info():
    shape = (400, 500)
    angle = 20

    shape_ = (547, 607)
    mat = np.array([0.939692620785908, 0.342020143325669, 0,
                    -0.342020143325669, 0.939692620785908, 0,
                    136.980873468657, 0.0564400114011505, 1])
    # mat_r = np.array([0.939692620785908, -0.342020143325669, 0,
    #                   0.342020143325669, 0.939692620785908, 0,
    #                   -128.739219608094, 46.7971817143945, 1])
    mat_out, shape_out = get_revolve_info(angle, shape)
    assert np.allclose(mat, np.asarray(mat_out).flatten())
    assert shape_[0] == shape_out[0] and shape_[1] == shape_out[1]


def test_revolve_points():
    shape = (500, 500)
    angle = 90
    points_1 = [[0.4, 0.4],
                [0.4, 0.5],
                [0.5, 0.6],
                [0.4, 0.6]]
    points_2 = [[0.6, 0.4],
                [0.5, 0.4],
                [0.4, 0.5],
                [0.4, 0.4]]
    points_out = revolve_points(points_1, shape, -angle)
    assert np.allclose(np.asarray(points_out),
                       np.asarray(points_2), atol=3.e-2)

    points_out = revolve_points(points_2, shape, angle)
    assert np.allclose(np.asarray(points_1),
                       np.asarray(points_out), atol=3.e-2)


def test_revolve_boxes():
    shape = (500, 500)
    angle = 90
    boxes = [[0.4, 0.4, 0.5, 0.5],
             [0.5, 0.5, 0.6, 0.6]]
    boxes_gt = [[0.4, 0.5, 0.5, 0.6],
                [0.5, 0.4, 0.6, 0.5]]
    boxes_out = revolve_boxes(boxes, shape, angle)
    assert np.allclose(np.asarray(boxes_gt), np.asarray(boxes_out), atol=3.e-2)

    boxes_out = revolve_boxes(boxes_gt, shape, -angle)
    assert np.allclose(np.asarray(boxes), np.asarray(boxes_out), atol=3.e-2)
