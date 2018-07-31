#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: liang kang
@contact: gangkanli1219@gmail.com
@time: 2018/3/17 13:35
@desc: 
"""
import numpy as np

from dltools.utils.basic import is_rectangle_overlap
from dltools.utils.basic import num_samples


def get_overlap_area(rect1, rect2):
    """
    计算两个矩形的重叠部分的面积，如果不重叠返回 0

    Parameters
    ----------
    rect1: 矩形1，形如[axis1_min, axis2_min, axis1_max, axis2_max]
    rect2: 矩形2，形如[axis1_min, axis2_min, axis1_max, axis2_max]

    Returns
    -------

    """
    if not is_rectangle_overlap(rect1, rect2):
        return 0.0
    xmin = max(rect1[0], rect2[0])
    ymin = max(rect1[1], rect2[1])
    xmax = min(rect1[2], rect2[2])
    ymax = min(rect1[3], rect2[3])
    return (xmax - xmin) * (ymax - ymin)


def get_point_from_box(boxes, weights=(0.5, 0.5)):
    """
    从一系列类似矩形的`box`中获取点

    Parameters
    ----------
    boxes: array-like, 一系列类似
           [axis1_min, axis2_min, axis1_max, axis2_max] 的`box`
    weights: list or tuple, axis1_min 和 axis2_min的权值

    Returns
    -------
    points: array-like, 点, shape = (number, 2)
    """
    boxes = np.asarray(boxes)
    weights = np.asarray(weights).reshape([1, 2])
    assert np.min(weights) < 1.0, '比例过大！'
    assert np.max(weights) > 0.0, '比例过小！'
    assert np.sum(weights) == 1.0, '各个比例之和应为 1！'
    return boxes[:, (0, 1)] * weights + (1.0 - weights) * boxes[:, (2, 3)]


def get_revolve_info(angle, shape,
                     center=(0.5, 0.5)):
    """
    获取旋转矩阵，旋转前或旋转后的图像形状

    Parameters
    ----------
    angle: int or float, 旋转角度， 大于零时逆时针旋转
    shape: list or tuple,
           旋转前或旋转后的图像形状, 当 `reverse` 为 `False`，为旋转前
    center: list or tuple, 旋转原点的位置

    Returns
    -------
    旋转矩阵， 旋转前或旋转后的图像形状
    """
    arc = angle * np.pi / 180
    _shape = (
        np.ceil(abs(shape[0] * np.cos(arc)) +
                abs(shape[1] * np.sin(arc))).astype(np.int32),
        np.ceil(abs(shape[0] * np.sin(arc)) +
                abs(shape[1] * np.cos(arc))).astype(np.int32))
    revolve_mat_1 = np.matrix([[1, 0, 0],
                               [0, -1, 0],
                               [-center[1] * shape[1],
                                center[0] * shape[0], 1]])
    revolve_mat_2 = np.matrix([[np.cos(arc), -np.sin(arc), 0],
                               [np.sin(arc), np.cos(arc), 0],
                               [0, 0, 1]])
    revolve_mat_3 = np.matrix([[1, 0, 0],
                               [0, -1, 0],
                               [center[1] * _shape[1],
                                center[0] * _shape[0], 1]])
    return revolve_mat_1 * revolve_mat_2 * revolve_mat_3, np.asarray(_shape)


def revolve_points(points, shape, angle):
    """
    将 points 旋转到目标角度, 默认逆时针旋转

    Parameters
    ----------
    points: array-like, 将要旋转的点。y 坐标在前，x 坐标在后, shape = (number, 2)
    shape: list or tuple, 点所在的图像 shape
    angle: 旋转角度

    Returns
    -------
    points: array-like, 将要旋转的点。y 坐标在前，x 坐标在后, shape = (number, 2),
            如果输入的点是相对坐标返回的也是相对坐标，
            如果输入的点是绝对坐标返回的也是绝对坐标
    """
    points = np.asarray(points)
    assert 2 == points.shape[1], 'Points\' shape should be (number, 2)'
    mat, out_shape = get_revolve_info(angle, shape)
    relative_axis = np.max(points) <= 1.0
    if relative_axis:
        points *= np.asarray(shape).reshape([1, 2])
    points = np.concatenate(
        [points[:, (1, 0)], np.ones([num_samples(points), 1])], axis=1)
    points = np.asarray(np.dot(points, mat))
    _points = points[:, (1, 0)]
    if relative_axis:
        _points /= np.reshape(out_shape, [1, 2])
    return _points


def revolve_boxes(boxes, src_shape, angle):
    """
    将 box 旋转到目标角度, 默认逆时针旋转

    Parameters
    ----------
    boxes: array-like, y 坐标在前，x 坐标在后, shape = (number, 4)
    angle: int or float, 旋转角度
    src_shape: list or tuple, 点所在的图像 shape

    Returns
    -------
    points: array-like, 将要旋转的点。y 坐标在前，x 坐标在后, shape = (number, 2)
    """
    boxes = np.asarray(boxes)
    assert 4 == boxes.shape[1], 'Boxes\' shape should be (number, 4)'
    points = boxes[:, (0, 1, 2, 1, 0, 3, 2, 3)]
    points = np.reshape(points, (-1, 2))
    points = revolve_points(points, src_shape[:2], angle)
    points = np.reshape(points, (-1, 8))
    axis_y = points[:, np.linspace(0, 6, 4, dtype=np.int32)]
    axis_x = points[:, np.linspace(1, 7, 4, dtype=np.int32)]
    return np.stack([np.min(axis_y, axis=1), np.min(axis_x, axis=1),
                     np.max(axis_y, axis=1), np.max(axis_x, axis=1)], axis=-1)


def change_points_shape(points, in_shape, out_shape, offset):
    """
    将一个图像尺寸（shape）中的归一化的 points 转换到另一个尺寸中

    Parameters
    ----------
    points: array-like, 将要旋转的点。y 坐标在前，x 坐标在后, shape = (number, 2)
    in_shape: list or tuple, 点所在的源图像的 shape
    out_shape: list or tuple, 点要转换到的目标图像的 shape
    offset: list or tuple, 目标图像的左上角，在源图像的位置

    Returns
    -------

    Raises
    -------
    Shape error
    """
    if in_shape[0] == out_shape[0] and in_shape[1] == out_shape[1]:
        return points
    elif in_shape[0] >= out_shape[0] and in_shape[1] >= out_shape[1]:
        assert (in_shape[0] >= out_shape[0] + offset[0]) and (
                in_shape[1] >= out_shape[1] + offset[1]), 'Shape error'
    elif in_shape[0] <= out_shape[0] and in_shape[1] <= out_shape[1]:
        assert (out_shape[0] >= in_shape[0] - offset[0]) and (
                out_shape[1] >= in_shape[1] - offset[1]), 'Shape error'
    else:
        raise ValueError('Shape error')
    points = np.asarray(points)
    assert 2 == points.shape[1], 'Points\' shape should be (number, 2)'
    in_shape = np.asarray(in_shape[:2])
    out_shape = np.asarray(out_shape[:2])
    offset = np.asarray(offset)
    return (points * in_shape - offset) / out_shape


def change_boxes_shape(boxes, in_shape, out_shape, offset):
    """
    将一个图像尺寸（shape）中的归一化 boxes 转换到另一个尺寸中

    Parameters
    ----------
    boxes: array-like, y 坐标在前，x 坐标在后, shape = (number, 4)
    in_shape: list or tuple, 点所在的源图像的 shape
    out_shape: list or tuple, 点要转换到的目标图像的 shape
    offset: list or tuple, 目标图像的左上角，在源图像的位置

    Returns
    -------

    Raises
    -------
    Shape error
    """
    boxes = np.asarray(boxes)
    assert 4 == boxes.shape[1], 'Boxes\' shape should be (number, 4)'
    points = np.reshape(boxes, (-1, 2))
    points = change_points_shape(points, in_shape, out_shape, offset)
    return np.reshape(points, (-1, 4))
