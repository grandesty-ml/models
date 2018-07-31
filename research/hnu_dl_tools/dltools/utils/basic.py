#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: liang kang
@contact: gangkanli1219@gmail.com
@time: 2018/3/9 21:15
@desc: 
"""
import os

import numpy as np


def num_samples(x):
    """
    返回类似数组的数据数量

    Parameters
    ----------
    x: array-like

    Returns
    -------

    """
    if not hasattr(x, '__len__') and not hasattr(x, 'shape'):
        if hasattr(x, '__array__'):
            x = np.asarray(x)
        else:
            raise TypeError("Expected sequence or array-like, got %s" %
                            type(x))
    if hasattr(x, 'shape'):
        if len(x.shape) == 0:
            raise TypeError("Singleton array %r cannot be considered"
                            " a valid collection." % x)
        return x.shape[0]
    else:
        return len(x)


def check_consistent_length(*arrays):
    """
    检查所有数组的第一维的长度是否相等

    Parameters
    ----------
    *arrays : list or tuple of input objects.
        Objects that will be checked for consistent length.
    """
    lengths = [num_samples(X) for X in arrays if X is not None]
    uniques = np.unique(lengths)
    if len(uniques) > 1:
        raise ValueError("Found input variables with inconsistent numbers of"
                         " samples: %r" % [int(l) for l in lengths])


def get_name_and_ext(file_name):
    """
    获取文件名，包含扩展名

    Parameters
    ----------
    file_name: 文件完整路径

    Returns
    -------

    """
    (_, temp_filename) = os.path.split(file_name)
    (shot_name, extension) = os.path.splitext(temp_filename)
    return shot_name, extension


def is_rectangle_overlap(rectangle1, rectangle2):
    """
    判断矩形是否存在重叠

    Parameters
    ----------
    rectangle1: list or tuple, 矩形1，形如
                [axis1_min, axis2_min, axis1_max, axis2_max]
    rectangle2: list or tuple, 矩形2，形如
                [axis1_min, axis2_min, axis1_max, axis2_max]

    Returns
    -------

    """
    return not ((rectangle1[0] >= rectangle2[2]) or
                (rectangle1[1] >= rectangle2[3]) or
                (rectangle1[2] <= rectangle2[0]) or
                (rectangle1[3] <= rectangle2[1]))


def is_in_rectangle(point, rectangle):
    """
    判断点是否在矩形内部

    Parameters
    ----------
    point: list or tuple, 点，形如[axis1, axis2]
    rectangle: list or tuple, 矩形，形如[axis1_min, axis2_min, axis1_max, axis2_max]

    Returns
    -------

    """
    return (rectangle[0] <= point[0] <= rectangle[2]) and \
           (rectangle[1] <= point[1] <= rectangle[3])


def is_in_polygon(point, polygon):
    """
    判断一个点是否在多边形内部

    Parameters
    ----------
    point: list or tuple, 点，形如 [axis1, axis2]
    polygon: array-like, 多边形，形如：
                         [[axis1_1, axis2_1],
                         [axis1_2, axis2_2],
                         [axis1_3, axis2_3]
                         ...]

    Returns
    -------

    """
    polygon = np.asarray(polygon)
    min_1 = np.min(polygon[:, 0])
    max_1 = np.max(polygon[:, 0])
    min_2 = np.min(polygon[:, 1])
    max_2 = np.max(polygon[:, 1])
    if not is_in_rectangle(point, [min_1, min_2, max_1, max_2]):
        return False
    flag = False
    last_point = num_samples(polygon) - 1
    for idx, vertex in enumerate(polygon):
        if (vertex[1] > point[1]) != (polygon[last_point][1] > point[1]):
            buf = polygon[last_point][0] - vertex[0]
            buf *= (point[1] - vertex[1])
            buf /= (polygon[last_point][1] - vertex[1])
            if point[0] < (vertex[0] + buf):
                flag = not flag
        last_point = idx
    return flag
